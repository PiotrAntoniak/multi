"""Dropped vs kept language dims: is the model's SHAP importance concentrated on the
dimensions that carry the 4-language signal?

Trains the English baseline (dev+devtest, 1503 rows) with baseline params and explains the
506-row held-out test set, once with en test inputs and once with it test inputs (the model is
never trained on all 2009 rows and never evaluated in-sample).  For every mode we rank dims by
the 4-language per-dim eta^2 (one-way ANOVA over en/it/de/fr), take the top-k dims that carry
25%/50% of the eta^2 mass ("dropped") and the rest ("kept"), then compare mean |SHAP| over the
two dim groups, and the dropped/kept ratio.

Writes pca_plots/shap_dropped_vs_kept_{mode}.png and shap_flores_dropped.md (+ .json).
"""
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
import xgboost_topic_optuna as X

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "pca_plots")
MODES = ["mean", "bos"]
INPUT_LANGS = ["en", "it"]  # test-input languages we explain
FRACTIONS = [(0.25, "25%"), (0.50, "50%")]


def eta2_lang(mode):
    """Per-dimension eta^2 of a one-way ANOVA over the 4 FLORES languages."""
    E = {l: X.load_one_embedding(mode, l) for l in X.LANGS}
    Xall = np.vstack([E[l] for l in X.LANGS])
    y = np.repeat(X.LANGS, len(E["en"]))
    overall = Xall.mean(0)
    ss_tot = ((Xall - overall) ** 2).sum(0)
    ss_bet = np.zeros(Xall.shape[1])
    for l in set(y):
        m = y == l
        ss_bet += m.sum() * (Xall[m].mean(0) - overall) ** 2
    return ss_bet / ss_tot


def mean_abs_shap(models, Xte):
    """Per-dim mean |SHAP| over rows, averaged across the 8 tag models."""
    imp = np.zeros(Xte.shape[1])
    for model in models:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(Xte, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else np.mean([np.abs(s) for s in sv], axis=0)
        imp += np.abs(sv).mean(0)
    return imp / len(models)


def main():
    df, Y, masks, counts = X.load_data("lenient")
    dev_dt, test = masks["dev_devtest"], masks["test"]
    params = X.make_params(X.BASELINE_PARAMS)
    t_start = time.time()
    rows = []

    for mode in MODES:
        t0 = time.time()
        # Held-out model: train on English dev+devtest only.
        Xen = X.load_one_embedding(mode, "en")
        Xtr, Ytr = Xen[dev_dt], Y[dev_dt]
        Xte = {l: X.load_one_embedding(mode, l)[test] for l in INPUT_LANGS}
        (probs,), models = X.fit_predict(params, Xtr, Ytr, [Xte["en"]])

        # Mean |SHAP| per dim for both input sets (en test, it test).
        imp = {l: mean_abs_shap(models, Xte[l]) for l in INPUT_LANGS}

        # 4-language eta^2 ranking + top-k dims for 25%/50% mass.
        e2 = eta2_lang(mode)
        order = np.argsort(-e2)
        cum = np.cumsum(e2[order]) / e2.sum()
        ks = {frac: int(np.searchsorted(cum, frac) + 1) for frac, _ in FRACTIONS}

        for frac, label in FRACTIONS:
            k = ks[frac]
            dropped, kept = order[:k], order[k:]
            row = dict(mode=mode, frac=frac, label=label, k=k, n_kept=len(kept))
            for l in INPUT_LANGS:
                d = float(imp[l][dropped].mean())
                kp = float(imp[l][kept].mean())
                row[l] = dict(dropped=d, kept=kp, ratio=d / kp)
            rows.append(row)
            print(f"[{mode} {label}] k={k} dims "
                  + " | ".join(f"{l}: dropped={row[l]['dropped']:.5f} "
                               f"kept={row[l]['kept']:.5f} ratio={row[l]['ratio']:.3f}"
                               for l in INPUT_LANGS))

        print(f"  {mode} done in {time.time() - t0:.1f}s")

        # Bar plot: dropped vs kept mean |SHAP| for en-test and it-test, one panel per k.
        fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(10, 4.2), sharey=True)
        for ax, (frac, label) in zip(axes, FRACTIONS):
            k = ks[frac]
            x = np.arange(len(INPUT_LANGS))
            w = 0.36
            dv = [row[l]["dropped"] for row in rows if row["mode"] == mode and row["frac"] == frac
                  for l in INPUT_LANGS]
            kv = [row[l]["kept"] for row in rows if row["mode"] == mode and row["frac"] == frac
                  for l in INPUT_LANGS]
            ax.bar(x - w / 2, dv, width=w, label="dropped (top-k eta^2)")
            ax.bar(x + w / 2, kv, width=w, label="kept")
            ax.set_xticks(x, [f"{l}-test" for l in INPUT_LANGS])
            ax.set_title(f"{mode}: top {k} dims ({label} of lang eta^2)", fontsize=9)
            ax.set_ylabel("mean |SHAP|")
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(OUT, f"shap_dropped_vs_kept_{mode}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("  wrote", p)

    # Markdown + JSON.
    lines = ["# Dropped vs kept language dims — SHAP (FLORES baseline, en)\n"]
    lines.append("Baseline params, train en dev+devtest (1503), explain test (506); mean |SHAP| over "
                 "8 tag models; dropped = top-k dims by 4-language per-dim eta^2 (one-way ANOVA over "
                 "en/it/de/fr), kept = the rest. Cells are mean |SHAP| and the dropped/kept ratio.\n")
    lines.append("| mode | mass | k dims | en dropped | en kept | en ratio | it dropped | it kept | it ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['mode']} | {r['label']} | {r['k']} | "
                     f"{r['en']['dropped']:.5f} | {r['en']['kept']:.5f} | {r['en']['ratio']:.3f} | "
                     f"{r['it']['dropped']:.5f} | {r['it']['kept']:.5f} | {r['it']['ratio']:.3f} |")
    lines.append("")
    with open(os.path.join(BASE, "shap_flores_dropped.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE, "shap_flores_dropped.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("wrote shap_flores_dropped.md/.json")
    print(f"total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
