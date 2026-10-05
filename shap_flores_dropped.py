"""Dropped vs kept language dims: is the model's SHAP importance concentrated on the
dimensions that carry the 4-language signal?

Trains the English baseline (dev+devtest, 1503 rows) with baseline params and explains the
506-row held-out test set, once with en test inputs and once with it test inputs (the model is
never trained on all 2009 rows and never evaluated in-sample).  For every mode we rank dims by
the 4-language per-dim eta^2 (one-way ANOVA over en/it/de/fr), take the top-k dims that carry
25%/50% of the eta^2 mass ("dropped") and the rest ("kept"), then compare the summed per-dim
mean |SHAP| over the two dim groups, and the dropped share of that sum.

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
from lang_metrics import eta2_lang

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "pca_plots")
MODES = ["mean", "bos"]
INPUT_LANGS = ["en", "it"]  # test-input languages we explain
FRACTIONS = [(0.25, "25%"), (0.50, "50%")]


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
    _, Y, masks, _ = X.load_data("lenient")
    dev_dt, test = masks["dev_devtest"], masks["test"]
    params = X.make_params(X.BASELINE_PARAMS)
    t_start = time.time()
    rows = []
    it_rows = []  # symmetric Italian-model cells (model=it, inputs=it-test)

    for mode in MODES:
        t0 = time.time()
        # Held-out model: train on English dev+devtest only.
        Xen = X.load_one_embedding(mode, "en")
        Xtr, Ytr = Xen[dev_dt], Y[dev_dt]
        Xte = {l: X.load_one_embedding(mode, l)[test] for l in INPUT_LANGS}
        (probs,), models = X.fit_predict(params, Xtr, Ytr, [Xte["en"]])

        # Mean |SHAP| per dim for both input sets (en test, it test).
        imp = {l: mean_abs_shap(models, Xte[l]) for l in INPUT_LANGS}

        # Symmetric Italian model: train on Italian dev+devtest, same params;
        # explain the IT-TEST inputs.
        Xit = X.load_one_embedding(mode, "it")
        (probs_it,), models_it = X.fit_predict(params, Xit[dev_dt], Ytr, [Xte["it"]])
        imp_it = mean_abs_shap(models_it, Xte["it"])

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
                ds = float(imp[l][dropped].sum())
                ksum = float(imp[l][kept].sum())
                row[l] = dict(dropped_sum=ds, kept_sum=ksum,
                              share_dropped=ds / (ds + ksum), random_share=k / 768)
            rows.append(row)

            # Symmetric Italian-model row (single input set: it-test).
            ds_it = float(imp_it[dropped].sum())
            ksum_it = float(imp_it[kept].sum())
            it_rows.append(dict(model="it", mode=mode, frac=frac, label=label, k=k,
                                n_kept=len(kept), inputs="it-test",
                                dropped_sum=ds_it, kept_sum=ksum_it,
                                share_dropped=ds_it / (ds_it + ksum_it), random_share=k / 768))

            print(f"[{mode} {label}] k={k} dims "
                  + " | ".join(f"{l}: dropped_sum={row[l]['dropped_sum']:.4f} "
                               f"kept_sum={row[l]['kept_sum']:.4f} "
                               f"share={row[l]['share_dropped']:.3f} "
                               f"rand={row[l]['random_share']:.3f}"
                               for l in INPUT_LANGS))
            print(f"  [model=it inputs=it-test] k={k} dims "
                  f"dropped_sum={ds_it:.4f} kept_sum={ksum_it:.4f} "
                  f"share={ds_it / (ds_it + ksum_it):.3f} rand={k / 768:.3f}")

        print(f"  {mode} done in {time.time() - t0:.1f}s")

        # Bar plot: en-model (en-test, it-test) + symmetric it-model (it-test), one panel per k.
        r_en = {r["frac"]: r for r in rows if r["mode"] == mode}
        r_it = {r["frac"]: r for r in it_rows if r["mode"] == mode}
        groups = [("en-model / en-test", "en"), ("en-model / it-test", "it")]
        fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(11.5, 4.2), sharey=True)
        for ax, (frac, label) in zip(axes, FRACTIONS):
            k = ks[frac]
            w = 0.36
            dv = [r_en[frac][l]["dropped_sum"] for _, l in groups] + [r_it[frac]["dropped_sum"]]
            kv = [r_en[frac][l]["kept_sum"] for _, l in groups] + [r_it[frac]["kept_sum"]]
            x = np.arange(len(dv))
            ax.bar(x - w / 2, dv, width=w, label="dropped (top-k eta^2)")
            ax.bar(x + w / 2, kv, width=w, label="kept")
            ax.set_xticks(x, [g[0].replace(" / ", "\n") for g in groups] + ["it-model\nit-test"],
                          fontsize=7)
            ax.set_title(f"{mode}: top {k} dims ({label} of lang eta^2)", fontsize=9)
            ax.set_ylabel("sum |SHAP|")
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(OUT, f"shap_dropped_vs_kept_{mode}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("  wrote", p)

    # Markdown + JSON: one unified table with all three model/input combinations.
    SETUP = {("en", "en-test"): "English model on English text",
             ("en", "it-test"): "English model on Italian text",
             ("it", "it-test"): "Italian model on Italian text"}
    final_rows = []
    for r, ir in zip(rows, it_rows):
        final_rows.append(dict(model="en", inputs="en-test",
                               setup=SETUP[("en", "en-test")], mode=r["mode"],
                               label=r["label"], frac=r["frac"], k=r["k"], n_kept=r["n_kept"],
                               dropped_sum=r["en"]["dropped_sum"], kept_sum=r["en"]["kept_sum"],
                               share_dropped=r["en"]["share_dropped"],
                               random_share=r["en"]["random_share"]))
        final_rows.append(dict(model="en", inputs="it-test",
                               setup=SETUP[("en", "it-test")], mode=r["mode"],
                               label=r["label"], frac=r["frac"], k=r["k"], n_kept=r["n_kept"],
                               dropped_sum=r["it"]["dropped_sum"], kept_sum=r["it"]["kept_sum"],
                               share_dropped=r["it"]["share_dropped"],
                               random_share=r["it"]["random_share"]))
        final_rows.append(dict(ir, setup=SETUP[("it", "it-test")]))

    lines = ["# Dropped vs kept language dims — SHAP (FLORES baseline)\n"]
    lines.append("Baseline params; train on dev+devtest (1503), explain test (506); per-dim mean |SHAP| "
                 "over 8 tag models; dropped = top-k dims by 4-language per-dim eta^2 (one-way ANOVA "
                 "over en/it/de/fr), kept = the rest. Cells are the SUM over each dim group of the "
                 "per-dim mean |SHAP| (total contribution), the dropped share of the summed |SHAP|, "
                 "and the random baseline share k/768.\n")
    lines.append("| model | inputs | setup | mode | drop set | k dims | sum dropped | sum kept | "
                 "share dropped | random share |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for fr in final_rows:
        lines.append(f"| {fr['model']} | {fr['inputs']} | {fr['setup']} | {fr['mode']} | "
                     f"{fr['label']} | {fr['k']} | {fr['dropped_sum']:.4f} | {fr['kept_sum']:.4f} | "
                     f"{fr['share_dropped']:.3f} | {fr['random_share']:.3f} |")
    lines.append("")
    with open(os.path.join(BASE, "shap_flores_dropped.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE, "shap_flores_dropped.json"), "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2)
    print("wrote shap_flores_dropped.md/.json")
    print(f"total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
