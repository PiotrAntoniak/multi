"""SHAP on the unrestricted FLORES baseline (en, mean/bos): are the language-separable dims
(per-dim eta^2 across the 4 languages) the ones driving topic predictions?

Trains the baseline (dev+devtest, 1503 rows) and explains the 506-row test set with
shap.TreeExplainer, one explainer per tag; aggregates mean |SHAP| per dim and compares it
against the 4-language eta^2 ranking (Spearman, top-50 overlap, SHAP mass on the dims that
carry 25%/50% of the language signal vs the random-expectation share).

Writes pca_plots/shap_vs_langdims_{mode}.png and shap_flores_lang.md (+ .json).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import shap
import xgboost_topic_optuna as X

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "pca_plots")
MODES = ["mean", "bos"]


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


def main():
    df, Y, masks, counts = X.load_data("lenient")
    dev_dt, test = masks["dev_devtest"], masks["test"]
    params = X.make_params(X.BASELINE_PARAMS)
    rows = []

    for mode in MODES:
        Xf = X.load_one_embedding(mode, "en")
        Xtr, Ytr, Xte = Xf[dev_dt], Y[dev_dt], Xf[test]
        t0 = time.time()
        (probs,), models = X.fit_predict(params, Xtr, Ytr, [Xte])

        imp = np.zeros(Xf.shape[1])
        for model in models:
            ex = shap.TreeExplainer(model)
            sv = ex.shap_values(Xte, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) == 2 else np.mean([np.abs(s) for s in sv], axis=0)
            imp += np.abs(sv).mean(0)
        imp /= len(models)

        e2 = eta2_lang(mode)
        order = np.argsort(-e2)
        cum = np.cumsum(e2[order]) / e2.sum()
        k25 = int(np.searchsorted(cum, 0.25) + 1)
        k50 = int(np.searchsorted(cum, 0.50) + 1)

        rho, _ = spearmanr(imp, e2)
        top_shap, top_e2 = set(np.argsort(-imp)[:50]), set(order[:50])
        overlap = len(top_shap & top_e2)
        mass25 = float(imp[order[:k25]].sum() / imp.sum())
        mass50 = float(imp[order[:k50]].sum() / imp.sum())
        exp25, exp50 = k25 / len(imp), k50 / len(imp)
        # SHAP rank of the top-5 language dims (1 = most important)
        rank_of = np.empty(len(imp), dtype=int)
        rank_of[np.argsort(-imp)] = np.arange(1, len(imp) + 1)
        lang5 = [(int(d), float(e2[d]), int(rank_of[d])) for d in order[:5]]
        top10 = [(int(d), float(imp[d])) for d in np.argsort(-imp)[:10]]

        rows.append(dict(mode=mode, spearman=float(rho), top50_overlap=int(overlap),
                         k25=k25, mass25=mass25, exp25=exp25,
                         k50=k50, mass50=mass50, exp50=exp50,
                         lang5_ranks=lang5, top10_shap=top10,
                         time_s=round(time.time() - t0, 1)))
        print(f"[{mode}] spearman={rho:.3f} top50_overlap={overlap}/50 | "
              f"SHAP mass on lang dims: 25%->k={k25} {mass25:.3f} (random {exp25:.3f}), "
              f"50%->k={k50} {mass50:.3f} (random {exp50:.3f}) | {time.time() - t0:.1f}s")
        print(f"   top-5 lang dims (dim, SHAP rank): {lang5}")

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(e2, imp, s=6, alpha=0.4)
        axes[0].set_xlabel("per-dim eta^2 (language separability)")
        axes[0].set_ylabel("mean |SHAP| (topic)")
        axes[0].set_title(f"{mode}: SHAP vs language eta^2  (Spearman {rho:.3f})", fontsize=9)
        x = np.arange(2)
        axes[1].bar(x - 0.18, [mass25, mass50], width=0.36, label="SHAP mass")
        axes[1].bar(x + 0.18, [exp25, exp50], width=0.36, label="random expectation")
        axes[1].set_xticks(x, [f"top {k25} dims\n(25% of lang mass)", f"top {k50} dims\n(50% of lang mass)"])
        axes[1].set_ylabel("fraction of total SHAP importance")
        axes[1].set_title(f"{mode}: does the model use language dims?", fontsize=9)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"shap_vs_langdims_{mode}.png"), dpi=150)
        plt.close(fig)

    lines = ["# SHAP vs language dims — unrestricted FLORES baseline (en)\n"]
    lines.append("Baseline params, train dev+devtest (1503), explain test (506); mean |SHAP| over 8 tag "
                 "models; eta^2 = per-dim one-way ANOVA over en/it/de/fr.\n")
    lines.append("| mode | Spearman(SHAP, eta^2) | top-50 overlap | SHAP mass on 25%-lang dims | random | "
                 "SHAP mass on 50%-lang dims | random |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['mode']} | {r['spearman']:.3f} | {r['top50_overlap']}/50 | "
                     f"{r['mass25']:.3f} (k={r['k25']}) | {r['exp25']:.3f} | "
                     f"{r['mass50']:.3f} (k={r['k50']}) | {r['exp50']:.3f} |")
    for r in rows:
        lines.append(f"\n## {r['mode']}: top-5 language dims by SHAP rank\n")
        lines.append("| dim | eta^2 | SHAP rank |")
        lines.append("|---:|---:|---:|")
        for d, e, rk in r["lang5_ranks"]:
            lines.append(f"| {d} | {e:.3f} | {rk} |")
    with open(os.path.join(BASE, "shap_flores_lang.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE, "shap_flores_lang.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("wrote shap_flores_lang.md/.json + pca_plots/shap_vs_langdims_{mode}.png")


if __name__ == "__main__":
    main()
