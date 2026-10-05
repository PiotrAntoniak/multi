"""FLORES en/it XGBoost experiments (topic classification, lenient labels, baseline params).

(A) per-language reference: train en / it (mean and bos), on the standard split (dev+devtest
    -> test) and on all 2009 rows (in-sample);
(B) language-dim ablation: train on English with the top-k most language-separable dims zeroed
    (k = dims covering 25% / 50% of the 4-language eta^2 mass, ranked by per-dim one-way ANOVA)
    and evaluate on Italian (raw features, same dims zeroed). Baseline params for all fits.

Writes xgboost_flores_lang.md (+ .json) with subset accuracy, macro-F1, micro-F1.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
import time

import numpy as np

import xgboost_topic_optuna as X
from lang_metrics import top_dims

BASE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "it"]
MODES = ["mean", "bos"]
DROPS = [0.0, 0.25, 0.5]


def main():
    t0 = time.time()
    _, Y, masks, counts = X.load_data("lenient")
    pos = counts["all"]
    allm = np.ones(len(Y), dtype=bool)
    dev_dt, test = masks["dev_devtest"], masks["test"]
    params = X.make_params(X.BASELINE_PARAMS)

    def fmt(m):
        return (m["subset_acc"], m["macro_f1"], m["micro_f1"])

    rows = []
    print("== (A) per-language reference (baseline params, fixed 0.5) ==")
    for lang in LANGS:
        for mode in MODES:
            Xf = X.load_one_embedding(mode, lang)
            t = time.time()
            (probs_test,), _ = X.fit_predict(params, Xf[dev_dt], Y[dev_dt], [Xf[test]])
            m_test = X.score(Y[test], probs_test, pos_counts=pos)
            (probs_all,), _ = X.fit_predict(params, Xf[allm], Y[allm], [Xf[allm]])
            m_all = X.score(Y[allm], probs_all, pos_counts=pos)
            rows.append(dict(part="A", lang=lang, mode=mode, split="test",
                             subset=fmt(m_test)[0], macro=fmt(m_test)[1], micro=fmt(m_test)[2]))
            rows.append(dict(part="A", lang=lang, mode=mode, split="all-insample",
                             subset=fmt(m_all)[0], macro=fmt(m_all)[1], micro=fmt(m_all)[2]))
            print(f"[A] {lang}/{mode}: test subset={m_test['subset_acc']:.4f} "
                  f"macro={m_test['macro_f1']:.4f} micro={m_test['micro_f1']:.4f} | "
                  f"all-in-sample macro={m_all['macro_f1']:.4f} ({time.time() - t:.1f}s)")

    print("== (B) train en (all 2009) -> eval it (all 2009), language dims zeroed ==")
    for mode in MODES:
        Xe = X.load_one_embedding(mode, "en")
        Xi = X.load_one_embedding(mode, "it")
        for frac in DROPS:
            dims, k = top_dims(mode, frac) if frac > 0 else (np.array([], dtype=int), 0)
            Xe2, Xi2 = Xe.copy(), Xi.copy()
            if k:
                Xe2[:, dims] = 0.0
                Xi2[:, dims] = 0.0
            t = time.time()
            probs_en, probs_it = X.fit_predict(params, Xe2, Y, [Xe2, Xi2])[0]
            m_en = X.score(Y, probs_en, pos_counts=pos)
            m_it = X.score(Y, probs_it, pos_counts=pos)
            rows.append(dict(part="B", lang="en->it", mode=mode, drop=frac, k=k,
                             en_subset=m_en["subset_acc"], en_macro=m_en["macro_f1"],
                             it_subset=m_it["subset_acc"], it_macro=m_it["macro_f1"],
                             it_micro=m_it["micro_f1"]))
            print(f"[B] {mode} drop={frac:.2f} (k={k}): en macro={m_en['macro_f1']:.4f} | "
                  f"it subset={m_it['subset_acc']:.4f} macro={m_it['macro_f1']:.4f} "
                  f"micro={m_it['micro_f1']:.4f} ({time.time() - t:.1f}s)")

    lines = ["# XGBoost on FLORES en/it — per-language reference + language-dim ablation\n"]
    lines.append("Baseline params (max_depth=6, lr=0.1, n_estimators=300), fixed 0.5 threshold, "
                 "lenient 8-tag targets, EuroBERT-210m embeddings.\n")
    lines.append("## (A) per-language reference\n")
    lines.append("| lang | mode | split | subset acc | macro-F1 | micro-F1 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in rows:
        if r["part"] == "A":
            lines.append(f"| {r['lang']} | {r['mode']} | {r['split']} | {r['subset']:.4f} | "
                         f"{r['macro']:.4f} | {r['micro']:.4f} |")
    lines.append("\n## (B) train en (all 2009) -> eval it (all 2009), top language dims zeroed\n")
    lines.append("| mode | drop | k dims | en macro-F1 | it subset acc | it macro-F1 | it micro-F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["part"] == "B":
            lines.append(f"| {r['mode']} | {r['drop']:.0%} | {r['k']} | {r['en_macro']:.4f} | "
                         f"{r['it_subset']:.4f} | {r['it_macro']:.4f} | {r['it_micro']:.4f} |")
    with open(os.path.join(BASE, "xgboost_flores_lang.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE, "xgboost_flores_lang.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\ndone in {time.time() - t0:.1f}s -> xgboost_flores_lang.md")


if __name__ == "__main__":
    main()
