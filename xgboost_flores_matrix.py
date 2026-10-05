"""FLORES en/it full-row evaluation matrix (lenient labels, baseline params).

(E) train on all 2009 rows of en and of it (mean and bos); evaluate each model on en and it.
(F) train on all 2009 en rows with the top language-separable dims zeroed (25% / 50% of the
4-language eta^2 mass) and evaluate on it raw (no +D: the language offset is removed by
construction).

Metrics at fixed 0.5 and per-tag thresholds tuned in-sample on the model's training rows.
Writes xgboost_flores_matrix.md (+ .json).
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
MODES = ["mean", "bos"]
DROPS = [0.25, 0.50]


def main():
    t0 = time.time()
    _, Y, masks, counts = X.load_data("lenient")
    pos = counts["all"]
    params = X.make_params(X.BASELINE_PARAMS)
    rows = []

    def fit_eval(Xtr, Ytr, evals):
        tprobs, probs_list, _ = X.fit_predict_with_train(params, Xtr, Ytr, [e[1] for e in evals])
        ths = X.tune_thresholds(Ytr, tprobs)
        out = []
        for (name, Xte, Yte) in evals:
            p = probs_list[len(out)]
            out.append((name, X.score(Yte, p, pos_counts=pos),
                        X.score(Yte, p, ths, pos_counts=pos)))
        return out

    print("== (E) full-row models: train on all 2009 rows ==")
    for mode in MODES:
        E = X.load_one_embedding(mode, "en")
        I = X.load_one_embedding(mode, "it")
        for tr, Xtr, other, Xother in (("en", E, "it", I), ("it", I, "en", E)):
            res = fit_eval(Xtr, Y, [(f"{tr} in-sample", Xtr, Y), (f"{other} raw", Xother, Y)])
            for name, sf, st in res:
                print(f"[E] train={tr} {mode} eval={name}: fixed subset={sf['subset_acc']:.4f} "
                      f"macro={sf['macro_f1']:.4f} micro={sf['micro_f1']:.4f} | tuned "
                      f"subset={st['subset_acc']:.4f} macro={st['macro_f1']:.4f} "
                      f"micro={st['micro_f1']:.4f}")
                rows.append(dict(part="E", train=tr, mode=mode, eval=name,
                                 fixed_subset=sf["subset_acc"], fixed_macro=sf["macro_f1"],
                                 fixed_micro=sf["micro_f1"], tuned_subset=st["subset_acc"],
                                 tuned_macro=st["macro_f1"], tuned_micro=st["micro_f1"]))

    print("== (F) train en without language dims -> eval it raw ==")
    for mode in MODES:
        E = X.load_one_embedding(mode, "en")
        I = X.load_one_embedding(mode, "it")
        for frac in DROPS:
            dims, k = top_dims(mode, frac)
            E2, I2 = E.copy(), I.copy()
            E2[:, dims] = 0.0
            I2[:, dims] = 0.0
            res = fit_eval(E2, Y, [("en in-sample", E2, Y), ("it raw", I2, Y)])
            for name, sf, st in res:
                print(f"[F] {mode} drop={frac:.0%} (k={k}) eval={name}: fixed subset="
                      f"{sf['subset_acc']:.4f} macro={sf['macro_f1']:.4f} micro={sf['micro_f1']:.4f} "
                      f"| tuned subset={st['subset_acc']:.4f} macro={st['macro_f1']:.4f} "
                      f"micro={st['micro_f1']:.4f}")
                rows.append(dict(part="F", mode=mode, drop=frac, k=k, eval=name,
                                 fixed_subset=sf["subset_acc"], fixed_macro=sf["macro_f1"],
                                 fixed_micro=sf["micro_f1"], tuned_subset=st["subset_acc"],
                                 tuned_macro=st["macro_f1"], tuned_micro=st["micro_f1"]))

    lines = ["# FLORES en/it full-row matrix\n"]
    lines.append("Baseline params, all 2009 rows, lenient 8-tag targets; fixed 0.5 and in-sample "
                 "tuned thresholds.\n")
    lines.append("## (E) full models, cross-evaluated\n")
    lines.append("| train | mode | eval | fixed subset | fixed macro-F1 | fixed micro-F1 | "
                 "tuned subset | tuned macro-F1 | tuned micro-F1 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["part"] == "E":
            lines.append(f"| {r['train']} | {r['mode']} | {r['eval']} | {r['fixed_subset']:.4f} | "
                         f"{r['fixed_macro']:.4f} | {r['fixed_micro']:.4f} | {r['tuned_subset']:.4f} | "
                         f"{r['tuned_macro']:.4f} | {r['tuned_micro']:.4f} |")
    lines.append("\n## (F) train en (no language dims) -> it raw\n")
    lines.append("| mode | drop | k | eval | fixed subset | fixed macro-F1 | fixed micro-F1 | "
                 "tuned subset | tuned macro-F1 | tuned micro-F1 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["part"] == "F":
            lines.append(f"| {r['mode']} | {r['drop']:.0%} | {r['k']} | {r['eval']} | "
                         f"{r['fixed_subset']:.4f} | {r['fixed_macro']:.4f} | {r['fixed_micro']:.4f} | "
                         f"{r['tuned_subset']:.4f} | {r['tuned_macro']:.4f} | {r['tuned_micro']:.4f} |")
    with open(os.path.join(BASE, "xgboost_flores_matrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE, "xgboost_flores_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\ndone in {time.time() - t0:.1f}s -> xgboost_flores_matrix.md")


if __name__ == "__main__":
    main()
