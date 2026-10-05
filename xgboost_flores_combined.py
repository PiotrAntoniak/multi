"""FLORES en/it XGBoost follow-up: (C) combined en+it training, (D) en -> it +D transfer.

(C) train on en+it combined (dev+devtest, 3006 rows) -> evaluate en test and it test;
(D) train en (all 2009) -> evaluate it raw vs it+D, with D = mean(emb_en - emb_it) over all rows.

Metrics: subset accuracy / macro-F1 / micro-F1 at fixed 0.5 AND per-tag thresholds tuned
in-sample on the training rows. Baseline params (max_depth=6, lr=0.1, n_estimators=300),
lenient 8-tag targets. Writes xgboost_flores_combined.md (+ .json).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
import time

import numpy as np

import xgboost_topic_optuna as X

BASE = os.path.dirname(os.path.abspath(__file__))
MODES = ["mean", "bos"]


def main():
    t0 = time.time()
    df, Y, masks, counts = X.load_data("lenient")
    pos = counts["all"]
    dev_dt, test = masks["dev_devtest"], masks["test"]
    params = X.make_params(X.BASELINE_PARAMS)
    rows = []

    def evaluate(mode, label, Xtr, Ytr, evals):
        """Fit on Xtr, return fixed-0.5 and tuned-threshold score dicts for each eval set."""
        tprobs, probs_list, _ = X.fit_predict_with_train(params, Xtr, Ytr, [e[1] for e in evals])
        ths = X.tune_thresholds(Ytr, tprobs)
        out = []
        for (name, Xte, Yte) in evals:
            probs = probs_list[len(out)]
            sf = X.score(Yte, probs, pos_counts=pos)
            st = X.score(Yte, probs, ths, pos_counts=pos)
            out.append((name, sf, st))
        return out

    print("== (C) combined en+it training ==")
    for mode in MODES:
        E = X.load_one_embedding(mode, "en")
        I = X.load_one_embedding(mode, "it")
        Xtr = np.vstack([E[dev_dt], I[dev_dt]])
        Ytr = np.vstack([Y[dev_dt], Y[dev_dt]])
        res = evaluate(mode, "en+it", Xtr, Ytr,
                       [("en test", E[test], Y[test]), ("it test", I[test], Y[test])])
        for name, sf, st in res:
            print(f"[C] {mode} {name}: fixed subset={sf['subset_acc']:.4f} macro={sf['macro_f1']:.4f} "
                  f"micro={sf['micro_f1']:.4f} | tuned subset={st['subset_acc']:.4f} "
                  f"macro={st['macro_f1']:.4f} micro={st['micro_f1']:.4f}")
            rows.append(dict(part="C", mode=mode, eval=name, fixed_macro=sf["macro_f1"],
                             fixed_subset=sf["subset_acc"], fixed_micro=sf["micro_f1"],
                             tuned_macro=st["macro_f1"], tuned_subset=st["subset_acc"],
                             tuned_micro=st["micro_f1"]))
        # in-sample reference
        Xall = np.vstack([E, I])
        tp, pl, _ = X.fit_predict_with_train(params, Xall, np.vstack([Y, Y]), [Xall])
        si = X.score(np.vstack([Y, Y]), pl[0], pos_counts=pos)
        print(f"[C] {mode} en+it in-sample: macro={si['macro_f1']:.4f}")
        rows.append(dict(part="C", mode=mode, eval="en+it in-sample", fixed_macro=si["macro_f1"],
                         fixed_subset=si["subset_acc"], fixed_micro=si["micro_f1"],
                         tuned_macro=None, tuned_subset=None, tuned_micro=None))

    print("== (D) en -> it, raw vs +D ==")
    for mode in MODES:
        E = X.load_one_embedding(mode, "en")
        I = X.load_one_embedding(mode, "it")
        D = (E - I).mean(axis=0)
        res = evaluate(mode, "en->it", E, Y,
                       [("en in-sample", E, Y), ("it raw", I, Y), ("it+D", I + D, Y)])
        for name, sf, st in res:
            print(f"[D] {mode} {name}: fixed subset={sf['subset_acc']:.4f} macro={sf['macro_f1']:.4f} "
                  f"micro={sf['micro_f1']:.4f} | tuned subset={st['subset_acc']:.4f} "
                  f"macro={st['macro_f1']:.4f} micro={st['micro_f1']:.4f}")
            rows.append(dict(part="D", mode=mode, eval=name, fixed_macro=sf["macro_f1"],
                             fixed_subset=sf["subset_acc"], fixed_micro=sf["micro_f1"],
                             tuned_macro=st["macro_f1"], tuned_subset=st["subset_acc"],
                             tuned_micro=st["micro_f1"]))

    lines = ["# FLORES en/it follow-up: combined training + shifted transfer\n"]
    lines.append("Baseline params, lenient 8-tag targets; fixed 0.5 and in-sample tuned thresholds.\n")
    lines.append("## (C) train en+it (dev+devtest) -> test\n")
    lines.append("| mode | eval | fixed subset | fixed macro-F1 | fixed micro-F1 | tuned subset | tuned macro-F1 | tuned micro-F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["part"] == "C":
            fm = f"{r['tuned_macro']:.4f}" if r["tuned_macro"] is not None else "-"
            fs = f"{r['tuned_subset']:.4f}" if r["tuned_subset"] is not None else "-"
            fu = f"{r['tuned_micro']:.4f}" if r["tuned_micro"] is not None else "-"
            lines.append(f"| {r['mode']} | {r['eval']} | {r['fixed_subset']:.4f} | {r['fixed_macro']:.4f} | "
                         f"{r['fixed_micro']:.4f} | {fs} | {fm} | {fu} |")
    lines.append("\n## (D) train en (all 2009) -> it raw vs it+D\n")
    lines.append("| mode | eval | fixed subset | fixed macro-F1 | fixed micro-F1 | tuned subset | tuned macro-F1 | tuned micro-F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["part"] == "D":
            lines.append(f"| {r['mode']} | {r['eval']} | {r['fixed_subset']:.4f} | {r['fixed_macro']:.4f} | "
                         f"{r['fixed_micro']:.4f} | {r['tuned_subset']:.4f} | {r['tuned_macro']:.4f} | "
                         f"{r['tuned_micro']:.4f} |")
    with open(os.path.join(BASE, "xgboost_flores_combined.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE, "xgboost_flores_combined.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\ndone in {time.time() - t0:.1f}s -> xgboost_flores_combined.md")


if __name__ == "__main__":
    main()
