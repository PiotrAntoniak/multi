"""In-language CEILING for FLORES targets -- is the "0.9 x English-self" F1 target feasible?

Question
--------
``flores_calib`` measures how close ONE English-trained XGBoost gets to English self under
label-free calibration, and reports target/EN F1 ratios well below 0.9.  That raises a prior
question: is the 0.9 target even *achievable* for a target language, i.e. can that language reach
the target with its OWN model?  This script bounds it from above.

Method (same protocol as ``flores_calib`` / ``proof_mean_alllangs``)
-------------------------------------------------------------------
* EuroBERT-210m **mean** embeddings, ``perlang_z`` feature transform (L2-normalize, then z-score
  with each language's OWN **train** rows); applied both sides.
* ONE XGBoost per (dataset, language): ``multi:softprob``, ``num_class`` 12 (sent) / 11 (agg),
  ``tree_method='hist'``, ``max_bin=64``, ``device='cpu'``, ``n_jobs=2``, ``seed=0``,
  ``max_depth=8``, ``min_child_weight=5``, ``learning_rate=0.1``, ``n_estimators=400``.
* For each language L in {en, it, de, fr}: train on L's **train+val** rows -> predict L's **test**
  rows.  Reported metric = **in-language ceiling** (micro-F1, macro-F1, OVR-macro AUC) -- the best
  that language can do with its own model under this representation/protocol.

Comparisons
-----------
* ``target(EN-trained, calibrated) / target(in-language ceiling)``: the EN-transfer F1 (best
  label-free calibration from ``flores_calib.json``) as a fraction of what the target could reach
  with its own model.  Ratio -> 1 means transfer already matches the target's own ceiling.
* **Feasibility of "0.9 x English-self"**: 0.9 x (English in-language self) is compared to each
  target's own ceiling.  If ``0.9 * EN_self > target_ceiling`` then the target CANNOT reach the 0.9
  target even with a perfect EN->target transfer: the target's own best model is below the bar, so
  the 0.9 target is **INFEASIBLE** regardless of calibration.

Reads
-----
``flores_calib.json`` (the EN-trained calibrated target F1 + the EN baseline self).  Datasets via
``train_flores12`` (``prepare_sentence`` / ``prepare_agg`` + mean embeddings).

Outputs (create-only): ``flores_ceiling.py`` / ``flores_ceiling.json`` / ``flores_ceiling.md``.
Operational pid/log/status live under ``%TEMP%\\opencode\\flores_ceiling``.

Usage
-----
    python flores_ceiling.py            # foreground (4 models/dataset; ~3 min on this CPU box)
    python flores_ceiling.py --launch   # detached hidden run via launch_hidden.vbs
    python flores_ceiling.py --smoke    # tiny plumbing check (60 trees)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import json
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import flores_calib as C                       # noqa: E402  (build + perlang_z + metrics + XGB_COMMON)
import train_flores12 as T                     # noqa: E402

OUT_JSON = os.path.join(REPO, "flores_ceiling.json")
OUT_MD = os.path.join(REPO, "flores_ceiling.md")
CALIB_JSON = os.path.join(REPO, "flores_calib.json")

TMP = os.path.join(os.environ.get("TEMP", REPO), "opencode", "flores_ceiling")
os.makedirs(TMP, exist_ok=True)
LOG_PATH = os.path.join(TMP, "flores_ceiling.log")
PID_PATH = os.path.join(TMP, "flores_ceiling.pid")
STATUS_PATH = os.path.join(TMP, "flores_ceiling_status.json")

LANGS = ["en", "it", "de", "fr"]
TARGETS = ["it", "de", "fr"]
DATASETS = ["sent", "agg"]
THRESH = 0.9

_T0 = time.time()
_SMOKE = False


def log(msg):
    line = "[%7.1fs] %s" % (time.time() - _T0, msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(obj):
    obj = dict(obj)
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, STATUS_PATH)


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % (type(o),))


# ---------------------------------------------------------------------------
# Source: EN-trained calibrated target F1 from flores_calib.json
# ---------------------------------------------------------------------------
def load_calib_source():
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        doc = json.load(f)
    src = {"file": os.path.basename(CALIB_JSON), "datasets": {}}
    for ds in DATASETS:
        dsv = doc["datasets"][ds]
        summ = dsv["summary"]
        method = summ["best_method"]
        per_lang = dsv["per_lang"]
        en_baseline = per_lang["en"]["baseline"]
        targets = {}
        for lg in TARGETS:
            targets[lg] = {k: per_lang[lg][method][k] for k in ("micro_f1", "macro_f1", "acc", "auc")}
        src["datasets"][ds] = {
            "best_label_free_method": method,
            "best_min_ratio_both": summ["best_min_ratio_both"],
            "en_baseline_self": {k: en_baseline[k] for k in ("micro_f1", "macro_f1", "acc", "auc")},
            "en_trained_calibrated_target": targets,
        }
    return src


# ---------------------------------------------------------------------------
# In-language ceiling: train on L train+val -> predict L test
# ---------------------------------------------------------------------------
def ceiling_dataset(D):
    import xgboost as xgb
    name, Cc = D["name"], D["n_classes"]
    y, tr, va, te = D["y"], D["train"], D["val"], D["test"]
    log("==== DATASET %s (C=%d, train=%d val=%d test=%d) ====" % (name, Cc, len(tr), len(va), len(te)))

    Xt = C.perlang_z(D["X"], tr)          # stats from TRAIN rows only (same as flores_calib)
    trva = np.concatenate([tr, va])       # in-language ceiling trains on train+val

    p = dict(C.XGB_COMMON)
    p["num_class"] = Cc
    if _SMOKE:
        p["n_estimators"] = 60

    ceil = {}
    for lg in LANGS:
        t0 = time.time()
        clf = xgb.XGBClassifier(**p)
        clf.fit(Xt[lg][trva], y[trva])
        P = clf.predict_proba(Xt[lg][te]).astype(np.float64)
        ceil[lg] = C.metrics(y[te], P, Cc)
        log("  [%s] ceiling micro=%.4f macro=%.4f auc=%.4f (%.1fs)"
            % (lg, ceil[lg]["micro_f1"], ceil[lg]["macro_f1"], ceil[lg]["auc"], time.time() - t0))

    return {"name": name, "n_classes": Cc, "classes": D["classes"],
            "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                      "train_val": int(len(trva))},
            "ceiling": ceil}


# ---------------------------------------------------------------------------
# Analysis: ratios + feasibility of "0.9 x English-self"
# ---------------------------------------------------------------------------
def analyze(name, cds, calib_ds):
    """Return per-target ratios (calibrated/ceiling) and the 0.9-feasibility verdict."""
    ceil = cds["ceiling"]
    en = ceil["en"]
    cal = calib_ds["en_trained_calibrated_target"]
    en_base = calib_ds["en_baseline_self"]

    # Two candidate "English-self" references:
    #  * ceiling protocol (train+val) -- apples-to-apples with the target ceilings
    #  * flores_calib baseline (train only) -- the reference used by flores_calib's ratios
    en_self = {
        "en_ceiling_trainval": {"micro_f1": en["micro_f1"], "macro_f1": en["macro_f1"], "auc": en["auc"]},
        "en_calib_baseline_trainonly": {"micro_f1": en_base["micro_f1"], "macro_f1": en_base["macro_f1"],
                                        "auc": en_base["auc"]},
    }
    bar_ceiling = {k: THRESH * en[k] for k in ("micro_f1", "macro_f1")}
    bar_calib = {k: THRESH * en_base[k] for k in ("micro_f1", "macro_f1")}

    per_target = {}
    for lg in TARGETS:
        c = ceil[lg]
        cl = cal[lg]
        rec = {
            "ceiling": {"micro_f1": c["micro_f1"], "macro_f1": c["macro_f1"], "auc": c["auc"]},
            "en_trained_calibrated": {"micro_f1": cl["micro_f1"], "macro_f1": cl["macro_f1"],
                                      "auc": cl["auc"]},
            "ratio_calibrated_over_ceiling": {
                "micro": cl["micro_f1"] / c["micro_f1"] if c["micro_f1"] else None,
                "macro": cl["macro_f1"] / c["macro_f1"] if c["macro_f1"] else None,
                "auc": cl["auc"] / c["auc"] if c["auc"] else None,
            },
            "feasibility_0.9": {},
        }
        for ref, bar in (("en_ceiling_trainval", bar_ceiling),
                         ("en_calib_baseline_trainonly", bar_calib)):
            fm = bool(bar["micro_f1"] > c["micro_f1"])
            fa = bool(bar["macro_f1"] > c["macro_f1"])
            rec["feasibility_0.9"][ref] = {
                "bar_0.9x_en_self": {"micro_f1": float(bar["micro_f1"]), "macro_f1": float(bar["macro_f1"])},
                "target_ceiling": {"micro_f1": c["micro_f1"], "macro_f1": c["macro_f1"]},
                "infeasible_micro": fm,
                "infeasible_macro": fa,
                "margin_micro": float(bar["micro_f1"] - c["micro_f1"]),
                "margin_macro": float(bar["macro_f1"] - c["macro_f1"]),
                "verdict": ("INFEASIBLE (0.9*EN-self exceeds target's own ceiling on both micro "
                            "and macro)" if (fm and fa) else
                            "INFEASIBLE on micro only" if fm else
                            "INFEASIBLE on macro only" if fa else
                            "feasible in principle (target ceiling >= 0.9*EN-self)"),
            }
        per_target[lg] = rec

    return {"name": name, "en_self": en_self, "bar_0.9": {"en_ceiling_trainval": bar_ceiling,
                                                          "en_calib_baseline_trainonly": bar_calib},
            "per_target": per_target}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def build_md(src, datasets, analysis):
    md = []
    md.append("# FLORES in-language ceilings -- is \"0.9 x English-self\" feasible?\n")
    md.append("Same protocol as `flores_calib` / `proof_mean_alllangs`: EuroBERT-210m **mean**, "
              "`perlang_z` (L2-normalize + per-language *train* z-score), one XGBoost per "
              "(dataset, language) (softprob, depth 8 / mcw 5 / lr 0.1 / n 400, hist, max_bin 64, "
              "cpu, n_jobs 2, seed 0), trained on the language's **train+val** rows and evaluated "
              "on its **test** rows. This is each language's own ceiling (best it can do with its "
              "OWN model).\n")
    for ds in DATASETS:
        cds = datasets[ds]
        an = analysis[ds]
        calib_ds = src["datasets"][ds]
        md.append("\n## %s (%d classes, train+val=%d / test=%d)\n"
                  % (ds, cds["n_classes"], cds["sizes"]["train_val"], cds["sizes"]["test"]))
        md.append("### In-language ceilings\n")
        md.append("| lang | micro-F1 | macro-F1 | AUC |")
        md.append("|---|---:|---:|---:|")
        for lg in LANGS:
            c = cds["ceiling"][lg]
            md.append("| %s%s | %.4f | %.4f | %.4f |"
                      % (lg, " (EN self)" if lg == "en" else "", c["micro_f1"], c["macro_f1"], c["auc"]))
        md.append("")
        md.append("### EN-trained calibrated transfer vs ceiling (best label-free = `%s`)\n"
                  % calib_ds["best_label_free_method"])
        md.append("| target | cal micro | ceiling micro | ratio micro | cal macro | ceiling macro | ratio macro |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for lg in TARGETS:
            r = an["per_target"][lg]
            md.append("| %s | %.4f | %.4f | **%.3f** | %.4f | %.4f | **%.3f** |"
                      % (lg, r["en_trained_calibrated"]["micro_f1"], r["ceiling"]["micro_f1"],
                         r["ratio_calibrated_over_ceiling"]["micro"],
                         r["en_trained_calibrated"]["macro_f1"], r["ceiling"]["macro_f1"],
                         r["ratio_calibrated_over_ceiling"]["macro"]))
        md.append("")
        en_c = cds["ceiling"]["en"]
        en_b = calib_ds["en_baseline_self"]
        md.append("English self under the SAME (train+val) protocol: micro **%.4f**, macro **%.4f** "
                  "(0.9x -> micro %.4f, macro %.4f). For reference, the `flores_calib` train-only "
                  "EN baseline self is micro %.4f / macro %.4f (0.9x -> micro %.4f, macro %.4f).\n"
                  % (en_c["micro_f1"], en_c["macro_f1"], an["bar_0.9"]["en_ceiling_trainval"]["micro_f1"],
                     an["bar_0.9"]["en_ceiling_trainval"]["macro_f1"],
                     en_b["micro_f1"], en_b["macro_f1"],
                     an["bar_0.9"]["en_calib_baseline_trainonly"]["micro_f1"],
                     an["bar_0.9"]["en_calib_baseline_trainonly"]["macro_f1"]))
        md.append("### Feasibility of \"0.9 x English-self\" (vs each target's own ceiling)\n")
        md.append("| target | 0.9xEN-self micro | target ceil micro | infeasible? | 0.9xEN-self macro | "
                  "target ceil macro | infeasible? |")
        md.append("|---|---:|---:|:--:|---:|---:|:--:|")
        for lg in TARGETS:
            f = an["per_target"][lg]["feasibility_0.9"]["en_ceiling_trainval"]
            md.append("| %s | %.4f | %.4f | %s | %.4f | %.4f | %s |"
                      % (lg, f["bar_0.9x_en_self"]["micro_f1"], f["target_ceiling"]["micro_f1"],
                         ("**YES**" if f["infeasible_micro"] else "no"),
                         f["bar_0.9x_en_self"]["macro_f1"], f["target_ceiling"]["macro_f1"],
                         ("**YES**" if f["infeasible_macro"] else "no")))
        md.append("")
        for lg in TARGETS:
            f = an["per_target"][lg]["feasibility_0.9"]["en_ceiling_trainval"]
            fc = an["per_target"][lg]["feasibility_0.9"]["en_calib_baseline_trainonly"]
            md.append("- **%s**: %s. (micro margin 0.9*EN-self - ceiling = %+.4f; macro = %+.4f). "
                      "Using the flores_calib train-only EN self the verdict is: %s."
                      % (lg, f["verdict"], f["margin_micro"], f["margin_macro"], fc["verdict"]))
        md.append("")
        md.append("Note: a positive margin (0.9*EN-self > target ceiling) means the target's own "
                  "best model is notionally **below the 0.9 bar**, so no EN->target transfer can "
                  "reach it. A ratio `calibrated/ceiling` near 1 means the English model already "
                  "matches the target's own ceiling (little headroom left).\n")

    md.append("\n## Verdict\n")
    for ds in DATASETS:
        an = analysis[ds]
        inf = [lg for lg in TARGETS
               if an["per_target"][lg]["feasibility_0.9"]["en_ceiling_trainval"]["infeasible_micro"]
               or an["per_target"][lg]["feasibility_0.9"]["en_ceiling_trainval"]["infeasible_macro"]]
        ratios = {lg: an["per_target"][lg]["ratio_calibrated_over_ceiling"]["micro"] for lg in TARGETS}
        md.append("- **%s**: infeasible-on-some-metric targets (0.9 x EN train+val self) = %s; "
                  "calibrated/ceiling micro ratios = %s."
                  % (ds, ", ".join(inf) or "NONE",
                     ", ".join("%s %.3f" % (lg, ratios[lg]) for lg in TARGETS)))
    md.append("")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# Launch / main
# ---------------------------------------------------------------------------
def launch(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    script = os.path.join(REPO, os.path.basename(__file__))
    command = "%s %s %s" % (sys.executable, script, " ".join(passthrough))
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, command, LOG_PATH, REPO])
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    global _SMOKE
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="FLORES in-language ceilings vs 0.9x English-self.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args(argv)

    if args.launch:
        launch([a for a in argv if a != "--launch"])
        return 0

    _SMOKE = args.smoke
    global OUT_JSON, OUT_MD
    if _SMOKE:
        OUT_JSON = os.path.join(TMP, "flores_ceiling_smoke.json")
        OUT_MD = os.path.join(TMP, "flores_ceiling_smoke.md")

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    write_status({"phase": "start", "pid": os.getpid(), "smoke": _SMOKE})

    src = load_calib_source()
    log("loaded %s: best methods %s" % (src["file"],
        {ds: src["datasets"][ds]["best_label_free_method"] for ds in DATASETS}))

    datasets = {}
    analysis = {}
    for ds in DATASETS:
        write_status({"phase": "build", "dataset": ds})
        D = C.build(ds)
        write_status({"phase": "ceiling", "dataset": ds})
        cds = ceiling_dataset(D)
        datasets[ds] = cds
        analysis[ds] = analyze(ds, cds, src["datasets"][ds])

    payload = {
        "config": {"emb": "EuroBERT-210m mean", "feature": "perlang_z", "xgb": C.XGB_COMMON,
                   "langs": LANGS, "targets": TARGETS, "threshold": THRESH,
                   "ceiling_train": "train+val -> self test", "seed": 0, "smoke": _SMOKE,
                   "python": sys.version.split()[0]},
        "source_flores_calib": src,
        "datasets": datasets,
        "analysis": analysis,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_jd)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(build_md(src, datasets, analysis))
    write_status({"phase": "done", "smoke": _SMOKE})
    log("wrote %s , %s" % (OUT_JSON, OUT_MD))
    for ds in DATASETS:
        an = analysis[ds]
        log("%s: 0.9xEN-self micro=%.4f macro=%.4f | " % (
            ds, an["bar_0.9"]["en_ceiling_trainval"]["micro_f1"],
            an["bar_0.9"]["en_ceiling_trainval"]["macro_f1"]) +
            " | ".join("%s ceil micro=%.4f ratio_cal=%.3f" % (
                lg, an["per_target"][lg]["ceiling"]["micro_f1"],
                an["per_target"][lg]["ratio_calibrated_over_ceiling"]["micro"]) for lg in TARGETS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
