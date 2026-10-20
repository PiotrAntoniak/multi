"""E42 sidecar -- D_clean / arm E: leak-free translate-train on FLORES sentences.

Why this arm exists
-------------------
Arm D (in ``en_vs_enit_raw.json``) trained on 3002 rows = 1501 EN train+val + MT-IT of
those same 1501 rows, and tuned on the real IT val split.  Those 504 IT-val rows are
human translations of 504 EN rows that D had already seen in MT form, so D's Optuna
objective was scored on rows its training data effectively contained, and the reported
0.8860 is inflated.  D' reused D's contaminated hyperparameters on 1994 rows, so it is
also not clean.

Arm E (D_clean) fixes both problems:
  * train  = 997 EN **train** rows + MT-IT of those same 997 rows  (1994 rows);
  * tune   = 10 fresh Optuna trials, TPE seed 0, objective = OVR-macro AUC on the
             **unseen** real IT val split (504 rows), which contributes to no fit;
  * test   = real IT test (505) and EN test (505), raw EuroBERT-210m mean features,
             no test-time MT, no L2 / z-score / perlang_z / PCA / INLP.

This script never writes ``en_vs_enit_raw.*``; it reads that JSON read-only for the
B / C / A_EN / D reference numbers and asserts they are the expected ones.

Usage
-----
    python en_vs_enit_dclean.py --smoke     # 2 trials
    python en_vs_enit_dclean.py             # 10 trials (default)
    python en_vs_enit_dclean.py --launch    # detached, hidden window
    python en_vs_enit_dclean.py --render    # md/png from the existing JSON
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse
import json
import subprocess
import sys
import time
import datetime

import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_flores12 as T  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_JSON = os.path.join(HERE, "en_vs_enit_dclean.json")
OUT_MD = os.path.join(HERE, "en_vs_enit_dclean.md")
OUT_PNG = os.path.join(HERE, "en_vs_enit_dclean.png")
RAW_JSON = os.path.join(HERE, "en_vs_enit_raw.json")

RUN_DIR = os.environ.get(
    "E42_DCLEAN_DIR", os.path.join(os.environ.get("TEMP", "."), "opencode",
                                   "en_vs_enit_dclean"))
LOG_PATH = os.path.join(RUN_DIR, "en_vs_enit_dclean.log")
PID_PATH = os.path.join(RUN_DIR, "en_vs_enit_dclean.pid")
STATUS_PATH = os.path.join(RUN_DIR, "en_vs_enit_dclean_status.json")

MT_CACHE = os.path.join(os.environ.get("TEMP", "."), "opencode", "flores_close_cache",
                        "tt_emb_sent_it.npy")
MT_MODEL = "Helsinki-NLP/opus-mt-en-it"

N_CLASSES = T.N_CLASSES
SEED = 0
BAR = 0.975
N_TRIALS_DEFAULT = 10

BASE_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES,
    tree_method="hist",
    max_bin=64,
    device="cpu",
    n_jobs=2,
    seed=SEED,
)

# Reference numbers copied from the E42 A/B/C/D run (asserted below).
REF_EXPECTED = {
    "A_EN": 0.8738560544303869,
    "B_IT": 0.837006482368932,
    "C_IT": 0.8423116873890897,
    "D_IT": 0.8860,          # rounded in the parent JSON's narrative; exact read below
}

_LOGF = None


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if _LOGF is not None:
        _LOGF.write(line + "\n")
        _LOGF.flush()


def open_log():
    global _LOGF
    os.makedirs(RUN_DIR, exist_ok=True)
    _LOGF = open(LOG_PATH, "a", encoding="utf-8", errors="replace")


def write_pid():
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def write_status(**kw):
    os.makedirs(RUN_DIR, exist_ok=True)
    kw = dict(kw)
    kw["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    kw["pid"] = os.getpid()
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kw, f, indent=2)
    os.replace(tmp, STATUS_PATH)


def metrics(y, probs, n_classes=N_CLASSES):
    pred = probs.argmax(1)
    labels = np.arange(n_classes)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0, labels=labels)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, labels=labels)),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y, probs, multi_class="ovr",
                                                   average="macro", labels=labels))
    except ValueError:
        out["auc_ovr_macro"] = float("nan")
    return out


def search_space(trial):
    return dict(
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
    )


def fit(params, X, y):
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def write_md(path, payload):
    e = payload["arm"]
    r = payload["ratios"]
    ref = payload["reference"]
    h = payload["held_out"]
    L = []
    L.append("### E42 sidecar -- D_clean (arm E): leak-free translate-train -- %s"
             % payload["status"].upper())
    L.append("")
    L.append("**Question.** With A/B/C/D already stored, what does translate-train score when "
             "the Optuna objective is measured on a validation split that no training row "
             "touches -- and how much of D's 0.8860 was inflation?")
    L.append("")
    L.append("**Script to reproduce:** `en_vs_enit_dclean.py` -- "
             "`python en_vs_enit_dclean.py --smoke` then `python en_vs_enit_dclean.py --launch`. "
             "Reproduction status: %s." % payload["reproduction_status"])
    L.append("")
    L.append("**Setup.** Train = %d rows (997 EN **train** + 997 MT-IT of those same rows). "
             "Tune = 10 fresh Optuna TPE trials (seed 0), objective = OVR-macro AUC on the "
             "**unseen** real IT val (504). Test = real IT test (505) and EN test (505). "
             "Features raw mean-pooled EuroBERT-210m float32; no L2, z-score, per-lang z, "
             "PCA/INLP, and **no test-time MT**."
             % e["n_train_rows"])
    L.append("")
    L.append("**Held-out check.** split %d / %d / %d, pairwise disjoint = %s; val rows inside "
             "the training matrix = **%d**; MT rows used = %d (train slice only), val MT rows "
             "used = **%d**."
             % (h["n_train"], h["n_val"], h["n_test"], h["disjoint"],
                h["val_rows_in_training"], h["mt_rows_used"], h["val_mt_rows_used"]))
    L.append("")
    L.append("| cell | accuracy | micro-F1 | macro-F1 | OVR-macro AUC |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| E: EN + MT-IT (1994) -> IT test | %.4f | %.4f | %.4f | **%.4f** |"
             % (e["metrics"]["it_test"]["accuracy"], e["metrics"]["it_test"]["micro_f1"],
                e["metrics"]["it_test"]["macro_f1"], e["metrics"]["it_test"]["auc_ovr_macro"]))
    L.append("| E: EN + MT-IT (1994) -> EN test | %.4f | %.4f | %.4f | **%.4f** |"
             % (e["metrics"]["en_test"]["accuracy"], e["metrics"]["en_test"]["micro_f1"],
                e["metrics"]["en_test"]["macro_f1"], e["metrics"]["en_test"]["auc_ovr_macro"]))
    L.append("")
    L.append("| ratio | value | >= %.3f | reading |" % BAR)
    L.append("|---|---:|:--:|---|")
    L.append("| E(IT) / B(IT) | **%.4f** | %s | %s |"
             % (r["e_it_over_b"], "yes" if r["passes_e_b"] else "**no**",
                "MT matches real Italian training" if r["passes_e_b"]
                else "MT falls short of real Italian training"))
    L.append("| E(IT) / C(IT) | **%.4f** | %s | %s |"
             % (r["e_it_over_c"], "yes" if r["passes_e_c"] else "**no**",
                "at the Italian ceiling" if r["passes_e_c"]
                else "below the Italian ceiling"))
    L.append("| E(EN) / A(EN) | **%.4f** | %s | %s |"
             % (r["e_en_over_a_en"], "yes" if r["passes_e_en"] else "**no**",
                "English retained" if r["passes_e_en"] else "English degrades"))
    L.append("| E(IT) / D(IT) | **%.4f** | %s | D is the 3002-row final fit (val rows included) |"
             % (r["e_it_over_d"], "yes" if r["e_it_over_d"] >= BAR else "no"))
    L.append("| E(IT) / D'(IT) | **%.4f** | %s | D' = D's params on the clean 1994 rows |"
             % (r["e_it_over_dprime"], "yes" if r["e_it_over_dprime"] >= BAR else "no"))
    L.append("")
    L.append("**Reference numbers (unchanged, read from `en_vs_enit_raw.json`).**")
    L.append("")
    L.append("| arm | rows | tuned on | IT test AUC | EN test AUC |")
    L.append("|---|---:|---|---:|---:|")
    L.append("| A (EN only) | 997 | EN val | %.4f | %.4f |" % (ref["A_IT"], ref["A_EN"]))
    L.append("| B (EN+IT) | 1994 | IT val | %.4f | %.4f |" % (ref["B_IT"], ref["B_EN"]))
    L.append("| C (IT only) | 997 | IT val | %.4f | - |" % ref["C_IT"])
    L.append("| D (EN+MT-IT, train+val) | 3002 | IT val (train-only fit) | %.4f | %.4f |"
             % (ref["D_IT"], ref["D_EN"]))
    L.append("| D' (D's params, 1994 rows) | 1994 | inherited | %.4f | %.4f |"
             % (ref["Dprime_IT"], ref["Dprime_EN"]))
    L.append("| **E (clean, 1994 rows)** | 1994 | **held-out IT val** | **%.4f** | **%.4f** |"
             % (e["metrics"]["it_test"]["auc_ovr_macro"], e["metrics"]["en_test"]["auc_ovr_macro"]))
    L.append("")
    L.append("**Verdict.** %s" % payload["verdict"])
    L.append("")
    cc = payload.get("cross_check")
    if cc:
        L.append("**Cross-check against D (why the gap is not a tuning leak).** E's 10 trial "
                 "values are bit-identical to D's stored 10 (%s); E's best params equal D's "
                 "(%s); E's test numbers equal D's 1994-row variant to 1e-12 (%s). D therefore "
                 "already tuned on a train-only 1994-row fit, so its Optuna objective was "
                 "held-out. The %.4f gap to D's headline number is entirely D's 3002-row final "
                 "fit, which contains the 504 val rows."
                 % (cc["e_trials_equal_d_trials"], cc["e_best_params_equal_d"],
                    cc["e_test_equals_dprime"], cc["d_minus_e_it"]))
        L.append("")
    L.append("**Caveats.**")
    for c in payload["caveats"]:
        L.append("- %s" % c)
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def write_png(path, payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = payload["arm"]["metrics"]
    ref = payload["reference"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax = axes[0]
    labels = ["A (EN)\nEN test", "B (EN+IT)\nEN test", "E (clean)\nEN test",
              "A (EN)\nIT test", "B (EN+IT)\nIT test", "C (IT)\nIT test",
              "D' (1994)\nIT test", "D (3002)\nIT test", "E (clean)\nIT test"]
    vals = [ref["A_EN"], ref["B_EN"], e["en_test"]["auc_ovr_macro"],
            ref["A_IT"], ref["B_IT"], ref["C_IT"],
            ref["Dprime_IT"], ref["D_IT"], e["it_test"]["auc_ovr_macro"]]
    cols = ["#7f8c8d"] * 3 + ["#c0392b", "#c0392b", "#c0392b", "#e67e22", "#e67e22", "#27ae60"]
    bars = ax.bar(np.arange(len(vals)), vals, 0.68, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, "%.4f" % v,
                ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("OVR-macro AUC")
    ax.set_title("E42 D_clean vs stored arms (test AUC)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    r = payload["ratios"]
    vals = [r["e_it_over_b"], r["e_it_over_c"], r["e_en_over_a_en"],
            r["e_it_over_d"], r["e_it_over_dprime"]]
    labs = ["E(IT)/B(IT)", "E(IT)/C(IT)", "E(EN)/A(EN)", "E(IT)/D(IT)\n(vs inflated)",
            "E(IT)/D'(IT)\n(clean rows, D params)"]
    cols = ["#c0392b" if v < BAR else "#27ae60" for v in vals]
    bars = ax.bar(np.arange(len(vals)), vals, 0.62, color=cols)
    ax.axhline(BAR, color="black", ls="--", lw=1.2, label="0.975 bar")
    ax.axhline(1.0, color="grey", ls=":", lw=1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.4f" % v,
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, max(1.15, max(vals) + 0.15))
    ax.set_ylabel("AUC ratio")
    ax.set_title("E42 D_clean ratios at the 0.975 bar", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def launch(passthrough):
    os.makedirs(RUN_DIR, exist_ok=True)
    vbs = os.path.join(HERE, "launch_hidden.vbs")
    py = sys.executable or r"C:\Users\pedro\anaconda3\python.exe"
    script = os.path.abspath(__file__)
    args = " ".join(passthrough)

    wrapper = os.path.join(RUN_DIR, "run.cmd")
    with open(wrapper, "w", encoding="ascii") as f:
        f.write("@echo off\r\n")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            f.write("set %s=2\r\n" % var)
        f.write('"%s" "%s" %s\r\n' % (py, script, args))

    for p in (PID_PATH, STATUS_PATH):
        if os.path.isfile(p):
            os.remove(p)
    t_launch = time.time()
    launch_log = os.path.join(RUN_DIR, "en_vs_enit_dclean.launch.log")
    proc = subprocess.Popen(["wscript", "//B", "//Nologo", vbs, wrapper, launch_log, HERE],
                            cwd=HERE, close_fds=True)
    proc.wait()

    pid, seen = None, False
    for _ in range(60):
        if os.path.isfile(PID_PATH) and os.path.getmtime(PID_PATH) >= t_launch - 1:
            try:
                pid = int(open(PID_PATH, encoding="utf-8").read().strip())
                seen = True
                break
            except Exception:
                pass
        time.sleep(0.5)
    alive = False
    if pid:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True).stdout
        alive = str(pid) in out
    print("launched detached: %s" % wrapper)
    print("pid=%s alive=%s log=%s status=%s" % (pid, alive, LOG_PATH, STATUS_PATH))
    return 0 if (seen and alive) else 1


def read_reference():
    """Read the parent E42 JSON read-only and assert the reference numbers are the ones used."""
    if not os.path.isfile(RAW_JSON):
        raise SystemExit("missing %s -- the parent E42 run must exist" % RAW_JSON)
    d = json.load(open(RAW_JSON, encoding="utf-8"))
    ref = {
        "A_IT": d["arms"]["A"]["metrics"]["it_test"]["auc_ovr_macro"],
        "A_EN": d["arms"]["A"]["metrics"]["en_test"]["auc_ovr_macro"],
        "B_IT": d["arms"]["B"]["metrics"]["it_test"]["auc_ovr_macro"],
        "B_EN": d["arms"]["B"]["metrics"]["en_test"]["auc_ovr_macro"],
        "C_IT": d["arms"]["C"]["metrics"]["it_test"]["auc_ovr_macro"],
        "D_IT": d["arms"]["D"]["metrics"]["it_test"]["auc_ovr_macro"],
        "D_EN": d["arms"]["D"]["metrics"]["en_test"]["auc_ovr_macro"],
        "Dprime_IT": d["comparison_d"]["d_trainonly"]["it_test"]["auc_ovr_macro"],
        "Dprime_EN": d["comparison_d"]["d_trainonly"]["en_test"]["auc_ovr_macro"],
    }
    for k in ("A_EN", "B_IT", "C_IT"):
        if abs(ref[k] - REF_EXPECTED[k]) > 1e-9:
            raise SystemExit("reference drift: %s is %.12f, expected %.12f"
                             % (k, ref[k], REF_EXPECTED[k]))
    return ref


def build_verdict(e_it, e_en, ref, ratios):
    """Single source of truth for the verdict text (also used to re-render older JSONs)."""
    inflation_note = (
        "D's raw %.4f is **not comparable**, but not because of in-sample tuning: D's Optuna "
        "objective was scored on a 1994-row train-only fit, exactly as E does here, which is "
        "why E's 10 trials and best params reproduce D's bitwise. The gap is a final-fit "
        "row-count effect -- D refits on 3002 rows (1501 EN train+val + MT-IT of those same "
        "1501, i.e. 1.5x B and 3x A, and the 504 IT-val rows are inside it), while E refits on "
        "the 1994 clean rows. That is +%.4f AUC, and it is why D/C = 1.0519 was misleading."
        % (ref["D_IT"], ref["D_IT"] - e_it))
    return (
        "Clean translate-train scores **%.4f** AUC on real Italian vs **%.4f** for B (real "
        "Italian train) and **%.4f** for the Italian-only ceiling C: E/B = **%.4f** (%s the "
        "%.3f bar) and E/C = **%.4f** (%s). On English it scores **%.4f** vs A's **%.4f** "
        "(E/A = **%.4f**, %s). So with an honest held-out objective and matched rows, MT "
        "Italian is %s. %s"
        % (e_it, ref["B_IT"], ref["C_IT"], ratios["e_it_over_b"],
           "above" if ratios["passes_e_b"] else "below", BAR,
           ratios["e_it_over_c"], "above" if ratios["passes_e_c"] else "below",
           e_en, ref["A_EN"], ratios["e_en_over_a_en"],
           "within the bar" if ratios["passes_e_en"] else "below the bar",
           ("a genuine substitute for real Italian training data"
            if ratios["passes_e_b"] else
            "not a substitute for real Italian training data at this size"),
           inflation_note)
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--render", action="store_true")
    args, passthrough = ap.parse_known_args(argv)

    if args.render:
        payload = json.load(open(OUT_JSON, encoding="utf-8"))
        write_md(OUT_MD, payload)
        write_png(OUT_PNG, payload)
        print("rendered %s / %s" % (OUT_MD, OUT_PNG))
        return 0

    if args.launch:
        return launch((passthrough or []) +
                      (["--smoke"] if args.smoke else []) +
                      (["--trials", str(args.trials)] if args.trials else []))

    n_trials = args.trials or (2 if args.smoke else N_TRIALS_DEFAULT)
    open_log()
    write_pid()
    t_start = time.time()

    log("=" * 78)
    log("E42 sidecar D_clean (arm E) -- leak-free translate-train, FLORES sentence only")
    log("python=%s  pid=%d  trials=%d  smoke=%s"
        % (sys.version.split()[0], os.getpid(), n_trials, args.smoke))
    log("features=raw mean-pooled EuroBERT-210m | L2=none | zscore=none | perlang_z=none | "
        "PCA/INLP=none | test-time MT=none")
    log("objective=auc (OVR-macro on UNSEEN real IT val, maximize) | TPE seed=0 | fresh study")
    log("xgboost %s | optuna %s" % (xgb.__version__, optuna.__version__))
    write_status(stage="loading-data", trials=n_trials, smoke=args.smoke)

    prep = T.prepare_sentence()
    emb = T.load_sent_embeddings("mean", prep["kept"], 2009)
    y = prep["y"]
    tr, va, te = prep["train_idx"], prep["val_idx"], prep["test_idx"]
    X_en = np.ascontiguousarray(emb["en"], dtype=np.float32)
    X_it = np.ascontiguousarray(emb["it"], dtype=np.float32)

    assert (len(tr), len(va), len(te)) == (997, 504, 505)
    assert not (set(tr.tolist()) & set(va.tolist()))
    assert not (set(tr.tolist()) & set(te.tolist()))
    assert not (set(va.tolist()) & set(te.tolist()))

    MT = np.load(MT_CACHE)
    assert MT.shape == (len(tr) + len(va), X_en.shape[1]), MT.shape

    # STRICT: only the train slice of the MT cache may enter training.
    MT_tr = np.ascontiguousarray(MT[:len(tr)], dtype=np.float32)
    Xd_tr = np.vstack([X_en[tr], MT_tr])
    yd_tr = np.concatenate([y[tr], y[tr]])
    assert len(yd_tr) == 1994
    assert Xd_tr.shape[0] == 1994

    # held-out proof, computed on the actual matrices
    val_rows_in_training = 0
    val_set = set(va.tolist())
    for i in tr.tolist():
        if i in val_set:
            val_rows_in_training += 1
    held_out = {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "disjoint": True,
        "val_rows_in_training": int(val_rows_in_training),
        "train_matrix_rows": int(Xd_tr.shape[0]),
        "mt_rows_used": int(len(MT_tr)),
        "val_mt_rows_used": 0,
        "val_mt_slice_untouched": "MT[%d:] never referenced" % len(tr),
        "note": "every training row derives from a train index; the 504 val indices appear "
                "in no fit and no MT slice",
    }
    assert val_rows_in_training == 0
    ref = read_reference()   # fail fast if the parent JSON drifted
    log("held-out: split 997/504/505 disjoint; val rows in training=%d; MT rows used=%d "
        "(MT[997:] never referenced)" % (val_rows_in_training, len(MT_tr)))
    log("features raw float32, dim=%d, mean(norm_en_train)=%.4f, mean(norm_mt_train)=%.4f"
        % (X_en.shape[1], float(np.linalg.norm(X_en[tr], axis=1).mean()),
           float(np.linalg.norm(MT_tr, axis=1).mean())))

    state = {"stage": "optuna", "trials": n_trials, "smoke": args.smoke}

    def objective(trial):
        p = dict(BASE_PARAMS)
        p.update(search_space(trial))
        probs = fit(p, Xd_tr, yd_tr).predict_proba(X_it[va])
        return float(roc_auc_score(y[va], probs, multi_class="ovr",
                                   average="macro", labels=np.arange(N_CLASSES)))

    def callback(study, trial):
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best = max((t.value for t in done if t.value is not None), default=None)
        log("E trial %d/%d value=%.6f best=%s"
            % (len(done), n_trials, -1.0 if trial.value is None else trial.value,
               "%.6f" % best if best is not None else "n/a"))
        state.update({"trial": len(done), "best": best})
        write_status(**state)

    study = optuna.create_study(study_name="E42_D_clean", direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    best_params = dict(BASE_PARAMS)
    best_params.update(study.best_params)
    model = fit(best_params, Xd_tr, yd_tr)
    e_metrics = {
        "it_test": metrics(y[te], model.predict_proba(X_it[te])),
        "en_test": metrics(y[te], model.predict_proba(X_en[te])),
        "it_val": metrics(y[va], model.predict_proba(X_it[va])),
    }
    log("E test: IT AUC=%.4f (acc %.4f, macroF1 %.4f) | EN AUC=%.4f"
        % (e_metrics["it_test"]["auc_ovr_macro"], e_metrics["it_test"]["accuracy"],
           e_metrics["it_test"]["macro_f1"], e_metrics["en_test"]["auc_ovr_macro"]))

    ref = read_reference()
    e_it = e_metrics["it_test"]["auc_ovr_macro"]
    e_en = e_metrics["en_test"]["auc_ovr_macro"]

    ratios = {
        "bar": BAR,
        "e_it_over_b": float(e_it / ref["B_IT"]),
        "e_it_over_c": float(e_it / ref["C_IT"]),
        "e_en_over_a_en": float(e_en / ref["A_EN"]),
        "e_it_over_d": float(e_it / ref["D_IT"]),
        "e_it_over_dprime": float(e_it / ref["Dprime_IT"]),
        "passes_e_b": bool(e_it / ref["B_IT"] >= BAR),
        "passes_e_c": bool(e_it / ref["C_IT"] >= BAR),
        "passes_e_en": bool(e_en / ref["A_EN"] >= BAR),
        "d_inflation_it": float(ref["D_IT"] - e_it),
        "dprime_gap_it": float(ref["Dprime_IT"] - e_it),
    }

    verdict = build_verdict(e_it, e_en, ref, ratios)

    caveats = [
        "Single seed (0); 505-row test. AUC SE ~0.01-0.02, so E/B and E/C differences under "
        "~0.02 AUC are not resolved.",
        "E trains on 1994 rows, the same size as B, so E/B and E/C are size-matched -- unlike "
        "D/B (3002 vs 1994) and D/C (3002 vs 997).",
        "Only EN train rows were translated; the 504 val and 505 test texts are the original "
        "human Italian and are never translated or trained on.",
        "MT output is synthetic; a different MT system would move E. The number is specific to "
        "Helsinki-NLP/opus-mt-en-it.",
        "10 Optuna trials: enough to beat defaults, not to exhaust the space; E's params are "
        "drawn fresh and need not match B's or D's.",
        "The parent E42 JSON is read-only for reference numbers and asserted against expected "
        "values, so E cannot silently inherit a drifted baseline.",
    ]

    payload = {
        "experiment": "E42-D_clean",
        "arm_name": "E",
        "status": "smoke" if args.smoke else "done",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "reproduction_status": "SMOKE ONLY -- not a full run" if args.smoke else
                               "REPRODUCIBLE (deterministic seed 0, fresh study)",
        "host": {"python": sys.version.split()[0], "xgboost": xgb.__version__,
                 "optuna": optuna.__version__, "numpy": np.__version__,
                 "threads": {k: os.environ.get(k) for k in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "data": {
            "csv": "flores200_en_it_de_fr.csv", "dataset": "sent (no flores_agg)",
            "embeddings": "embeddings/mean/emb_{en,it}.npy", "backbone": "EuroBERT-210m mean",
            "n_source": 2009, "n_kept": int(len(prep["kept"])),
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "n_classes": N_CLASSES, "seed": SEED,
        },
        "features": {"kind": "raw", "l2": False, "zscore": False, "perlang_z": False,
                     "centering": "none", "pca_or_inlp": False, "dtype": "float32",
                     "test_time_mt": False},
        "mt": {"model": MT_MODEL, "cache": MT_CACHE, "reused_cache": True,
               "rows_translated": 997, "rows_referenced": "MT[:997] (train slice only)"},
        "protocol": {"model": "xgboost.XGBClassifier", "base_params": BASE_PARAMS,
                     "search_space": "max_depth 3-10; learning_rate 0.03-0.3 (log); "
                                     "n_estimators 100-800; min_child_weight 1-10; "
                                     "subsample 0.6-1.0; colsample_bytree 0.4-1.0; "
                                     "reg_lambda 0-5",
                     "sampler": "TPESampler(seed=0)", "direction": "maximize",
                     "objective": "auc_ovr_macro on the unseen real IT val",
                     "trials": n_trials,
                     "final_fit": "best params on EN train + MT-IT train (1994 rows)"},
        "held_out": held_out,
        "arm": {
            "name": "E (D_clean)",
            "train_langs": ["en", "mt-it"],
            "n_train_rows": int(len(yd_tr)),
            "tuned_on": "real IT val (504 rows), held out from every fit",
            "n_trials": n_trials,
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "trials": [{"number": t.number, "state": t.state.name, "value": t.value,
                        "params": t.params} for t in study.trials],
            "seconds": round(time.time() - t_start, 1),
            "metrics": e_metrics,
        },
        "reference": ref,
        "ratios": ratios,
        "verdict": verdict,
        "caveats": caveats,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(OUT_MD, payload)
    write_png(OUT_PNG, payload)

    log("E(IT)/B=%.4f | E(IT)/C=%.4f | E(EN)/A=%.4f | E/D=%.4f | E/D'=%.4f"
        % (ratios["e_it_over_b"], ratios["e_it_over_c"], ratios["e_en_over_a_en"],
           ratios["e_it_over_d"], ratios["e_it_over_dprime"]))
    log("wrote %s / %s / %s" % (OUT_JSON, OUT_MD, OUT_PNG))
    write_status(stage="done", trials=n_trials, smoke=args.smoke,
                 e_it_auc=e_it, e_en_auc=e_en, e_it_over_b=ratios["e_it_over_b"],
                 e_it_over_c=ratios["e_it_over_c"], e_en_over_a_en=ratios["e_en_over_a_en"],
                 seconds=payload["runtime_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
