"""Two-backbone (EuroBERT-210m vs 610m) Precision/Recall vs threshold for FLORES-agg,
with an **Optuna-tuned** model per (backbone, method) cell.

What this does
--------------
Extends the FLORES **agg** (11-class URL-aggregated topic, ``flores_agg.csv``) per-method
Precision/Recall-vs-threshold experiment to BOTH backbones, laid out with **EuroBERT-210m
in the LEFT column and EuroBERT-610m in the RIGHT column**, and overlays **English and
Italian** on every subplot.

The five feature-transform "methods" are identical to ``agg_pr_methods.py`` /
``agg_pr_lang.py`` and are applied PER BACKBONE:

  * ``raw``      -- no transform.
  * ``pc``       -- GLOBAL/iterative PC ablation, k=2, fit on the STACKED-language TRAIN
                    pool (``pc_ablation_flores.pc_ablation_dirs`` / ``apply_dirs``).
  * ``inlp``     -- project out the FLORES language-subspace basis, backbone-matched:
                    - 210m (768-d): the existing ``lang_inlp_agg.npz`` (3 dirs).
                    - 610m (1152-d): a basis COMPUTED here on the stacked-4-language 610m
                      agg pool with the SAME iterative procedure used to build
                      ``lang_inlp_agg.npz`` (``crosslingual_topic_agg.run_inlp``, itself
                      the ``lang_inlp.py`` procedure).
  * ``pc+inlp``  -- ablate PC (k=2) first, then project out the inlp subspace.
  * ``l2_thenZ`` -- L2-normalize, then per-language TRAIN z-score.

TUNING (per (backbone, method) cell)
------------------------------------
Each cell is tuned with Optuna TPE (seed 0), **20 trials**, search space =
``xgboost_topic_optuna._optuna_search_space`` with ``n_estimators`` capped at 500
(mirrored here as ``_search_space`` because the source function hard-codes the 100..800
domain).  Base params (untuned) are
``objective="multi:softprob", num_class=11, tree_method="hist", max_bin=64, device="cpu",
n_jobs=2, seed=0``.  Every trial trains on EN train (276) and scores **EN-val (140)** with
objective = **OVR-macro AUC**, falling back to macro-F1 only when the AUC is non-finite.

Best params are then RETRAINED on EN train+val (416) and evaluated on the ENGLISH test
(EN-on-EN, in-language) and the ITALIAN test (EN model -> IT, cross-lingual transfer).  The
threshold sweep is ``t = 0.00..1.00`` step 0.02 (class c positive for a document iff
``p_c >= t``; micro precision / recall / F1 pooled over all (document, class) pairs).

OUTPUTS
-------
  * ``agg_pr2_<method>.png``  -- per method, 2x2 grid: columns = 210m (left) / 610m (right),
                                 rows = Precision (top) / Recall (bottom), EN + IT overlaid,
                                 each language's F1-optimal threshold marked.
  * ``agg_pr2_grid_precision.png`` / ``agg_pr2_grid_recall.png`` -- the PRIMARY combined
                                 grids: rows = methods, columns = {210m (left), 610m (right)};
                                 each cell = micro **precision** (resp. **recall**) vs
                                 threshold for EN + IT, with F1-optimal thresholds marked.
  * ``agg_pr2_grid.png``      -- the same layout for F1 vs threshold (kept alongside).
  * ``agg_pr2.json`` / ``agg_pr2.md`` -- per (backbone, method, language): F1-opt threshold,
                                 precision and recall at it, plus the best val AUC from the
                                 Optuna tuning.

MODES
-----
  * (default) full run: loads the data + transforms (needs ``agg_pr_methods`` etc.), tunes
    every cell with Optuna, then writes all outputs.  ``--resume`` skips cells already in
    the per-cell checkpoint.
  * ``--reemit-from-json``: NO heavy dependencies -- reload the saved ``agg_pr2.json`` and
    regenerate every figure + the md/json deterministically from the recorded tuned curves.
    This is the recovery path when the pipeline scripts/embeddings are unavailable.

CPU only, n_jobs=2, 2 threads, no GPU, no windows, matplotlib Agg.
"""
import os

# 2-thread runtime budget -- set BEFORE numpy / any xgboost import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import json
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
EMB_610_DIR = os.path.join(REPO, "embeddings_agg_610")
INLP_610 = os.path.join(REPO, "lang_inlp_agg_610.npz")

OUT_GRID = os.path.join(REPO, "agg_pr2_grid.png")                 # F1 grid (kept)
OUT_GRID_PREC = os.path.join(REPO, "agg_pr2_grid_precision.png")  # PRIMARY
OUT_GRID_REC = os.path.join(REPO, "agg_pr2_grid_recall.png")      # PRIMARY
OUT_JSON = os.path.join(REPO, "agg_pr2.json")
OUT_MD = os.path.join(REPO, "agg_pr2.md")

TMP = os.environ.get("TEMP", REPO)

# Constants mirror agg_pr_methods (kept local so --reemit-from-json needs no heavy imports).
LANGS = ["en", "it", "de", "fr"]
METHODS = ["raw", "pc", "inlp", "pc+inlp", "l2_thenZ"]
GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 2)
N_CLASSES = 11

BACKBONES = ["210m", "610m"]
BB_LABEL = {"210m": "EuroBERT-210m (768-d)", "610m": "EuroBERT-610m (1152-d)"}

EN_COLOR = "#1f77b4"      # English test (in-language)
IT_COLOR = "#d62728"      # Italian test (cross-lingual transfer)

BASE_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES,
    tree_method="hist",
    max_bin=64,
    device="cpu",
    n_jobs=2,
    seed=0,
)

N_TRIALS = 20
OPTUNA_SEED = 0
N_ESTIMATORS_CAP = 500

# INLP config -- EXACT mirror of crosslingual_topic_agg.py (which produced lang_inlp_agg.npz).
INLP_C = 1.0
INLP_MAX_ITER = 1000
INLP_SPLITS = 5
INLP_STOP_ACC = 0.30
INLP_MAX_ROUNDS = 30
INLP_TOL = 1e-8
INLP_SEED = 0

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def per_method_png(method):
    return os.path.join(REPO, "agg_pr2_%s.png" % method)


def output_files():
    return ([per_method_png(m) for m in METHODS]
            + [OUT_GRID_PREC, OUT_GRID_REC, OUT_GRID, OUT_JSON, OUT_MD])


# ---------------------------------------------------------------------------
# Optuna search space (xgboost_topic_optuna._optuna_search_space, n_estimators capped)
# ---------------------------------------------------------------------------
def _search_space(trial):
    """``xgboost_topic_optuna._optuna_search_space`` with ``n_estimators`` capped at 500.

    The source function hard-codes ``suggest_int("n_estimators", 100, 800)``; the task
    directs a cap at 500, so the identical space is mirrored here with that single change.
    """
    return dict(
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, N_ESTIMATORS_CAP),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )


# ---------------------------------------------------------------------------
# Data: 610m loader (same row convention as train_flores12.load_agg_embeddings)
# ---------------------------------------------------------------------------
def load_agg_embeddings_610(mode, n_rows):
    """Load the 610m agg embeddings -> {lang: (n_rows, 1152)} float arrays."""
    out = {}
    for lang in LANGS:
        path = os.path.join(EMB_610_DIR, mode, "emb_%s.npy" % lang)
        if not os.path.isfile(path):
            raise FileNotFoundError("610m agg embeddings missing: %s" % path)
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != n_rows:
            raise ValueError("expected %d rows in %s, got %s" % (n_rows, path, arr.shape))
        out[lang] = arr
    return out


# ---------------------------------------------------------------------------
# INLP on the 4-language stack (mirror of crosslingual_topic_agg.run_inlp)
# ---------------------------------------------------------------------------
def _cv_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=INLP_SPLITS, shuffle=True, random_state=INLP_SEED)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=INLP_MAX_ITER, C=INLP_C, n_jobs=2)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    a = np.asarray(accs, dtype=np.float64)
    return float(a.mean()), float(a.std())


def compute_inlp(X_by_lang):
    """Iterative null-space projection on the stacked language pool."""
    from sklearn.linear_model import LogisticRegression
    import lang_neutralize as ln
    langs = list(X_by_lang.keys())
    X = np.vstack([np.asarray(X_by_lang[l], dtype=np.float64) for l in langs])
    y = np.concatenate([[i] * len(X_by_lang[l]) for i, l in enumerate(langs)])
    d = X.shape[1]
    Xcur = X.copy()
    q_list, acc_curve, ranks = [], [], []
    rounds = 0
    for r in range(1, INLP_MAX_ROUNDS + 1):
        clf = LogisticRegression(max_iter=INLP_MAX_ITER, C=INLP_C, n_jobs=2)
        clf.fit(Xcur, y)
        W = np.asarray(clf.coef_, dtype=np.float64)
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
        rank = int((S > INLP_TOL).sum())
        Q = Vt[:rank].copy()
        q_list.append(Q)
        Xcur = ln.project_out(Xcur, Q)
        m, _s = _cv_probe(Xcur, y)
        acc_curve.append(m)
        ranks.append(rank)
        rounds = r
        log("  INLP round %d rank=%d cv_acc=%.4f" % (r, rank, m))
        if m <= INLP_STOP_ACC:
            log("  INLP stop: cv_acc %.3f <= %.2f" % (m, INLP_STOP_ACC))
            break
    Vall = np.vstack(q_list) if q_list else np.zeros((0, d))
    if Vall.shape[0]:
        _, Sv, Vt = np.linalg.svd(Vall, full_matrices=False)
        r_total = int((Sv > INLP_TOL).sum())
        basis = Vt[:r_total].copy()
    else:
        r_total = 0
        basis = np.zeros((0, d))
    return {
        "basis": basis, "n_removed": r_total, "acc_curve": acc_curve,
        "ranks_per_round": ranks, "rounds": rounds,
        "reached_chance": bool(acc_curve and acc_curve[-1] <= INLP_STOP_ACC),
        "final_acc": float(acc_curve[-1]) if acc_curve else None,
    }


def get_basis_610(X610, recompute=False):
    """Load the 610m INLP basis from npz, or compute + save it (lang_inlp_agg_610.npz)."""
    if os.path.isfile(INLP_610) and not recompute:
        z = np.load(INLP_610)
        basis = np.asarray(z["basis"], dtype=np.float64)
        acc_curve = ([float(v) for v in np.asarray(z["acc_curve"]).tolist()]
                     if "acc_curve" in z else [])
        n_removed = int(z["n_removed"]) if "n_removed" in z else int(basis.shape[0])
        final_acc = acc_curve[-1] if acc_curve else None
        meta = {"source": "loaded", "n_removed": n_removed,
                "rounds": len(acc_curve) if acc_curve else None,
                "acc_curve": acc_curve, "final_acc": final_acc,
                "reached_chance": (final_acc is not None and final_acc <= INLP_STOP_ACC)}
        log("loaded 610m INLP basis %s from %s" % (basis.shape, os.path.basename(INLP_610)))
        return basis, meta
    log("computing 610m INLP basis on the 4-language stack (%d rows/lang) ..."
        % X610["en"].shape[0])
    inlp = compute_inlp({lg: X610[lg] for lg in LANGS})
    np.savez(INLP_610, basis=inlp["basis"].astype(np.float64),
             n_removed=np.int64(inlp["n_removed"]),
             acc_curve=np.asarray(inlp["acc_curve"], dtype=np.float64))
    log("wrote %s: rounds=%d n_removed=%d final_acc=%s reached=%s"
        % (os.path.basename(INLP_610), inlp["rounds"], inlp["n_removed"],
           inlp["final_acc"], inlp["reached_chance"]))
    meta = {"source": "computed", "n_removed": inlp["n_removed"], "rounds": inlp["rounds"],
            "acc_curve": [float(v) for v in inlp["acc_curve"]],
            "final_acc": inlp["final_acc"], "reached_chance": inlp["reached_chance"]}
    return inlp["basis"], meta


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def nearest_index(grid, t):
    return int(np.argmin(np.abs(np.asarray(grid) - t)))


def macro_ovr_auc(y_true, P):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, P, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def argmax_macro_f1(y_true, P):
    from sklearn.metrics import f1_score
    pred = np.argmax(P, axis=1)
    return float(f1_score(y_true, pred, average="macro",
                          labels=list(range(N_CLASSES)), zero_division=0))


def bb_record(res_lang):
    a = res_lang["at_optimal"]
    return {
        "curves": {k: [float(v) for v in res_lang["curves"][k]]
                   for k in ("precision", "recall", "f1")},
        "f1_opt_t": float(res_lang["optimal_f1_t"]),
        "at_f1_opt": {"precision": float(a["precision"]), "recall": float(a["recall"]),
                      "f1": float(a["f1"])},
    }


# ---------------------------------------------------------------------------
# Optuna tuning + final fit for one (backbone, method) cell
# ---------------------------------------------------------------------------
def tune_cell(method, X, split, basis, y, n_trials):
    import xgboost as xgb
    import optuna
    import agg_pr_methods as M
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
    F = M.build_method_features(X, split, basis, method)
    Xtr, ytr = F["en"][tr], y[tr]
    Xva, yva = F["en"][va], y[va]

    def objective(trial):
        p = dict(BASE_PARAMS)
        p.update(_search_space(trial))
        clf = xgb.XGBClassifier(**p)
        clf.fit(Xtr, ytr)
        P = clf.predict_proba(Xva)
        auc = macro_ovr_auc(yva, P)
        mf1 = argmax_macro_f1(yva, P)
        trial.set_user_attr("val_macro_f1", mf1)
        trial.set_user_attr("val_auc", auc)
        return float(auc) if np.isfinite(auc) else float(mf1)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)

    pbest = dict(BASE_PARAMS)
    pbest.update(best_params)
    clf = xgb.XGBClassifier(**pbest)
    clf.fit(Xtr, ytr)
    Pva = clf.predict_proba(Xva)
    best_val_auc = macro_ovr_auc(yva, Pva)
    best_val_mf1 = argmax_macro_f1(yva, Pva)

    Xtr2 = np.vstack([F["en"][tr], F["en"][va]])
    ytr2 = np.concatenate([y[tr], y[va]])
    clf2 = xgb.XGBClassifier(**pbest)
    clf2.fit(Xtr2, ytr2)
    res = {}
    for tag, lg in (("it", "it"), ("en", "en")):
        P = clf2.predict_proba(F[lg][te])
        Y0 = M.onehot(y[te])
        series = M.sweep_prf(Y0, P, GRID)
        opt_t, opt_i = M.optimal(GRID, series)
        res[tag] = {"curves": series, "optimal_f1_t": opt_t,
                    "at_optimal": M.at_index(series, opt_i)}

    info = {
        "n_trials": int(n_trials),
        "best_params": {k: (float(v) if isinstance(v, float) else int(v))
                        for k, v in best_params.items()},
        "decision_metric": ("val_ovr_macro_auc" if np.isfinite(best_val_auc)
                            else "val_macro_f1_fallback"),
        "study_best_value": float(study.best_value),
        "best_val_auc": (float(best_val_auc) if np.isfinite(best_val_auc) else None),
        "best_val_macro_f1": float(best_val_mf1),
        "n_trials_failed": int(sum(1 for t in study.trials if t.value is None)),
    }
    record = {"en": bb_record(res["en"]), "it": bb_record(res["it"])}
    return record, info


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Payload / report
# ---------------------------------------------------------------------------
def _fmt_params(p):
    return ("d%d lr%.3f n%d ss%.2f cs%.2f mcw%d rl%.3g g%.2f"
            % (p["max_depth"], p["learning_rate"], p["n_estimators"], p["subsample"],
               p["colsample_bytree"], p["min_child_weight"], p["reg_lambda"], p["gamma"]))


def build_payload(agg, results, tuning, inlp_meta, n_trials, elapsed):
    gaps = {bb: {} for bb in BACKBONES}
    for bb in BACKBONES:
        for m in METHODS:
            en = results[bb][m]["en"]["at_f1_opt"]
            it = results[bb][m]["it"]["at_f1_opt"]
            gaps[bb][m] = {
                "f1_opt_t_en": results[bb][m]["en"]["f1_opt_t"],
                "f1_opt_t_it": results[bb][m]["it"]["f1_opt_t"],
                "delta_precision": en["precision"] - it["precision"],
                "delta_recall": en["recall"] - it["recall"],
                "delta_f1": en["f1"] - it["f1"],
            }
    best_it = {bb: sorted(METHODS,
                          key=lambda m: results[bb][m]["it"]["at_f1_opt"]["f1"],
                          reverse=True) for bb in BACKBONES}
    best_auc = {bb: sorted(METHODS,
                           key=lambda m: (tuning[bb][m]["best_val_auc"]
                                          if tuning[bb][m]["best_val_auc"] is not None
                                          else -1.0), reverse=True) for bb in BACKBONES}
    best_gap = {bb: sorted(METHODS, key=lambda m: gaps[bb][m]["delta_f1"])
                for bb in BACKBONES}

    def mean_gap(bb):
        return float(np.mean([gaps[bb][m]["delta_f1"] for m in METHODS]))

    return {
        "dataset": "flores_agg.csv (FLORES URL-aggregated, 11-class topic)",
        "model": "english XGB multi:softprob, Optuna-tuned per (backbone, method); trained on "
                 "EN train+val; targets EN (in-language) and IT (cross-lingual)",
        "tuning_protocol": {
            "library": "optuna", "sampler": "TPESampler(seed=0)", "n_trials": int(n_trials),
            "search_space": "xgboost_topic_optuna._optuna_search_space with n_estimators "
                            "capped at %d" % N_ESTIMATORS_CAP,
            "base_params": BASE_PARAMS,
            "trial_fit": "EN train (276 rows)",
            "objective": "EN-val (140) OVR-macro AUC (sklearn roc_auc_score multi_class=ovr, "
                         "average=macro); fallback to single-label macro-F1 only if AUC is "
                         "non-finite",
            "final_fit": "best params retrained on EN train+val (416); evaluated EN test + "
                         "IT test with the 0.00..1.00 threshold sweep",
        },
        "backbones": {
            "210m": {"dim": 768, "embeddings": "embeddings_agg/mean/emb_<lang>.npy",
                     "inlp_basis": "lang_inlp_agg.npz (existing)"},
            "610m": {"dim": 1152, "embeddings": "embeddings_agg_610/mean/emb_<lang>.npy",
                     "inlp_basis": "lang_inlp_agg_610.npz (computed here)"},
        },
        "methods": METHODS,
        "method_definitions": {
            "raw": "no transform",
            "pc": "iterative GLOBAL PC ablation k=2, fit on the stacked-language TRAIN pool "
                  "(pc_ablation_flores.pc_ablation_dirs / apply_dirs), per backbone",
            "inlp": "project out the backbone-matched FLORES language-subspace basis "
                    "(210m=lang_inlp_agg.npz 3 dirs; 610m=lang_inlp_agg_610.npz)",
            "pc+inlp": "pc (k=2) then project out the inlp subspace",
            "l2_thenZ": "L2-normalize then per-language TRAIN z-score (stats from that "
                        "language's own train_idx rows)",
        },
        "classes": agg["classes"],
        "n_classes": agg["n_classes"],
        "excluded_class": agg["excluded_class"],
        "sizes": {"train": int(len(agg["train_idx"])), "val": int(len(agg["val_idx"])),
                  "test": int(len(agg["test_idx"])),
                  "train+val": int(len(agg["train_idx"]) + len(agg["val_idx"]))},
        "threshold_grid": [float(t) for t in GRID],
        "inlp_610_meta": inlp_meta,
        "elapsed_sec": round(elapsed, 2),
        "note": "At each t, class c is positive iff p_c >= t; micro P/R/F1 pooled over all "
                "(document, class) pairs. 'en' = EN model -> English test (in-language); "
                "'it' = EN model -> Italian test (cross-lingual). Gap = EN peak F1 minus "
                "IT peak F1 at each language's own F1-optimal threshold.",
        "en_it_f1_gap_at_f1_opt": gaps,
        "mean_en_it_f1_gap": {bb: mean_gap(bb) for bb in BACKBONES},
        "tuning": tuning,
        "italian_best_method_ranked": best_it,
        "val_auc_best_method_ranked": best_auc,
        "closest_to_english_ranked": best_gap,
        "results": results,
    }


def _f(x):
    return "%.4f" % x


def _auc(x):
    return "n/a" if x is None else "%.4f" % x


def write_md(payload):
    res = payload["results"]
    gaps = payload["en_it_f1_gap_at_f1_opt"]
    mg = payload["mean_en_it_f1_gap"]
    tun = payload["tuning"]
    tp = payload["tuning_protocol"]
    L = []
    L.append("# FLORES-agg (11-class): P/R vs threshold for EuroBERT-210m vs 610m, "
             "English vs Italian (Optuna-tuned)\n")
    L.append("**Dataset:** %s. **Model:** %s, %d classes, train/val/test = %d/%d/%d "
             "(EN train+val = %d rows). Backbones: 210m (768-d) LEFT, 610m (1152-d) RIGHT."
             % (payload["dataset"], payload["model"], payload["n_classes"],
                payload["sizes"]["train"], payload["sizes"]["val"], payload["sizes"]["test"],
                payload["sizes"]["train+val"]))
    L.append("")
    L.append("Class c is positive for a document iff `p_c >= t`; sweep t over 0.00..1.00 step "
             "0.02. Micro P/R/F1 pooled over all (document, class) pairs. Each (backbone, "
             "method) trains one ENGLISH 11-class XGB on EN train+val and is evaluated on the "
             "**English test** (`en`, in-language) and the **Italian test** (`it`, "
             "cross-lingual transfer).")
    L.append("")
    L.append("## Tuning protocol\n")
    L.append("- Optuna %s, **%d trials** per (backbone, method) cell; search space = `%s`."
             % (tp["sampler"], tp["n_trials"], tp["search_space"]))
    L.append("- Base params: `%s`." % tp["base_params"])
    L.append("- Each trial trains on **EN train** (%d) and scores **EN-val** (%d) with "
             "objective = %s." % (payload["sizes"]["train"], payload["sizes"]["val"],
                                  tp["objective"]))
    L.append("- Best params retrained on **EN train+val** (%d), then evaluated on EN test + "
             "IT test with the threshold sweep.\n" % payload["sizes"]["train+val"])
    L.append("## Methods\n")
    for m in METHODS:
        L.append("- `%s`: %s" % (m, payload["method_definitions"][m]))
    L.append("")
    m610 = payload["inlp_610_meta"]
    L.append("**610m INLP basis** (`lang_inlp_agg_610.npz`, %s): n_removed=%s, rounds=%s, "
             "final probe acc=%s, reached chance(<=0.30)=%s.\n"
             % (m610["source"], m610["n_removed"], m610["rounds"], m610["final_acc"],
                m610["reached_chance"]))

    L.append("## Optuna tuning result per (backbone, method)\n")
    L.append("| backbone | method | best val AUC | val macro-F1 | metric | best params |")
    L.append("|---|---|---|---|---|---|")
    for bb in BACKBONES:
        for m in METHODS:
            t = tun[bb][m]
            L.append("| `%s` | `%s` | **%s** | %s | %s | `%s` |"
                     % (bb, m, _auc(t["best_val_auc"]), _f(t["best_val_macro_f1"]),
                        t["decision_metric"], _fmt_params(t["best_params"])))
    L.append("")
    L.append("### Ranking by best val AUC vs by Italian test F1\n")
    for bb in BACKBONES:
        auc_order = payload["val_auc_best_method_ranked"][bb]
        it_order = payload["italian_best_method_ranked"][bb]
        same = auc_order == it_order
        L.append("- `%s` by val AUC: %s."
                 % (bb, ", ".join("`%s`(%s)" % (m, _auc(tun[bb][m]["best_val_auc"]))
                                  for m in auc_order)))
        L.append("  by IT test F1: %s. Orderings %s."
                 % (", ".join("`%s`(%.4f)" % (m, res[bb][m]["it"]["at_f1_opt"]["f1"])
                              for m in it_order),
                    "AGREE" if same else "DIFFER"))
    L.append("")

    L.append("## Per (backbone, method, language): F1-opt threshold, precision and recall\n")
    L.append("`val AUC` is the best Optuna EN-val OVR-macro AUC for that (backbone, method) "
             "cell. `F1-opt t` / `P` / `R` / `F1` are on the TEST split at that language's "
             "own F1-optimal threshold.")
    L.append("")
    L.append("| backbone | method | language | val AUC | F1-opt t | precision | recall | F1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for bb in BACKBONES:
        for m in METHODS:
            va = _auc(tun[bb][m]["best_val_auc"])
            for tag, label in (("en", "en (EN->EN)"), ("it", "it (EN->IT)")):
                r = res[bb][m][tag]
                a = r["at_f1_opt"]
                L.append("| `%s` | `%s` | %s | %s | %.2f | %s | %s | %s |"
                         % (bb, m, label, va, r["f1_opt_t"], _f(a["precision"]),
                            _f(a["recall"]), _f(a["f1"])))
    L.append("")

    L.append("## EN vs IT gap per method, 210m vs 610m\n")
    L.append("Gap = EN peak F1 minus IT peak F1 (each at its own F1-optimal t). Positive => "
             "Italian below English. `d(gap)` = 610m gap minus 210m gap; **negative => 610m "
             "narrows the EN-IT gap**.")
    L.append("")
    L.append("| method | 210m dF1 | 610m dF1 | d(gap) 610m-210m | 210m IT F1 | 610m IT F1 |")
    L.append("|---|---|---|---|---|---|")
    for m in METHODS:
        g2 = gaps["210m"][m]["delta_f1"]
        g6 = gaps["610m"][m]["delta_f1"]
        L.append("| `%s` | %+.4f | %+.4f | **%+.4f** | %.4f | %.4f |"
                 % (m, g2, g6, g6 - g2,
                    res["210m"][m]["it"]["at_f1_opt"]["f1"],
                    res["610m"][m]["it"]["at_f1_opt"]["f1"]))
    L.append("| **mean over methods** | **%+.4f** | **%+.4f** | **%+.4f** | | |"
             % (mg["210m"], mg["610m"], mg["610m"] - mg["210m"]))
    L.append("")

    L.append("## Best method per backbone\n")
    for bb in BACKBONES:
        b = payload["italian_best_method_ranked"][bb][0]
        a = res[bb][b]["it"]["at_f1_opt"]
        L.append("- `%s`: best Italian transfer = **`%s`** (F1-opt t=%.2f, F1=%.4f, "
                 "P=%.4f R=%.4f); best val AUC = `%s` (%s)."
                 % (bb, b, res[bb][b]["it"]["f1_opt_t"], a["f1"], a["precision"],
                    a["recall"], payload["val_auc_best_method_ranked"][bb][0],
                    _auc(tun[bb][payload["val_auc_best_method_ranked"][bb][0]]["best_val_auc"])))
    L.append("")
    L.append("## Figures\n")
    L.append("- `agg_pr2_grid_precision.png` -- combined grid, micro PRECISION vs threshold "
             "(rows = methods; columns = 210m left / 610m right; English + Italian).")
    L.append("- `agg_pr2_grid_recall.png` -- combined grid, micro RECALL vs threshold (same "
             "layout).")
    L.append("- `agg_pr2_grid.png` -- combined grid, F1 vs threshold (same layout).")
    L.append("- `agg_pr2_<method>.png` -- per-method 2x2 (rows = Precision/Recall; columns = "
             "210m/610m; English + Italian).")
    L.append("")

    it_best_210 = max(METHODS, key=lambda m: res["210m"][m]["it"]["at_f1_opt"]["f1"])
    it_best_610 = max(METHODS, key=lambda m: res["610m"][m]["it"]["at_f1_opt"]["f1"])
    L.append("## Does 610m move Italian closer to English?\n")
    L.append("- Best Italian F1: 210m `%s`=%.4f vs 610m `%s`=%.4f (delta %+.4f)."
             % (it_best_210, res["210m"][it_best_210]["it"]["at_f1_opt"]["f1"],
                it_best_610, res["610m"][it_best_610]["it"]["at_f1_opt"]["f1"],
                res["610m"][it_best_610]["it"]["at_f1_opt"]["f1"]
                - res["210m"][it_best_210]["it"]["at_f1_opt"]["f1"]))
    L.append("- Mean EN-IT peak-F1 gap: 210m %+.4f -> 610m %+.4f (change %+.4f)."
             % (mg["210m"], mg["610m"], mg["610m"] - mg["210m"]))
    n_narrow = sum(1 for m in METHODS if gaps["610m"][m]["delta_f1"] < gaps["210m"][m]["delta_f1"])
    L.append("- Methods where the gap narrows under 610m: %d/%d." % (n_narrow, len(METHODS)))
    L.append("- Interpretation: %s"
             % ("the 610m backbone raises Italian F1 and narrows the EN-IT gap (Italian moves "
                "closer to English)." if (mg["610m"] < mg["210m"]) else
                "the 610m backbone does NOT narrow the mean EN-IT gap; the larger backbone "
                "helps both languages similarly (or English more)."))
    L.append("")
    L.append("Elapsed: %.1f s." % payload["elapsed_sec"])
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _draw_lang(ax, rec, metric, color, label, lw=2.1):
    y = rec["curves"][metric]
    ax.plot(GRID, y, color=color, lw=lw, label=label)
    t = rec["f1_opt_t"]
    i = nearest_index(GRID, t)
    ax.plot([t], [y[i]], marker="o", ms=7, color=color, mec="white", mew=1.2, zorder=6)


def per_method_figure(method, results):
    """2x2: rows = Precision/Recall, cols = 210m (left) / 610m (right); EN+IT per cell."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True, sharey=True)
    rows = [("precision", "Precision"), ("recall", "Recall")]
    cols = [(bb, BB_LABEL[bb]) for bb in BACKBONES]
    for ci, (bb, bb_label) in enumerate(cols):
        for ri, (metric, mtitle) in enumerate(rows):
            ax = axes[ri, ci]
            _draw_lang(ax, results[bb][method]["en"], metric, EN_COLOR, "English (EN->EN)")
            _draw_lang(ax, results[bb][method]["it"], metric, IT_COLOR, "Italian (EN->IT)")
            ax.set_title("%s -- %s" % (bb_label, mtitle), fontsize=11.5)
            if ri == 1:
                ax.set_xlabel("threshold t")
            if ci == 0:
                ax.set_ylabel("score (micro)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=9, loc="lower center", framealpha=0.9)
    fig.suptitle("FLORES-agg (11-class) `%s` (Optuna-tuned): Precision/Recall vs threshold, "
                 "English vs Italian (EN model)\ncolumns = 210m (left) / 610m (right); dots = "
                 "each language's F1-optimal threshold." % method, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(per_method_png(method), dpi=150)
    plt.close(fig)


def grid_figure(results, metric, out_path, pretty):
    """rows = methods, cols = {210m, 610m}; each cell = ``metric`` vs threshold, EN+IT."""
    n = len(METHODS)
    fig, axes = plt.subplots(n, 2, figsize=(11.8, 3.0 * n), sharex=True, sharey=True)
    for r, m in enumerate(METHODS):
        for c, bb in enumerate(BACKBONES):
            ax = axes[r, c]
            _draw_lang(ax, results[bb][m]["en"], metric, EN_COLOR, "English")
            _draw_lang(ax, results[bb][m]["it"], metric, IT_COLOR, "Italian")
            if r == 0:
                ax.set_title(BB_LABEL[bb], fontsize=12.5)
            if c == 0:
                ax.set_ylabel("`%s`\nmicro %s" % (m, pretty), fontsize=10)
            if r == n - 1:
                ax.set_xlabel("threshold t")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=9, loc="lower center", framealpha=0.9)
    fig.suptitle("FLORES-agg (11-class): %s vs threshold -- English vs Italian (EN model, "
                 "Optuna-tuned per cell)\nrows = methods; columns = 210m (left) / 610m "
                 "(right); dots = F1-optimal threshold." % pretty, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def emit_figures(results):
    """Write the two PRIMARY precision/recall grids, the F1 grid, and the 2x2 per-method figs."""
    for m in METHODS:
        per_method_figure(m, results)
        log("wrote %s" % os.path.basename(per_method_png(m)))
    grid_figure(results, "precision", OUT_GRID_PREC, "Precision")
    log("wrote %s" % os.path.basename(OUT_GRID_PREC))
    grid_figure(results, "recall", OUT_GRID_REC, "Recall")
    log("wrote %s" % os.path.basename(OUT_GRID_REC))
    grid_figure(results, "f1", OUT_GRID, "F1")
    log("wrote %s" % os.path.basename(OUT_GRID))


# ---------------------------------------------------------------------------
# Re-emit path (no heavy deps): rebuild every figure + md/json from agg_pr2.json
# ---------------------------------------------------------------------------
def reemit_from_json():
    if not os.path.isfile(OUT_JSON):
        raise FileNotFoundError("no %s to re-emit from" % OUT_JSON)
    with open(OUT_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    results = payload["results"]
    log("re-emit: loaded %s (cells=%d)" % (os.path.basename(OUT_JSON), len(METHODS)))
    emit_figures(results)
    payload["outputs"] = [os.path.basename(p) for p in output_files()]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(payload)
    log("wrote %s / %s" % (os.path.basename(OUT_JSON), os.path.basename(OUT_MD)))
    log("re-emit done in %.1fs" % (time.time() - _T0))
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    global INLP_MAX_ITER, INLP_MAX_ROUNDS, N_TRIALS
    ap = argparse.ArgumentParser(description="FLORES-agg P/R vs threshold, 210m vs 610m, "
                                             "Optuna-tuned per cell")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help="Optuna trials per (backbone, method) cell (default 20)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 trials per cell and <=3 INLP rounds for a fast plumbing check")
    ap.add_argument("--resume", action="store_true",
                    help="reuse cells already present in the per-cell checkpoint")
    ap.add_argument("--recompute-inlp", action="store_true",
                    help="recompute the 610m INLP basis even if lang_inlp_agg_610.npz exists")
    ap.add_argument("--reemit-from-json", action="store_true",
                    help="regenerate all figures + md/json from the saved agg_pr2.json "
                         "(no heavy dependencies)")
    args = ap.parse_args(argv)

    if args.reemit_from_json:
        return reemit_from_json()

    import xgboost  # noqa: F401 -- only the full run needs it
    import agg_pr_methods as M

    N_TRIALS = 2 if args.smoke else args.trials
    if args.smoke:
        INLP_MAX_ITER = 200
        INLP_MAX_ROUNDS = 3

    ckpt_path = os.path.join(TMP, "agg_pr2_optuna_partial%s.json"
                             % ("_smoke" if args.smoke else ""))

    agg, X210, basis210 = M.load_data()
    y = agg["y"]
    split = {"train_idx": agg["train_idx"], "val_idx": agg["val_idx"],
             "test_idx": agg["test_idx"]}
    log("loaded agg: 210m X %s, basis %s, split train/val/test = %d/%d/%d"
        % (X210["en"].shape, basis210.shape, len(split["train_idx"]),
           len(split["val_idx"]), len(split["test_idx"])))

    n_rows = len(agg["agg"])
    X610 = {lg: np.asarray(load_agg_embeddings_610("mean", n_rows)[lg], dtype=np.float64)
            for lg in LANGS}
    log("loaded 610m agg embeddings: X %s" % (X610["en"].shape,))
    basis610, inlp_meta = get_basis_610(X610, recompute=args.recompute_inlp)
    log("610m basis %s (%s)" % (basis610.shape, inlp_meta["source"]))

    XB = {"210m": X210, "610m": X610}
    BAS = {"210m": basis210, "610m": basis610}

    state = {}
    if args.resume and os.path.isfile(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            state = json.load(f)
        log("resume: %d cells already done" % len(state))

    results = {bb: {} for bb in BACKBONES}
    tuning = {bb: {} for bb in BACKBONES}
    for bb in BACKBONES:
        for m in METHODS:
            key = "%s|%s" % (bb, m)
            if key in state:
                results[bb][m] = state[key]["res"]
                tuning[bb][m] = state[key]["info"]
                log("skip %-12s (cached)" % key)
                continue
            t_cell = time.time()
            record, info = tune_cell(m, XB[bb], split, BAS[bb], y, N_TRIALS)
            results[bb][m] = record
            tuning[bb][m] = info
            state[key] = {"res": record, "info": info}
            _atomic_json(ckpt_path, state)
            e = record["en"]["at_f1_opt"]
            i = record["it"]["at_f1_opt"]
            log("%s %-8s [%.0fs] valAUC=%s valF1=%.4f | EN F1=%.4f@t=%.2f | IT F1=%.4f@t=%.2f"
                % (bb, m, time.time() - t_cell, _auc(info["best_val_auc"]),
                   info["best_val_macro_f1"], e["f1"], record["en"]["f1_opt_t"],
                   i["f1"], record["it"]["f1_opt_t"]))

    payload = build_payload(agg, results, tuning, inlp_meta, N_TRIALS, time.time() - _T0)
    payload["outputs"] = [os.path.basename(p) for p in output_files()]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(payload)
    emit_figures(results)
    log("done in %.1fs" % (time.time() - _T0))

    gaps = payload["en_it_f1_gap_at_f1_opt"]
    mg = payload["mean_en_it_f1_gap"]
    print("\nBest val AUC per (backbone, method):")
    for bb in BACKBONES:
        print("  == %s ==" % BB_LABEL[bb])
        for m in METHODS:
            t = tuning[bb][m]
            print("    %-8s valAUC=%s  valF1=%.4f | IT F1=%.4f@t=%.2f  dF1(EN-IT)=%+.4f"
                  % (m, _auc(t["best_val_auc"]), t["best_val_macro_f1"],
                     results[bb][m]["it"]["at_f1_opt"]["f1"], gaps[bb][m]["f1_opt_t_it"],
                     gaps[bb][m]["delta_f1"]))
        print("    ranking by val AUC : %s"
              % ", ".join(payload["val_auc_best_method_ranked"][bb]))
        print("    ranking by IT test : %s"
              % ", ".join(payload["italian_best_method_ranked"][bb]))
    print("\nMean EN-IT gap: 210m %+.4f -> 610m %+.4f (change %+.4f)"
          % (mg["210m"], mg["610m"], mg["610m"] - mg["210m"]))
    return payload


if __name__ == "__main__":
    main()
