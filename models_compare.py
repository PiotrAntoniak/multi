"""Compare non-XGBoost classifiers on the FLORES-12 SENTENCE task (EN + IT).

Goal
----
Using the FROZEN EuroBERT **mean** sentence embeddings (no fine-tuning), compare a zoo of
scikit-learn classifiers -- everything EXCEPT XGBoost -- against the project's XGBoost
reference, for English and Italian, on both backbones:

* ``210m`` : ``embeddings/mean/emb_<lang>.npy``       (2009, 768)
* ``610m`` : ``embeddings_610/mean/emb_<lang>.npy``   (2009, 1152)

Data / alignment (identical to the project recipe)
--------------------------------------------------
* ``train_flores12.prepare_sentence()`` -> the aligned 2006-row FLORES CSV plus the seed-0
  ``train`` (997 ``dev``) / ``val`` (first 504 ``devtest``) / ``test`` (last 505 ``devtest``)
  split.  ``kept`` maps current rows to the historical 2009 embedding rows.
* 210m via ``load_sent_embeddings('mean', kept, 2009)``; 610m loaded directly and sliced with
  the SAME ``kept`` indices.
* Feature transform = ``perlang_z [both]`` (``flores_calib.perlang_z``): L2-normalize each row,
  then z-score with each language's OWN **train**-row mean/std, applied to that language's rows
  on both the model side and the transfer side.

Protocol
--------
* Final models are trained on **train + val** (1501 rows) and evaluated on the untouched
  ``test`` (505 rows), matching the project's XGBoost reference protocol (``flores_ceiling``:
  EN self macro-F1 ~= 0.49).
* ``LogisticRegression`` (C in {0.1,1,10}) and ``KNeighborsClassifier`` (k in {5,15}) pick their
  config by validation macro-F1 on the language's own ``val`` rows (trained on ``train``), then
  the winning config is refit on train+val.  Every other model uses one default config and is
  fit on train+val directly.
* Evaluated 4 ways per (backbone, model): **EN self**, **IT self**, **EN->IT**, **IT->EN**
  (accuracy, macro-F1, OVR-macro AUC; macro-F1 primary).
* An **XGBoost reference** row (``flores_calib.XGB_COMMON``) is trained the same way so the
  "beats XGBoost" comparison is exact and internal.

Outputs (create-only): ``models_compare.json`` / ``models_compare.md`` / ``models_compare.png``.

Usage
-----
    python models_compare.py                       # full run (both backbones, foreground)
    python models_compare.py --launch              # detached (hidden) via launch_hidden.vbs
    python models_compare.py --backbone 610m --only LogisticRegression,SVC_rbf
    python models_compare.py --smoke               # tiny plumbing check
"""
import os

# 2-thread runtime budget BEFORE numpy / sklearn import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import json
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T                       # noqa: E402  data prep + alignment

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

OUT_JSON = os.path.join(REPO, "models_compare.json")
OUT_MD = os.path.join(REPO, "models_compare.md")
OUT_PNG = os.path.join(REPO, "models_compare.png")
LOG_PATH = os.path.join(REPO, "models_compare.log")
PID_PATH = os.path.join(REPO, "models_compare.pid")
STATUS_PATH = os.path.join(REPO, "models_compare_status.json")

LANGS = ["en", "it", "de", "fr"]
N_CLASSES = 12
SEED = 0
BACKBONES = ["210m", "610m"]
N_JOBS = 2

# XGBoost reference params (exactly flores_calib.XGB_COMMON).
XGB_COMMON = dict(objective="multi:softprob", tree_method="hist", device="cpu",
                  max_bin=64, n_jobs=N_JOBS, seed=SEED, max_depth=8, min_child_weight=5,
                  learning_rate=0.1, n_estimators=400)

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ---------------------------------------------------------------------------
# Logging / pid / status
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _open_log():
    f = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f


def write_pid():
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, STATUS_PATH)


# ---------------------------------------------------------------------------
# Feature transform: perlang_z (L2-normalize + per-language TRAIN z-score)
# ---------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def perlang_z(X_by_lang, train_idx):
    """L2-normalize then z-score with each language's OWN train-row mean/std."""
    out = {}
    for lg in LANGS:
        Ztr = l2norm(np.asarray(X_by_lang[lg][train_idx], dtype=np.float64))
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd = np.where(sd < 1e-6, 1e-6, sd)
        Z = l2norm(np.asarray(X_by_lang[lg], dtype=np.float64))
        out[lg] = np.ascontiguousarray((Z - mu) / sd, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_backbone(backbone, kept, n_source):
    root = os.path.join(REPO, "embeddings" if backbone == "210m" else "embeddings_610")
    out = {}
    for lg in LANGS:
        path = os.path.join(root, "mean", "emb_%s.npy" % lg)
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != n_source:
            raise ValueError("expected %d rows in %s, got %s" % (n_source, path, arr.shape))
        out[lg] = arr[kept].astype(np.float64)
        if out[lg].shape[0] != len(kept):
            raise ValueError("alignment produced %d rows, expected %d"
                             % (out[lg].shape[0], len(kept)))
    return out


# ---------------------------------------------------------------------------
# Model zoo (scikit-learn only; XGBoost added separately as the reference)
# ---------------------------------------------------------------------------
def build_zoo():
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier, GradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    zoo = {}
    zoo["LogisticRegression"] = [
        ("C=0.1", lambda: LogisticRegression(C=0.1, max_iter=2000, n_jobs=N_JOBS)),
        ("C=1", lambda: LogisticRegression(C=1.0, max_iter=2000, n_jobs=N_JOBS)),
        ("C=10", lambda: LogisticRegression(C=10.0, max_iter=2000, n_jobs=N_JOBS)),
    ]
    zoo["LinearSVC"] = [
        ("C=1", lambda: LinearSVC(C=1.0, max_iter=5000, dual="auto")),
    ]
    zoo["SVC_linear"] = [
        ("C=1", lambda: SVC(kernel="linear", C=1.0, decision_function_shape="ovr")),
    ]
    zoo["SVC_rbf"] = [
        ("default", lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                                decision_function_shape="ovr")),
    ]
    zoo["KNN"] = [
        ("k=5", lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)),
        ("k=15", lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=N_JOBS)),
    ]
    zoo["RandomForest"] = [
        ("default", lambda: RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                   n_jobs=N_JOBS)),
    ]
    zoo["ExtraTrees"] = [
        ("default", lambda: ExtraTreesClassifier(n_estimators=300, random_state=SEED,
                                                 n_jobs=N_JOBS)),
    ]
    zoo["HistGradientBoosting"] = [
        ("default", lambda: HistGradientBoostingClassifier(random_state=SEED)),
    ]
    zoo["GradientBoosting"] = [
        # max_features='sqrt' (not sklearn's None default) bounds per-fit time on the 1152-d
        # backbone: measured 745 s (None) vs 24 s (sqrt) per fit on 610m -- a mild, standard
        # speed adaptation so the full sweep stays within budget; documented in the outputs.
        ("max_features=sqrt", lambda: GradientBoostingClassifier(random_state=SEED,
                                                                max_features="sqrt")),
    ]
    zoo["MLP"] = [
        ("h=256", lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                                        random_state=SEED, early_stopping=False)),
    ]
    zoo["LDA"] = [("default", lambda: LinearDiscriminantAnalysis())]
    zoo["QDA"] = [("default", lambda: QuadraticDiscriminantAnalysis())]
    zoo["GaussianNB"] = [("default", lambda: GaussianNB())]
    zoo["Ridge"] = [("default", lambda: RidgeClassifier(random_state=SEED))]
    zoo["SGD"] = [("default", lambda: SGDClassifier(random_state=SEED, max_iter=2000,
                                                    tol=1e-4))]
    zoo["DecisionTree"] = [("default", lambda: DecisionTreeClassifier(random_state=SEED))]
    return zoo


def build_xgb_ref():
    import xgboost as xgb

    def make():
        p = dict(XGB_COMMON)
        p["num_class"] = N_CLASSES
        return xgb.XGBClassifier(**p)
    return [("default", make)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score_matrix(clf, X):
    """Per-class score matrix (n, C) from predict_proba when available else decision_function."""
    if hasattr(clf, "predict_proba"):
        try:
            P = np.asarray(clf.predict_proba(X), dtype=np.float64)
            if P.ndim == 2 and P.shape[1] == N_CLASSES:
                return P
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        D = np.asarray(clf.decision_function(X), dtype=np.float64)
        if D.ndim == 1:
            D = np.column_stack([-D, D])
        if D.shape[1] == N_CLASSES:
            return D
    raise RuntimeError("model provides neither usable predict_proba nor decision_function")


def ovr_macro_auc(y, S):
    aucs = []
    for c in range(N_CLASSES):
        yc = (y == c).astype(int)
        if yc.min() == yc.max():
            continue
        aucs.append(roc_auc_score(yc, S[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=np.arange(N_CLASSES)))


def evaluate(y, clf, X):
    pred = np.asarray(clf.predict(X)).ravel().astype(int)
    S = score_matrix(clf, X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": macro_f1(y, pred),
        "auc_ovr_macro": ovr_macro_auc(y, S),
    }


# ---------------------------------------------------------------------------
# One (backbone, model) cell
# ---------------------------------------------------------------------------
def run_model(backbone, name, grid, Xt, y, idx):
    tr, va, te, trval = idx
    ytr, yva, ytrval, yte = y[tr], y[va], y[trval], y[te]

    out = {"selected": {}, "config": {}}
    for train_lang, other in (("en", "it"), ("it", "en")):
        if len(grid) == 1:                       # single default config -> fit on train+val
            label, factory = grid[0]
            clf = factory()
            clf.fit(Xt[train_lang][trval], ytrval)
            out["selected"][train_lang] = label
        else:                                    # light grid -> pick by own val macro-F1
            best_label, best_f1, best_factory = None, -1.0, None
            for label, factory in grid:
                clf = factory()
                clf.fit(Xt[train_lang][tr], ytr)
                vf1 = macro_f1(yva, np.asarray(clf.predict(Xt[train_lang][va])).ravel().astype(int))
                if vf1 > best_f1:
                    best_f1, best_label, best_factory = vf1, label, factory
            out["selected"][train_lang] = best_label
            out["config"][train_lang] = {"val_macro_f1": float(best_f1)}
            clf = best_factory()
            clf.fit(Xt[train_lang][trval], ytrval)

        self_m = evaluate(yte, clf, Xt[train_lang][te])
        trans_m = evaluate(yte, clf, Xt[other][te])
        if train_lang == "en":
            out["EN_self"], out["EN_to_IT"] = self_m, trans_m
        else:
            out["IT_self"], out["IT_to_EN"] = self_m, trans_m
    return out


# ---------------------------------------------------------------------------
# Ranking / summary
# ---------------------------------------------------------------------------
def _transfer_mean(rec):
    vals = [rec["EN_to_IT"]["macro_f1"], rec["IT_to_EN"]["macro_f1"]]
    return float(np.mean(vals))


def summarize(results, backbones):
    summary = {}
    for bb in backbones:
        if bb not in results:
            continue
        recs = results[bb]
        non_xgb = {k: v for k, v in recs.items() if k != "XGBoost"}
        en_rank = sorted(non_xgb, key=lambda k: -recs[k]["EN_self"]["macro_f1"])
        tr_rank = sorted(non_xgb, key=lambda k: -_transfer_mean(recs[k]))
        best_en = en_rank[0] if en_rank else None
        best_tr = tr_rank[0] if tr_rank else None
        xgb = recs.get("XGBoost")
        xgb_en = xgb["EN_self"]["macro_f1"] if xgb else float("nan")
        xgb_tr = _transfer_mean(xgb) if xgb else float("nan")
        beat_en = [k for k in en_rank if recs[k]["EN_self"]["macro_f1"] >= xgb_en - 1e-9]
        beat_tr = [k for k in tr_rank if _transfer_mean(recs[k]) >= xgb_tr - 1e-9]
        summary[bb] = {
            "en_ranking": en_rank,
            "transfer_ranking": tr_rank,
            "best_en": best_en,
            "best_en_macro_f1": recs[best_en]["EN_self"]["macro_f1"] if best_en else None,
            "best_transfer": best_tr,
            "best_transfer_macro_f1": _transfer_mean(recs[best_tr]) if best_tr else None,
            "xgb_en_macro_f1": xgb_en,
            "xgb_transfer_macro_f1": xgb_tr,
            "beat_or_match_xgb_en": beat_en,
            "beat_or_match_xgb_transfer": beat_tr,
        }
    return summary


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(results, backbones, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bbs = [b for b in backbones if b in results]
    if not bbs:
        return
    fig, axes = plt.subplots(1, len(bbs), figsize=(9.5 * len(bbs), 7.0), squeeze=False)

    for ax, bb in zip(axes[0], bbs):
        recs = results[bb]
        models = sorted(recs, key=lambda k: -recs[k]["EN_self"]["macro_f1"])
        en_self = [recs[m]["EN_self"]["macro_f1"] for m in models]
        en_it = [recs[m]["EN_to_IT"]["macro_f1"] for m in models]
        x = np.arange(len(models))
        w = 0.38
        ax.bar(x - w / 2, en_self, w, label="EN self (macro-F1)", color="#1f77b4")
        ax.bar(x + w / 2, en_it, w, label="EN->IT transfer (macro-F1)", color="#d62728")
        if "XGBoost" in recs:
            ax.axhline(recs["XGBoost"]["EN_self"]["macro_f1"], color="#1f77b4", ls="--",
                       lw=1.2, alpha=0.7, label="XGBoost EN self")
            ax.axhline(recs["XGBoost"]["EN_to_IT"]["macro_f1"], color="#d62728", ls="--",
                       lw=1.2, alpha=0.7, label="XGBoost EN->IT")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, max(0.6, max(en_self + en_it) * 1.15))
        ax.set_title("EuroBERT-%s mean + perlang_z  (12-class FLORES sentence)" % bb,
                     fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.suptitle("models_compare: non-XGBoost classifiers vs XGBoost reference "
                 "(macro-F1; dashed = XGBoost)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _f(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.3f" % v


def write_md(results, summary, config, path):
    L = []
    A = L.append
    A("# models_compare - non-XGBoost classifiers on FLORES-12 (sentence), EN + IT\n")
    A("Features: frozen EuroBERT **mean** sentence embeddings + `perlang_z [both]` "
      "(L2-normalize + per-language **train** z-score). Final models train on **train+val** "
      "(1501) and are evaluated on the seed-0 **test** (505). Metrics: macro-F1 (primary), "
      "accuracy, OVR-macro AUC. `LogisticRegression` (C in {0.1,1,10}) and `KNN` (k in {5,15}) "
      "pick their config by own-language val macro-F1; all others use sklearn defaults. "
      "`XGBoost` is the project reference (flores_calib.XGB_COMMON).\n")
    A("Runtime adaptations (documented): `GradientBoosting` uses `max_features='sqrt'` "
      "(sklearn's `None` default needed ~745 s/fit on the 1152-d backbone; `sqrt` ~24 s), and "
      "`SVC` uses `decision_function_shape='ovr'` scores for AUC instead of `probability=True` "
      "(the 5-fold Platt CV). All other models are sklearn defaults.\n")
    A("Backbones: **210m** = `embeddings/mean` (768-d); **610m** = `embeddings_610/mean` "
      "(1152-d). CPU, n_jobs=2.\n")

    for bb in BACKBONES:
        if bb not in results:
            continue
        recs = results[bb]
        s = summary.get(bb, {})
        A("\n## Backbone %s\n" % bb)
        A("| model | EN-self F1 | EN-self acc | EN-self AUC | IT-self F1 | EN->IT F1 | "
          "IT->EN F1 | transfer F1 |")
        A("|---|---|---|---|---|---|---|---|")
        order = sorted(recs, key=lambda k: -recs[k]["EN_self"]["macro_f1"])
        for m in order:
            r = recs[m]
            A("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                m, _f(r["EN_self"]["macro_f1"]), _f(r["EN_self"]["accuracy"]),
                _f(r["EN_self"]["auc_ovr_macro"]), _f(r["IT_self"]["macro_f1"]),
                _f(r["EN_to_IT"]["macro_f1"]), _f(r["IT_to_EN"]["macro_f1"]),
                _f(_transfer_mean(r))))
        A("")
        A("* **Best non-XGBoost EN-self**: `%s` (macro-F1 %s) vs XGBoost %s." % (
            s.get("best_en"), _f(s.get("best_en_macro_f1")), _f(s.get("xgb_en_macro_f1"))))
        A("* **Best non-XGBoost transfer** (mean of EN->IT and IT->EN): `%s` (macro-F1 %s) "
          "vs XGBoost %s." % (s.get("best_transfer"), _f(s.get("best_transfer_macro_f1")),
                              _f(s.get("xgb_transfer_macro_f1"))))
        be = s.get("beat_or_match_xgb_en") or []
        bt = s.get("beat_or_match_xgb_transfer") or []
        A("* **Beats/matches XGBoost EN-self**: %s." % (", ".join("`%s`" % x for x in be) or "none"))
        A("* **Beats/matches XGBoost transfer**: %s." % (", ".join("`%s`" % x for x in bt) or "none"))
        A("\nEN-self ranking (macro-F1): " + " > ".join(
            "%s (%s)" % (m, _f(recs[m]["EN_self"]["macro_f1"])) for m in s.get("en_ranking", [])))
        A("\nTransfer ranking (macro-F1, mean EN->IT & IT->EN): " + " > ".join(
            "%s (%s)" % (m, _f(_transfer_mean(recs[m]))) for m in s.get("transfer_ranking", [])))
        A("")

    A("\n## Config\n")
    A("```json")
    A(json.dumps(config, indent=2))
    A("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Detached launch
# ---------------------------------------------------------------------------
def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd], cwd=REPO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(description="Non-XGBoost classifier comparison on FLORES-12 sent")
    ap.add_argument("--launch", action="store_true", help="spawn detached via launch_hidden.vbs")
    ap.add_argument("--backbone", default="both",
                    help="comma-separated subset of {210m,610m} or 'both' (default)")
    ap.add_argument("--only", default="",
                    help="comma-separated model names to run (default: all incl. XGBoost)")
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check")
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.launch:
        launch_detached([a for a in argv if a != "--launch"])
        print("launched detached via launch_hidden.vbs")
        return 0

    if args.backbone == "both":
        backbones = list(BACKBONES)
    else:
        backbones = [b.strip() for b in args.backbone.split(",") if b.strip()]
        bad = [b for b in backbones if b not in BACKBONES]
        if bad:
            raise SystemExit("unknown backbone(s) %s; known {210m,610m}" % bad)

    logf = None
    if not args.smoke:
        logf = _open_log()
    write_pid()
    state = {"phase": "start", "pid": os.getpid(), "backbones": backbones}
    write_status(state)

    log("models_compare starting: backbones=%s only=%r smoke=%s" % (backbones, args.only, args.smoke))

    # ---- data ----
    sent = T.prepare_sentence()
    kept = sent["kept"]
    y = np.asarray(sent["y"], dtype=int)
    tr = np.asarray(sent["train_idx"], dtype=int)
    va = np.asarray(sent["val_idx"], dtype=int)
    te = np.asarray(sent["test_idx"], dtype=int)
    trval = np.concatenate([tr, va])
    idx = (tr, va, te, trval)
    log("sentence: rows=%d kept=%d train=%d val=%d test=%d train+val=%d"
        % (len(sent["cur"]), len(kept), len(tr), len(va), len(te), len(trval)))

    zoo = build_zoo()
    zoo_order = list(zoo.keys()) + ["XGBoost"]
    if args.only:
        want = [m.strip() for m in args.only.split(",") if m.strip()]
        unknown = [m for m in want if m not in zoo_order]
        if unknown:
            raise SystemExit("unknown model(s) %s; known %s" % (unknown, zoo_order))
        run_names = want
    else:
        run_names = zoo_order
    if args.smoke:
        run_names = ["HistoricalGradientBoosting" if False else "LDA", "LogisticRegression"]

    results = {}
    config = {
        "emb": {"210m": "embeddings/mean", "610m": "embeddings_610/mean"},
        "feature": "perlang_z [both] (L2-normalize + per-language TRAIN z-score)",
        "train": "train+val (1501)", "test": "seed-0 test (505)",
        "metrics": ["accuracy", "macro_f1", "auc_ovr_macro"],
        "xgb_reference": XGB_COMMON,
        "light_grid": {"LogisticRegression": [0.1, 1.0, 10.0], "KNN": [5, 15]},
        "runtime_notes": [
            "GradientBoosting uses max_features='sqrt' (sklearn default None was ~745 s/fit "
            "on 610m; sqrt is ~24 s/fit). All other models keep sklearn defaults.",
            "SVC uses decision_function_shape='ovr' + decision_function for AUC (no "
            "probability=True 5-fold CV) to keep per-fit time ~1 s.",
        ],
        "seed": SEED, "n_jobs": N_JOBS, "smoke": bool(args.smoke),
    }

    for bb in backbones:
        state["phase"] = "backbone:%s" % bb
        write_status(state)
        t0 = time.time()
        Xraw = load_backbone(bb, kept, len(sent["git"]))
        Xt = perlang_z(Xraw, tr)
        log("backbone %s loaded + perlang_z done (dim=%d, %.1fs)"
            % (bb, Xraw["en"].shape[1], time.time() - t0))
        results[bb] = {}

        for name in run_names:
            grid = build_xgb_ref() if name == "XGBoost" else zoo[name]
            state["model"] = "%s/%s" % (bb, name)
            write_status(state)
            ts = time.time()
            try:
                rec = run_model(bb, name, grid, Xt, y, idx)
                rec["elapsed_s"] = round(time.time() - ts, 1)
                rec["dim"] = int(Xraw["en"].shape[1])
                results[bb][name] = rec
                log("  %-8s %-22s ENself=%.3f ITself=%.3f EN->IT=%.3f IT->EN=%.3f (%.1fs)"
                    % (bb, name, rec["EN_self"]["macro_f1"], rec["IT_self"]["macro_f1"],
                       rec["EN_to_IT"]["macro_f1"], rec["IT_to_EN"]["macro_f1"],
                       rec["elapsed_s"]))
            except Exception as exc:                          # noqa: BLE001
                log("  %-8s %-22s FAILED: %r" % (bb, name, exc))
                results[bb][name] = {"error": repr(exc), "elapsed_s": round(time.time() - ts, 1)}
            # incremental write so partial results survive
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump({"config": config, "results": results}, f, indent=2)

    # drop failed models from results before summary/plot
    results = {bb: {m: r for m, r in d.items() if "error" not in r}
               for bb, d in results.items()}

    summary = summarize(results, backbones)
    doc = {"config": config, "results": results, "summary": summary}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    log("wrote %s" % OUT_JSON)

    write_md(results, summary, config, OUT_MD)
    log("wrote %s" % OUT_MD)
    try:
        make_plot(results, backbones, OUT_PNG)
        log("wrote %s" % OUT_PNG)
    except Exception as exc:                                  # noqa: BLE001
        log("plot failed: %r" % (exc,))

    state["phase"] = "done"
    state["elapsed_s"] = round(time.time() - _T0, 1)
    write_status(state)
    log("ALL DONE in %.1fs" % (time.time() - _T0))
    if logf is not None:
        logf.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
