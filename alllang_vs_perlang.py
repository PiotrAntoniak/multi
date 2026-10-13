r"""alllang_vs_perlang.py -- does training ONE model on ALL languages beat training one model
per language?  Two problems, kept strictly separate (their data is never mixed):

  * FLORES  : the 12-class ``sent`` topic set and the 11-class ``agg`` set (via
              ``train_flores12.prepare_sentence`` / ``prepare_agg`` -- their alignment and splits).
  * TOXICITY: the binary ``data/toxi_clean/{en,it,de,fr,es}_{train,validation,test}.parquet`` sets.

Two training regimes, compared on every language's held-out TEST split:
  * ``all-langs`` : ONE model trained on the pooled TRAIN rows of every language of the dataset.
  * ``per-lang``  : one model trained per language on its own TRAIN rows (in-language).

Features
--------
``perlang_z`` applied to EVERY row of BOTH regimes: L2-normalize the vector, then standardize
with that ROW'S OWN language's TRAIN mean/std (L2-normalize first, then z-score).  It is applied
row-by-row before any pooling, so the pooled ``all-langs`` training matrix is a stack of
per-language-normalized rows -- this is what makes the pooled regime well-posed.

Model
-----
XGBoost, fixed params (no Optuna): tree_method=hist, max_bin=64, device=cpu, n_jobs=2, seed=0,
max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400.
``multi:softprob`` (num_class = 12/11) for FLORES; ``binary:logistic`` for toxicity.

Evaluation
----------
Multiclass: argmax accuracy, macro-F1, micro-F1, OVR-macro AUC (per-class AUC averaged, degenerate
classes skipped).  Binary: accuracy, macro-F1, positive-class F1, ROC-AUC (threshold 0.5).
Macro averages are the plain mean over the dataset's languages.

Embeddings
----------
Read ONLY from the bf16-consistent cache ``emb_bf16/`` (masked-mean pooling, bf16 forward -> fp32
storage), which already holds every needed array for both backbones (210m 768-d, 610m 1152-d):
  emb_bf16/flores_sent/<bb>/mean/emb_<lang>.npy      (aligned to the current 2006-row CSV)
  emb_bf16/flores_agg/<bb>/mean/emb_<lang>.npy
  emb_bf16/toxic/<bb>/emb_<lang>_{train,validation,test}.npy

Outputs (create-only, these four files only)
--------------------------------------------
alllang_vs_perlang.py / alllang_vs_perlang.json / alllang_vs_perlang.md / alllang_vs_perlang.png
Operational pid/log/status live under %TEMP%\opencode\alllang_vs_perlang\.

Usage
-----
    python alllang_vs_perlang.py                 # foreground full run
    python alllang_vs_perlang.py --launch        # detached hidden run (no window)
    python alllang_vs_perlang.py --smoke         # tiny plumbing check
    python alllang_vs_perlang.py --render        # rebuild md/png from the JSON
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import subprocess
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BF16_ROOT = os.path.join(REPO, "emb_bf16")

OUT_JSON = os.path.join(REPO, "alllang_vs_perlang.json")
OUT_MD = os.path.join(REPO, "alllang_vs_perlang.md")
OUT_PNG = os.path.join(REPO, "alllang_vs_perlang.png")

TMP = os.path.join(os.environ.get("TEMP", REPO), "opencode", "alllang_vs_perlang")
os.makedirs(TMP, exist_ok=True)
LOG_PATH = os.path.join(TMP, "alllang_vs_perlang.log")
PID_PATH = os.path.join(TMP, "alllang_vs_perlang.pid")
STATUS_PATH = os.path.join(TMP, "alllang_vs_perlang_status.json")

SEED = 0
XGB_COMMON = dict(tree_method="hist", max_bin=64, device="cpu", n_jobs=2, seed=SEED,
                  max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)

# dataset key -> (kind, order in the report/figure)
DATASET_KIND = {"flores_sent": "mc", "flores_agg": "mc", "toxic": "binary"}
DATASET_ORDER = ["flores_sent", "flores_agg", "toxic"]
BACKBONE_ORDER = ["210m", "610m"]
LANGS_FLORES = ["en", "it", "de", "fr"]
LANGS_TOXI = ["en", "it", "de", "fr", "es"]
# metric keys per kind (shared three drive the figure)
MC_KEYS = ["accuracy", "macro_f1", "micro_f1", "auc"]
BIN_KEYS = ["accuracy", "macro_f1", "f1", "auc"]
HEADLINE = {"mc": "macro_f1", "binary": "macro_f1"}
FIG_KEYS = ["accuracy", "macro_f1", "auc"]
FIG_LABEL = {"accuracy": "accuracy", "macro_f1": "macro-F1", "auc": "OVR-AUC"}

_T0 = time.time()
_SMOKE = False
_LIMIT = 0
_TAGGED = False   # True when writing a per-pair shard into TMP (--tag); skips md/png render


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % (type(o),))


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    try:
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Feature transform: perlang_z [both]  (L2-normalize, then own-train z-score)
# ---------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def perlang_z(ds):
    """L2-normalize every row, then z-score it with its own language's TRAIN mean/std.

    Stats are fit on the L2-normalized TRAIN rows of that language only (model-free: they never
    touch labels).  Returns new per-split dicts of float32 arrays.
    """
    Ztr, Zva, Zte = {}, {}, {}
    for lg in ds["langs"]:
        Z = l2norm(np.asarray(ds["Xtr"][lg], dtype=np.float64))
        mu, sd = Z.mean(0), Z.std(0)
        sd = np.where(sd < 1e-6, 1e-6, sd)
        Ztr[lg] = ((l2norm(np.asarray(ds["Xtr"][lg], dtype=np.float64)) - mu) / sd).astype(np.float32)
        Zva[lg] = ((l2norm(np.asarray(ds["Xva"][lg], dtype=np.float64)) - mu) / sd).astype(np.float32)
        Zte[lg] = ((l2norm(np.asarray(ds["Xte"][lg], dtype=np.float64)) - mu) / sd).astype(np.float32)
    return {"Xtr": Ztr, "Xva": Zva, "Xte": Zte}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc_ovr_macro(y, P, C):
    aucs = []
    for c in range(C):
        yc = (y == c).astype(int)
        if yc.min() == yc.max():
            continue
        aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def mc_metrics(y, P, C):
    pred = P.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(C)), zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", labels=list(range(C)), zero_division=0)),
        "auc": auc_ovr_macro(y, P, C),
    }


def bin_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    if y.min() == y.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, pred, average="binary", zero_division=0)),
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Dataset loaders (row order == embedding row order; nothing reindexed)
# ---------------------------------------------------------------------------
def _rng_idx(n):
    if _LIMIT and n > _LIMIT:
        return np.arange(_LIMIT)
    return np.arange(n)


def load_flores(ds, backbone):
    import train_flores12 as T
    sent = T.prepare_sentence()
    classes = sent["classes"]
    if ds == "flores_sent":
        base = os.path.join(BF16_ROOT, "flores_sent", backbone, "mean")
        X = {lg: np.load(os.path.join(base, "emb_%s.npy" % lg)) for lg in LANGS_FLORES}
        tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
        y, n_classes = sent["y"], 12
        n_rows = len(sent["cur"])
    else:
        agg = T.prepare_agg(sent["cur"], classes)
        base = os.path.join(BF16_ROOT, "flores_agg", backbone, "mean")
        X = {lg: np.load(os.path.join(base, "emb_%s.npy" % lg)) for lg in LANGS_FLORES}
        tr, va, te = agg["train_idx"], agg["val_idx"], agg["test_idx"]
        y, n_classes = agg["y"], agg["n_classes"]
        n_rows = len(agg["agg"])
    for lg in LANGS_FLORES:
        if X[lg].shape[0] != n_rows:
            raise ValueError("%s/%s: %s has %d rows, expected %d"
                             % (ds, backbone, lg, X[lg].shape[0], n_rows))
    return {
        "name": ds, "kind": "mc", "n_classes": int(n_classes), "langs": list(LANGS_FLORES),
        "classes": classes if ds == "flores_sent" else agg["classes"],
        "Xtr": {lg: np.asarray(X[lg][tr]) for lg in LANGS_FLORES},
        "Xva": {lg: np.asarray(X[lg][va]) for lg in LANGS_FLORES},
        "Xte": {lg: np.asarray(X[lg][te]) for lg in LANGS_FLORES},
        # labels are shared across languages (parallel CSV): same y per row for every language
        "ytr": {lg: y[tr] for lg in LANGS_FLORES},
        "yva": {lg: y[va] for lg in LANGS_FLORES},
        "yte": {lg: y[te] for lg in LANGS_FLORES},
        "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
    }


def load_toxic(backbone):
    import pandas as pd
    root = os.path.join(BF16_ROOT, "toxic", backbone)
    pq = os.path.join(REPO, "data", "toxi_clean")
    Xtr, Xva, Xte, ytr, yva, yte = {}, {}, {}, {}, {}, {}
    sizes = {"train": 0, "val": 0, "test": 0}
    for lg in LANGS_TOXI:
        df = pd.read_parquet(os.path.join(pq, "%s_train.parquet" % lg))
        dv = pd.read_parquet(os.path.join(pq, "%s_validation.parquet" % lg))
        dt = pd.read_parquet(os.path.join(pq, "%s_test.parquet" % lg))
        tr, va, te = _rng_idx(len(df)), _rng_idx(len(dv)), _rng_idx(len(dt))
        a = np.load(os.path.join(root, "emb_%s_train.npy" % lg))
        b = np.load(os.path.join(root, "emb_%s_validation.npy" % lg))
        c = np.load(os.path.join(root, "emb_%s_test.npy" % lg))
        if a.shape[0] != len(df) or b.shape[0] != len(dv) or c.shape[0] != len(dt):
            raise ValueError("toxic/%s/%s row mismatch: %s vs %s/%s/%s"
                             % (backbone, lg, (a.shape[0], b.shape[0], c.shape[0]),
                                len(df), len(dv), len(dt)))
        Xtr[lg], Xva[lg], Xte[lg] = a[tr], b[va], c[te]
        ytr[lg] = df["is_toxic"].astype(int).to_numpy()[tr]
        yva[lg] = dv["is_toxic"].astype(int).to_numpy()[va]
        yte[lg] = dt["is_toxic"].astype(int).to_numpy()[te]
        sizes["train"] += len(tr); sizes["val"] += len(va); sizes["test"] += len(te)
    return {"name": "toxic", "kind": "binary", "n_classes": None, "langs": list(LANGS_TOXI),
            "classes": ["not_toxic", "toxic"],
            "Xtr": Xtr, "Xva": Xva, "Xte": Xte, "ytr": ytr, "yva": yva, "yte": yte,
            "sizes": sizes}


def load_dataset(ds, backbone):
    if ds in ("flores_sent", "flores_agg"):
        return load_flores(ds, backbone)
    if ds == "toxic":
        return load_toxic(backbone)
    raise ValueError("unknown dataset %r" % ds)


# ---------------------------------------------------------------------------
# Train / evaluate the two regimes
# ---------------------------------------------------------------------------
def _fit_clf(kind, n_classes, X, y):
    import xgboost as xgb
    p = dict(XGB_COMMON)
    if kind == "mc":
        p["objective"] = "multi:softprob"; p["num_class"] = int(n_classes)
    else:
        p["objective"] = "binary:logistic"
    if _SMOKE:
        p["n_estimators"] = 30
    clf = xgb.XGBClassifier(**p)
    clf.fit(X, y)
    return clf


def _predict(clf, kind, X):
    if kind == "mc":
        return clf.predict_proba(X)                 # (n, C) softprob
    return clf.predict_proba(X)[:, 1]               # (n,) positive-class prob


def _score(kind, y, pred, C):
    return mc_metrics(y, pred, C) if kind == "mc" else bin_metrics(y, pred)


def evaluate(ds, Z):
    """Return {'per_lang': {lg: metrics}, 'all_langs': {lg: metrics}} for one dataset object.

    per_lang[lg]  = the model trained on language lg's own TRAIN rows, scored on lg's TEST rows.
    all_langs[lg] = the single pooled-TRAIN model, scored on lg's TEST rows.
    """
    kind, C, langs = ds["kind"], ds["n_classes"], ds["langs"]

    # --- per-lang regime ---
    per_lang = {}
    for lg in langs:
        clf = _fit_clf(kind, C, Z["Xtr"][lg], ds["ytr"][lg])
        per_lang[lg] = _score(kind, ds["yte"][lg], _predict(clf, kind, Z["Xte"][lg]), C)

    # --- all-langs regime: ONE model on the pooled TRAIN rows of every language ---
    Xall = np.vstack([Z["Xtr"][lg] for lg in langs])
    yall = np.concatenate([np.asarray(ds["ytr"][lg]) for lg in langs])
    clf_all = _fit_clf(kind, C, Xall, yall)
    all_langs = {lg: _score(kind, ds["yte"][lg], _predict(clf_all, kind, Z["Xte"][lg]), C)
                 for lg in langs}
    return {"per_lang": per_lang, "all_langs": all_langs,
            "pooled_train_rows": int(Xall.shape[0])}


def _mean_over(metrics_by_lang, langs, keys):
    return {k: float(np.nanmean([metrics_by_lang[lg][k] for lg in langs])) for k in keys}


def macro_and_delta(D, kind, langs):
    keys = MC_KEYS if kind == "mc" else BIN_KEYS
    macro = {"per_lang": _mean_over(D["per_lang"], langs, keys),
             "all_langs": _mean_over(D["all_langs"], langs, keys)}
    hl = HEADLINE[kind]
    delta = {lg: {k: D["all_langs"][lg][k] - D["per_lang"][lg][k] for k in keys} for lg in langs}
    macro_delta = {k: macro["all_langs"][k] - macro["per_lang"][k] for k in keys}
    wins = sum(1 for lg in langs if delta[lg][hl] > 0)
    losses = sum(1 for lg in langs if delta[lg][hl] < 0)
    verdict = ("all-langs helps" if macro_delta[hl] > 0.002
               else "per-lang helps" if macro_delta[hl] < -0.002 else "about equal")
    return {"macro": macro, "delta": delta, "macro_delta": macro_delta,
            "headline": hl, "wins": wins, "losses": losses, "verdict": verdict}


# ---------------------------------------------------------------------------
# JSON / MD / PNG
# ---------------------------------------------------------------------------
def load_json():
    if os.path.isfile(OUT_JSON):
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"datasets": {}, "meta": {}}


def save_json(doc):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=_json_default)
    os.replace(tmp, OUT_JSON)


def _fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return ("%." + str(nd) + "f") % v
    except Exception:
        return str(v)


def _sgn(v, nd=3):
    try:
        if v is None or not np.isfinite(v):
            return "n/a"
        return ("%+." + str(nd) + "f") % v
    except Exception:
        return str(v)


def render(doc=None):
    if doc is None:
        doc = load_json()
    datasets = doc.get("datasets", {})
    order = [d for d in DATASET_ORDER if d in datasets]
    if not order:
        log("render: nothing to render")
        return
    lines = []
    lines.append("# all-langs vs per-lang training -- EuroBERT mean pooling, fixed XGBoost\n")
    lines.append("FLORES (sent + agg) and TOXICITY are two SEPARATE problems (their data is never "
                 "mixed). For each dataset and backbone: **all-langs** = ONE XGBoost trained on the "
                 "pooled TRAIN rows of every language; **per-lang** = one XGBoost trained per "
                 "language on its own TRAIN rows. Both are scored on EVERY language's held-out "
                 "TEST split. Features: `perlang_z [both]` on every row (L2-normalize, then z-score "
                 "with that row's own language TRAIN mean/std). Params fixed (hist, max_bin=64, cpu, "
                 "n_jobs=2, seed=0, depth 8, mcw 5, lr 0.1, n400); no Optuna. Metrics: accuracy, "
                 "macro-F1, micro-F1 / positive-class F1, OVR-macro AUC. `delta = all-langs - "
                 "per-lang`; headline is macro-F1.\n")
    for d in order:
        res = datasets[d]
        for bb in BACKBONE_ORDER:
            if bb not in res["backbones"]:
                continue
            B = res["backbones"][bb]
            langs = res["langs"]; kind = res["kind"]
            keys = MC_KEYS if kind == "mc" else BIN_KEYS
            hl = B["headline"]
            lines.append("## %s / %s  (%s%s)\n"
                         % (d, bb, "multiclass" if kind == "mc" else "binary",
                            "" if kind == "mc" else ""))
            lines.append("pooled all-langs TRAIN rows = %d | per-lang train rows = %s | "
                         "test rows (per language) = %s\n"
                         % (B["pooled_train_rows"], B["per_lang_train_rows"],
                            B["test_rows"]))
            for key in keys:
                mark = " (**headline**)" if key == hl else ""
                lines.append("**%s%s**\n" % (key, mark))
                lines.append("| language | per-lang | all-langs | delta (all-per) |")
                lines.append("|---|---|---|---|")
                for lg in langs:
                    lines.append("| %s | %s | %s | %s |"
                                 % (lg, _fmt(B["per_lang"][lg][key]),
                                    _fmt(B["all_langs"][lg][key]),
                                    _sgn(B["delta"][lg][key])))
                lines.append("| **macro** | **%s** | **%s** | **%s** |"
                             % (_fmt(B["macro"]["per_lang"][key]),
                                _fmt(B["macro"]["all_langs"][key]),
                                _sgn(B["macro_delta"][key])))
                lines.append("")
            lines.append("verdict (%s macro): **%s** -- macro delta %s; "
                         "per-language wins(all>per)=%d losses=%d out of %d.\n"
                         % (hl, B["verdict"], _sgn(B["macro_delta"][hl]),
                            B["wins"], B["losses"], len(langs)))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("wrote %s" % OUT_MD)

    # --- figure: 3 panels (accuracy / macro-F1 / AUC), 6 groups, 2 bars each ---
    groups = [(d, bb) for d in order for bb in BACKBONE_ORDER
              if bb in datasets[d].get("backbones", {})]
    fig, axes = plt.subplots(1, len(FIG_KEYS), figsize=(6.4 * len(FIG_KEYS), 4.8), squeeze=False)
    x = np.arange(len(groups))
    w = 0.38
    for ax, key in zip(axes[0], FIG_KEYS):
        pl = [datasets[d]["backbones"][bb]["macro"]["per_lang"][key] for (d, bb) in groups]
        al = [datasets[d]["backbones"][bb]["macro"]["all_langs"][key] for (d, bb) in groups]
        ax.bar(x - w / 2, pl, w, label="per-lang", color="#1f77b4")
        ax.bar(x + w / 2, al, w, label="all-langs", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(["%s\n%s" % (d.replace("flores_", ""), bb) for (d, bb) in groups],
                           fontsize=8)
        ax.set_title(FIG_LABEL[key], fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("macro-average over languages")
    axes[0][-1].legend(fontsize=9)
    fig.suptitle("all-langs vs per-lang: macro-average metric per dataset x backbone "
                 "(perlang_z, fixed XGBoost)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=130)
    log("wrote %s" % OUT_PNG)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(pairs):
    doc = load_json() if not _SMOKE else {"datasets": {}, "meta": {}}
    doc.setdefault("datasets", {})
    doc["meta"] = {"xgb_common": XGB_COMMON, "seed": SEED, "smoke": _SMOKE,
                   "features": "perlang_z [both]",
                   "started": doc.get("meta", {}).get("started", time.strftime("%Y-%m-%d %H:%M:%S"))}
    for ds_name, bb in pairs:
        res = doc["datasets"].setdefault(ds_name, {"name": ds_name, "kind": DATASET_KIND[ds_name],
                                                   "langs": [], "backbones": {}})
        if bb in res["backbones"] and not _SMOKE:
            log("%s/%s: already present, skipping" % (ds_name, bb))
            continue
        write_status({"phase": "loading", "dataset": ds_name, "backbone": bb})
        ds = load_dataset(ds_name, bb)
        res["kind"] = ds["kind"]
        res["n_classes"] = ds["n_classes"]
        res["langs"] = ds["langs"]
        res["classes"] = ds["classes"]
        log("=== %s/%s loaded (langs=%s) ===" % (ds_name, bb, ds["langs"]))
        t0 = time.time()
        Z = perlang_z(ds)
        log("%s/%s perlang_z done in %.1fs" % (ds_name, bb, time.time() - t0))
        write_status({"phase": "training", "dataset": ds_name, "backbone": bb})
        D = evaluate(ds, Z)
        B = macro_and_delta(D, ds["kind"], ds["langs"])
        B.update({"per_lang": D["per_lang"], "all_langs": D["all_langs"],
                  "pooled_train_rows": D["pooled_train_rows"],
                  "per_lang_train_rows": {lg: int(len(ds["ytr"][lg])) for lg in ds["langs"]},
                  "test_rows": {lg: int(len(ds["yte"][lg])) for lg in ds["langs"]},
                  "headline": HEADLINE[ds["kind"]], "seconds": round(time.time() - t0, 1)})
        res["backbones"][bb] = B
        doc["datasets"][ds_name] = res
        save_json(doc)
        if not _TAGGED:
            render()
        log("%s/%s done in %.1fs -> %s (macro %s delta %s)"
            % (ds_name, bb, time.time() - t0, B["verdict"], HEADLINE[ds["kind"]],
               _sgn(B["macro_delta"][HEADLINE[ds["kind"]]])))
    doc["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(doc)
    if not _TAGGED:
        render()


def merge_shards():
    """Merge every TMP/alllang_vs_perlang__*.json shard into the repo JSON and render."""
    import glob as _glob
    merged = {"datasets": {}, "meta": {}}
    paths = sorted(_glob.glob(os.path.join(TMP, "alllang_vs_perlang__*.json")))
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as exc:                              # noqa: BLE001
            log("merge: skip %s (%r)" % (p, exc))
            continue
        for name, res in d.get("datasets", {}).items():
            tgt = merged["datasets"].setdefault(
                name, {"name": name, "kind": res.get("kind"), "langs": res.get("langs", []),
                       "backbones": {}})
            tgt["kind"] = res.get("kind", tgt.get("kind"))
            tgt["n_classes"] = res.get("n_classes", tgt.get("n_classes"))
            tgt["langs"] = res.get("langs", tgt["langs"])
            tgt["classes"] = res.get("classes", tgt.get("classes"))
            tgt["backbones"].update(res.get("backbones", {}))
        if not merged["meta"]:
            merged["meta"] = d.get("meta", {})
    merged["meta"]["merged_from"] = [os.path.basename(p) for p in paths]
    merged["meta"]["merged"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(merged)
    render(merged)
    log("merged %d shards -> %s" % (len(paths), OUT_JSON))
    return 0


def launch(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    script = os.path.join(REPO, os.path.basename(__file__))
    assert " " not in sys.executable and " " not in script, "space in launch path"
    command = "%s %s %s" % (sys.executable, script, " ".join(passthrough))
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, command, LOG_PATH, REPO])
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    global _SMOKE, _LIMIT, _TAGGED, OUT_JSON, OUT_MD, OUT_PNG, LOG_PATH, PID_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASET_ORDER))
    ap.add_argument("--backbones", default=",".join(BACKBONE_ORDER))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--tag", default="",
                    help="write this shard's json to TMP (parallel shard; merge with --merge)")
    ap.add_argument("--merge", action="store_true",
                    help="merge TMP shards into the repo json/md/png and exit")
    args = ap.parse_args(argv)
    _SMOKE = args.smoke
    if _SMOKE:
        _LIMIT = 1500

    if args.merge:
        return merge_shards()

    if args.smoke:
        OUT_JSON = os.path.join(TMP, "alllang_vs_perlang_smoke.json")
        OUT_MD = os.path.join(TMP, "alllang_vs_perlang_smoke.md")
        OUT_PNG = os.path.join(TMP, "alllang_vs_perlang_smoke.png")
    elif args.tag:
        _TAGGED = True
        OUT_JSON = os.path.join(TMP, "alllang_vs_perlang__%s.json" % args.tag)
        OUT_MD = os.path.join(TMP, "alllang_vs_perlang__%s.md" % args.tag)
        OUT_PNG = os.path.join(TMP, "alllang_vs_perlang__%s.png" % args.tag)
        LOG_PATH = os.path.join(TMP, "alllang_vs_perlang__%s.log" % args.tag)
        PID_PATH = os.path.join(TMP, "alllang_vs_perlang__%s.pid" % args.tag)

    if args.render:
        render()
        return 0

    if args.launch:
        passthrough = ["--datasets", args.datasets, "--backbones", args.backbones]
        if args.tag:
            passthrough += ["--tag", args.tag]
        launch(passthrough)
        return 0

    dss = [s.strip() for s in args.datasets.split(",") if s.strip()]
    bbs = [s.strip() for s in args.backbones.split(",") if s.strip()]
    pairs = [(d, b) for d in dss for b in bbs]

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    run(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
