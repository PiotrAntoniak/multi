"""proof_mean_alllangs.py -- out-of-the-box cross-lingual proof with MEAN pooling only.

Claim under test
----------------
ONE English-trained XGBoost per dataset, fed **EuroBERT-210m mean-pooled** embeddings with
**model-free feature operations only** (normalization / centering / scaling / threshold
calibration on the SAME embeddings), works **comparably** across languages.  No second model,
no machine translation, no translate-train, no target labels.  The single "mild target sample"
variants (`perlang_*`) additionally use each language's own *train* texts (unlabeled) through
the SAME encoder to compute feature stats -- nothing else.

Datasets (read-only; nothing is ever written back to them)
----------------------------------------------------------
* ``sent`` : FLORES 12-class single-label   -- ``train_flores12.prepare_sentence`` +
             ``load_sent_embeddings('mean', kept, 2009)``.
* ``agg``  : FLORES 11-class                 -- ``prepare_agg`` + ``load_agg_embeddings('mean',562)``.
* ``mel``  : MultiEURLEX 21-dim multi-label  -- ``multieurlex/emb_mean_{en,de,fr}_{train,test}.npy``
             + ``multieurlex/<lang>_{train,test}.jsonl`` (``multi_hot``, 21 dims).
Languages: FLORES {en, it, de, fr}; MultiEURLEX {en, de, fr}.  English is the sole model language;
every target-language test set is scored with that one model.

Feature-config ladder (all model-free, applied to the mean embeddings)
----------------------------------------------------------------------
1. ``raw``             nothing.
2. ``l2``              L2-normalize each vector (row-wise; no stats, no target info).
3. ``global_center``   subtract the mean of the ENGLISH train matrix (single global stats).
4. ``global_z``        English-train mean/std standardize (single global stats).
5. ``perlang_center``  subtract each language's OWN train mean  (mild target sample).
6. ``perlang_z``       L2-normalize then per-language OWN train z-score, both sides
                       (== repo ``l2_thenZ [both]``; mild target sample).

For every config the classifier is also evaluated with a threshold tuned on the **English
validation** split (global model threshold); for ``mel`` additionally with
``quantile_match_enval`` (per-concept target threshold whose predicted-positive rate equals the
EN-val TRUE positive rate -- label-free for the target).

Model (one per dataset per config, English only)
------------------------------------------------
``sent`` / ``agg`` : ``XGBClassifier(objective='multi:softprob', num_class=12/11, ...)``.
``mel``            : 21 x ``XGBClassifier(objective='binary:logistic', ...)`` (OVR).
Params (fixed): tree_method=hist, max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8,
min_child_weight=5, lr=0.1, n_estimators=400.

Outputs (create-only: these four files only)
--------------------------------------------
``proof_mean_alllangs.py`` / ``.json`` / ``.md`` / ``proof_mean_alllangs.png``.
Operational pid/log/status live under ``%TEMP%\\opencode\\proof_mean_alllangs``.

Usage
-----
    python proof_mean_alllangs.py --dataset flores          # fast: sent + agg, foreground
    python proof_mean_alllangs.py --dataset mel --launch    # heavy: detached hidden run
    python proof_mean_alllangs.py --render                  # (re)build md/png from the JSON
    python proof_mean_alllangs.py --smoke                   # tiny plumbing check
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
from sklearn.metrics import f1_score, roc_auc_score

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT_JSON = os.path.join(REPO, "proof_mean_alllangs.json")
OUT_MD = os.path.join(REPO, "proof_mean_alllangs.md")
OUT_PNG = os.path.join(REPO, "proof_mean_alllangs.png")

TMP = os.path.join(os.environ.get("TEMP", REPO), "opencode", "proof_mean_alllangs")
os.makedirs(TMP, exist_ok=True)
LOG_PATH = os.path.join(TMP, "proof_mean_alllangs.log")
PID_PATH = os.path.join(TMP, "proof_mean_alllangs.pid")
STATUS_PATH = os.path.join(TMP, "proof_mean_alllangs_status.json")

GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 2)
SEED = 0
VAL_SIZE_MEL = 1000

# Config ladder, in order (simplest first).
CONFIGS = ["raw", "l2", "global_center", "global_z", "perlang_center", "perlang_z"]
NO_TARGET = {"raw", "l2", "global_center", "global_z"}
TARGET_SAMPLE = {"perlang_center", "perlang_z"}

XGB_COMMON = dict(tree_method="hist", max_bin=64, device="cpu", n_jobs=2, seed=SEED,
                  max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)

_T0 = time.time()
_SMOKE = False


def log(msg):
    print("[%8.1fs] %s" % (time.time() - _T0, msg), flush=True)


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    try:
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass


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


# ---------------------------------------------------------------------------
# Model-free feature transforms (all operate on the SAME mean embeddings)
# ---------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def transform(X_by_lang, stats_idx_by_lang, config, ref="en"):
    """Return {lang: transformed full matrix}.  ``stats_idx_by_lang[lg]`` selects the rows used
    to compute that language's stats (its TRAIN rows).  ``global_*`` uses ``ref`` stats for all;
    ``perlang_*`` uses each language's own.  No target labels are ever touched."""
    out = {}
    if config == "raw":
        for lg in X_by_lang:
            out[lg] = np.ascontiguousarray(X_by_lang[lg], dtype=np.float32)
    elif config == "l2":
        for lg in X_by_lang:
            out[lg] = np.ascontiguousarray(l2norm(X_by_lang[lg]), dtype=np.float32)
    elif config == "global_center":
        mu = X_by_lang[ref][stats_idx_by_lang[ref]].mean(0)
        for lg in X_by_lang:
            out[lg] = np.ascontiguousarray(X_by_lang[lg] - mu, dtype=np.float32)
    elif config == "global_z":
        Xr = X_by_lang[ref][stats_idx_by_lang[ref]].astype(np.float64)
        mu, sd = Xr.mean(0), Xr.std(0)
        sd = np.where(sd < 1e-6, 1e-6, sd)
        for lg in X_by_lang:
            out[lg] = np.ascontiguousarray((X_by_lang[lg] - mu) / sd, dtype=np.float32)
    elif config == "perlang_center":
        for lg in X_by_lang:
            mu = X_by_lang[lg][stats_idx_by_lang[lg]].mean(0)
            out[lg] = np.ascontiguousarray(X_by_lang[lg] - mu, dtype=np.float32)
    elif config == "perlang_z":
        for lg in X_by_lang:
            Ztr = l2norm(X_by_lang[lg][stats_idx_by_lang[lg]].astype(np.float64))
            mu, sd = Ztr.mean(0), Ztr.std(0)
            sd = np.where(sd < 1e-6, 1e-6, sd)
            out[lg] = np.ascontiguousarray((l2norm(X_by_lang[lg].astype(np.float64)) - mu) / sd,
                                           dtype=np.float32)
    else:
        raise ValueError("unknown config %r" % (config,))
    return out


# ---------------------------------------------------------------------------
# Multiclass metrics (sent / agg) -- softprob, OVR thresholding
# ---------------------------------------------------------------------------
def prf_ovr(y, P, t):
    n, C = P.shape
    Y = np.zeros((n, C), dtype=bool)
    Y[np.arange(n), y] = True
    Ppos = P >= t
    tp = int((Ppos & Y).sum())
    fp = int((Ppos & ~Y).sum())
    fn = int((~Ppos & Y).sum())
    micro = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    f1s = []
    for c in range(C):
        pp, yy = Ppos[:, c], Y[:, c]
        a = int((pp & yy).sum()); b = int((pp & ~yy).sum()); cc = int((~pp & yy).sum())
        d = 2 * a + b + cc
        f1s.append((2.0 * a / d) if d > 0 else 0.0)
    return micro, float(np.mean(f1s)), f1s


def best_t_global(yv, Pv, mode="micro"):
    bt, bv = 0.5, -1.0
    for t in GRID:
        mic, mac, _ = prf_ovr(yv, Pv, t)
        v = mic if mode == "micro" else mac
        if v > bv:
            bv, bt = v, float(t)
    return bt


def mc_metrics(y_true, P, y_val, P_val, C):
    pred = P.argmax(1)
    micro = float(f1_score(y_true, pred, average="micro", labels=list(range(C)), zero_division=0))
    macro = float(f1_score(y_true, pred, average="macro", labels=list(range(C)), zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, P, multi_class="ovr", average="macro",
                                  labels=list(range(C))))
    except Exception:
        auc = float("nan")
    t_mic = best_t_global(y_val, P_val, "micro")
    t_mac = best_t_global(y_val, P_val, "macro")
    mic_t, mac_t, _ = prf_ovr(y_true, P, t_mic)
    mic_T, mac_T, _ = prf_ovr(y_true, P, t_mac)
    return {
        "micro_f1": micro, "macro_f1": macro, "auc": auc,
        "t_micro": t_mic, "micro_f1_t": float(mic_t), "macro_f1_t": float(mac_t),
        "t_macro": t_mac, "micro_f1_tmacro": float(mic_T), "macro_f1_tmacro": float(mac_T),
    }


# ---------------------------------------------------------------------------
# Multilabel metrics (mel) -- 21 OVR binary, global + quantile-match thresholds
# ---------------------------------------------------------------------------
def micro_macro(pred, gold):
    pred = np.asarray(pred, dtype=bool); gold = np.asarray(gold, dtype=bool)
    tp = int((pred & gold).sum()); fp = int((pred & ~gold).sum()); fn = int((~pred & gold).sum())
    micro = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    f1s = []
    for c in range(gold.shape[1]):
        p, g = pred[:, c], gold[:, c]
        a = int((p & g).sum()); b = int((p & ~g).sum()); cc = int((~p & g).sum())
        d = 2 * a + b + cc
        f1s.append(0.0 if d == 0 else 2 * a / d)
    return micro, float(np.mean(f1s))


def auc_macro(scores, gold):
    aucs = []
    for c in range(gold.shape[1]):
        g = gold[:, c]
        if g.min() == g.max():
            continue
        aucs.append(roc_auc_score(g, scores[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def best_t_micro_ml(yv, Sv):
    best, bt = -1.0, 0.5
    for t in GRID:
        mic, _ = micro_macro(Sv >= t, yv)
        if mic > best:
            best, bt = mic, float(t)
    return bt


def quantile_match_thresholds(scores, val_true):
    rates = np.asarray(val_true, dtype=np.float64).mean(0)
    thr = np.empty(rates.shape[0], dtype=np.float64)
    for c, r in enumerate(rates):
        if r <= 0:
            thr[c] = np.inf
        elif r >= 1:
            thr[c] = -np.inf
        else:
            thr[c] = np.quantile(scores[:, c], 1.0 - r)
    return thr


def ml_metrics(Y_true, S, Y_val, S_val):
    t = best_t_micro_ml(Y_val, S_val)
    microm, macrom = micro_macro(S >= t, Y_true)
    micro05, macro05 = micro_macro(S >= 0.5, Y_true)
    auc = auc_macro(S, Y_true)
    qthr = quantile_match_thresholds(S, Y_val)
    micro_q, macro_q = micro_macro(S >= qthr, Y_true)
    return {
        "micro_f1": float(microm), "macro_f1": float(macrom), "auc": float(auc),
        "t_global": float(t),
        "micro_f1_05": float(micro05), "macro_f1_05": float(macro05),
        "micro_f1_qm": float(micro_q), "macro_f1_qm": float(macro_q),
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_flores(ds):
    import train_flores12 as T
    prep = T.prepare_sentence()
    if ds == "sent":
        X = T.load_sent_embeddings("mean", prep["kept"], n_source=2009)
        return {"name": "sent", "kind": "mc", "n_classes": 12, "classes": prep["classes"],
                "y": prep["y"].astype(int),
                "train": prep["train_idx"], "val": prep["val_idx"], "test": prep["test_idx"],
                "X": {lg: np.asarray(X[lg], dtype=np.float64) for lg in X},
                "langs": ["en", "it", "de", "fr"], "targets": ["it", "de", "fr"]}
    agp = T.prepare_agg(prep["cur"], prep["classes"])
    X = T.load_agg_embeddings("mean", n_rows=562)
    return {"name": "agg", "kind": "mc", "n_classes": 11, "classes": agp["classes"],
            "y": agp["y"].astype(int),
            "train": agp["train_idx"], "val": agp["val_idx"], "test": agp["test_idx"],
            "X": {lg: np.asarray(X[lg], dtype=np.float64) for lg in X},
            "langs": ["en", "it", "de", "fr"], "targets": ["it", "de", "fr"]}


def _read_mel_any(path, n_expect):
    Y = np.zeros((n_expect, 21), dtype=np.int8)
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            Y[n] = np.asarray(json.loads(line)["multi_hot"], dtype=np.int8)
            n += 1
    return Y[:n]


def load_mel():
    R = os.path.join(REPO, "multieurlex")
    langs = ["en", "de", "fr"]
    Xtr, Xte = {}, {}
    for lg in langs:
        Xtr[lg] = np.load(os.path.join(R, "emb_mean_%s_train.npy" % lg)).astype(np.float64)
        Xte[lg] = np.load(os.path.join(R, "emb_mean_%s_test.npy" % lg)).astype(np.float64)
    n_tr = min(Xtr[lg].shape[0] for lg in langs)
    n_te = min(Xte[lg].shape[0] for lg in langs)
    Xall = {lg: np.vstack([Xtr[lg][:n_tr], Xte[lg][:n_te]]) for lg in langs}
    # EN (== shared, parallel) test labels; per-language train labels for the mild-target stats
    # are only used to compute feature statistics (never as supervision).
    Y_train_en = _read_mel_any(os.path.join(R, "en_train.jsonl"), n_tr)
    Y_test = _read_mel_any(os.path.join(R, "en_test.jsonl"), n_te)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_tr)
    val = np.sort(perm[:VAL_SIZE_MEL])
    tr = np.sort(perm[VAL_SIZE_MEL:])
    test = np.arange(n_tr, n_tr + n_te)
    if _SMOKE:
        tr = tr[:2000]; val = val[:500]; test = test[:800]
    return {"name": "mel", "kind": "ml", "n_concepts": 21, "classes": None,
            "Y_train_en": Y_train_en, "Y_test": Y_test, "n_tr": n_tr,
            "train": tr, "val": val, "test": test,
            "X": Xall, "langs": ["en", "de", "fr"], "targets": ["de", "fr"]}


# ---------------------------------------------------------------------------
# One dataset x config evaluation
# ---------------------------------------------------------------------------
def train_score_mc(Xt, ds, config, stats_idx):
    import xgboost as xgb
    C = ds["n_classes"]; y = ds["y"]
    p = dict(XGB_COMMON); p["objective"] = "multi:softprob"; p["num_class"] = C
    if _SMOKE:
        p["n_estimators"] = 50
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xt["en"][ds["train"]], y[ds["train"]])
    P_val = clf.predict_proba(Xt["en"][ds["val"]])
    out = {}
    for lg in ds["langs"]:
        P = clf.predict_proba(Xt[lg][ds["test"]])
        out[lg] = mc_metrics(y[ds["test"]], P, y[ds["val"]], P_val, C)
    return out


def train_score_ml(Xt, ds, config, stats_idx):
    import xgboost as xgb
    Ytr = ds["Y_train_en"]; C = ds["n_concepts"]
    p = dict(XGB_COMMON); p["objective"] = "binary:logistic"
    if _SMOKE:
        p["n_estimators"] = 50
    models = []
    for c in range(C):
        m = xgb.XGBClassifier(**p)
        m.fit(Xt["en"][ds["train"]], Ytr[ds["train"], c].astype(int))
        models.append(m)
    S_val = np.stack([m.predict_proba(Xt["en"][ds["val"]])[:, 1] for m in models], axis=1)
    Yte = ds["Y_test"][ds["test"] - ds["n_tr"]]      # test labels are shared/parallel
    out = {}
    for lg in ds["langs"]:
        S = np.stack([m.predict_proba(Xt[lg][ds["test"]])[:, 1] for m in models], axis=1)
        out[lg] = ml_metrics(Yte, S, Ytr[ds["val"]], S_val)
    return out


def eval_dataset(ds):
    name = ds["name"]
    stats_idx = {lg: ds["train"] for lg in ds["langs"]}   # per-language train rows (same range)
    result = {"name": name, "kind": ds["kind"], "langs": ds["langs"], "targets": ds["targets"],
              "n_classes": ds.get("n_classes", ds.get("n_concepts")),
              "train_n": int(len(ds["train"])), "val_n": int(len(ds["val"])),
              "test_n": int(len(ds["test"])), "configs": {}}
    for config in CONFIGS:
        t0 = time.time()
        Xt = transform(ds["X"], stats_idx, config)
        if ds["kind"] == "mc":
            per_lang = train_score_mc(Xt, ds, config, stats_idx)
        else:
            per_lang = train_score_ml(Xt, ds, config, stats_idx)
        result["configs"][config] = {"per_lang": per_lang, "seconds": round(time.time() - t0, 1)}
        log("%s / %-15s done in %.1fs" % (name, config, time.time() - t0))
        write_status({"phase": "computed", "dataset": name, "config": config})
    return result


# ---------------------------------------------------------------------------
# Summary: spread + target/EN ratio
# ---------------------------------------------------------------------------
PRIMARY = {"mc": ["micro_f1", "macro_f1", "auc"],
           "ml": ["micro_f1", "macro_f1", "auc", "micro_f1_qm", "macro_f1_qm"]}
# Headline metric for the "comparable?" verdict / the plot.  AUC is the threshold-free,
# calibration-robust cross-lingual measure (F1 at a shared threshold also reflects prior shift,
# which threshold calibration cannot fully fix for single-label tasks).  For mel the repo's
# deployable headline is micro-F1 under quantile_match_enval.
HEADLINE = {"sent": "auc", "agg": "auc", "mel": "micro_f1_qm"}


def summarize(result):
    kind = result["kind"]
    langs, targets = result["langs"], result["targets"]
    keys = PRIMARY[kind]
    summary = {"metrics": {}, "headline": HEADLINE[result["name"]]}
    for config, blob in result["configs"].items():
        pl = blob["per_lang"]
        m = {}
        for key in keys:
            vals = {lg: pl[lg][key] for lg in langs if pl[lg].get(key) is not None}
            finite = {lg: v for lg, v in vals.items() if np.isfinite(v)}
            if not finite:
                continue
            en = finite.get("en")
            spread = max(finite.values()) - min(finite.values())
            ratios = {tg: (finite[tg] / en if (en and tg in finite and en != 0) else None)
                      for tg in targets}
            m[key] = {"per_lang": {k: round(v, 4) for k, v in finite.items()},
                      "spread": round(float(spread), 4),
                      "target_over_en": {k: (round(v, 4) if v is not None else None)
                                         for k, v in ratios.items()},
                      "min_ratio": (round(min([v for v in ratios.values() if v is not None]), 4)
                                    if any(v is not None for v in ratios.values()) else None)}
        summary["metrics"][config] = m
    # comparability per metric: every target ratio >= 0.9
    comp_by_metric = {}
    for key in keys:
        comp_by_metric[key] = [c for c in CONFIGS
                               if summary["metrics"].get(c, {}).get(key)
                               and all((v is not None and v >= 0.9)
                                       for v in summary["metrics"][c][key]["target_over_en"].values())]
    summary["comparable_by_metric"] = comp_by_metric
    # simplest config achieving comparability (all target ratios >= 0.9 on the headline metric)
    hl = summary["headline"]
    def comparable(config):
        m = summary["metrics"].get(config, {}).get(hl)
        if not m:
            return False
        rs = m["target_over_en"]
        return all((v is not None and v >= 0.9) for v in rs.values())
    summary["comparable_configs"] = comp_by_metric.get(hl, [])
    summary["simplest_comparable"] = (summary["comparable_configs"][0]
                                      if summary["comparable_configs"] else None)
    summary["simplest_no_target"] = next((c for c in CONFIGS if c in NO_TARGET and comparable(c)),
                                         None)
    summary["simplest_target_sample"] = next((c for c in CONFIGS if c in TARGET_SAMPLE and comparable(c)),
                                             None)
    # same three picks but for the threshold-free AUC and for micro-F1 (informative)
    for key in ("auc", "micro_f1"):
        lst = comp_by_metric.get(key, [])
        summary["simplest_%s" % key] = lst[0] if lst else None
    return summary


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


def render():
    doc = load_json()
    datasets = doc.get("datasets", {})
    order = [d for d in ["sent", "agg", "mel"] if d in datasets]
    if not order:
        log("render: nothing to render")
        return
    # --- markdown ---
    lines = []
    lines.append("# Out-of-the-box cross-lingual proof -- EuroBERT-210m mean pooling + ONE English XGBoost\n")
    lines.append("Model = one English-trained XGBoost per dataset; features are **mean-pooled** "
                 "EuroBERT-210m with **model-free** transforms only (no MT, no second model, no "
                 "target labels). English is the sole model language; every target test set is "
                 "scored by that single model. Metric: FLORES micro/macro-F1 (argmax) + OVR-macro "
                 "AUC; MultiEURLEX micro/macro-F1 at the EN-val global threshold + `quantile_match_enval` "
                 "+ OVR-macro AUC. Ratio = target / English self; **comparable when ratio >= 0.9**.\n")
    for d in order:
        res = datasets[d]
        summ = res["summary"]
        hl = summ["headline"]
        lines.append("## %s (%s, %d classes/concepts)\n"
                     % (d, "single-label multiclass" if res["kind"] == "mc" else "21-dim multi-label",
                        res["n_classes"]))
        lines.append("train_n=%d val_n=%d test_n=%d | headline metric = `%s` | languages = %s\n"
                     % (res["train_n"], res["val_n"], res["test_n"], hl,
                        ", ".join(res["langs"])))
        # per-config headline table
        langs = res["langs"]; targets = res["targets"]
        lines.append("| config | " + " | ".join(langs) + " | spread | " +
                     " | ".join("%s/EN" % t for t in targets) + " | min-ratio |")
        lines.append("|" + "---|" * (len(langs) + 2 + len(targets) + 1))
        for config in CONFIGS:
            m = summ["metrics"].get(config, {}).get(hl)
            if not m:
                continue
            row = ["`%s`" % config]
            row += ["%.3f" % m["per_lang"][lg] if lg in m["per_lang"] else "n/a" for lg in langs]
            row.append("%.3f" % m["spread"])
            row += ["%.3f" % m["target_over_en"][t] if m["target_over_en"].get(t) is not None else "n/a"
                    for t in targets]
            row.append("%.3f" % m["min_ratio"] if m["min_ratio"] is not None else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # full metric matrix (all metrics)
        for key in PRIMARY[res["kind"]]:
            lines.append("**%s**\n" % key)
            lines.append("| config | " + " | ".join(langs) + " | spread | " +
                         " | ".join("%s/EN" % t for t in targets) + " |")
            lines.append("|" + "---|" * (len(langs) + 2 + len(targets)))
            for config in CONFIGS:
                m = summ["metrics"].get(config, {}).get(key)
                if not m:
                    continue
                row = ["`%s`" % config]
                row += ["%.3f" % m["per_lang"][lg] if lg in m["per_lang"] else "n/a" for lg in langs]
                row.append("%.3f" % m["spread"])
                row += ["%.3f" % m["target_over_en"][t] if m["target_over_en"].get(t) is not None else "n/a"
                        for t in targets]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        lines.append("comparable configs (headline `%s` >= 0.9 for every target): %s\n"
                     % (hl, ", ".join("`%s`" % c for c in summ["comparable_configs"]) or "NONE"))
        for key in PRIMARY[res["kind"]]:
            lst = summ.get("comparable_by_metric", {}).get(key, [])
            lines.append("- comparable on `%s`: %s"
                         % (key, ", ".join("`%s`" % c for c in lst) or "NONE"))
        lines.append("")
        lines.append("simplest comparable (headline `%s`): **%s** | simplest NO-target-info: **%s** | "
                     "simplest mild-target-sample: **%s**"
                     % (hl, summ["simplest_comparable"], summ["simplest_no_target"],
                        summ["simplest_target_sample"]))
        lines.append("simplest comparable on AUC: **%s** | on micro-F1: **%s**\n"
                     % (summ.get("simplest_auc"), summ.get("simplest_micro_f1")))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("wrote %s" % OUT_MD)

    # --- plot ---
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.6), squeeze=False)
    colors = {"en": "black", "it": "#1f77b4", "de": "#d62728", "fr": "#2ca02c"}
    for ax, d in zip(axes[0], order):
        res = datasets[d]; summ = res["summary"]; hl = summ["headline"]
        xs = [c for c in CONFIGS if summ["metrics"].get(c, {}).get(hl)]
        for lg in res["langs"]:
            ys = [summ["metrics"][c][hl]["per_lang"].get(lg, np.nan) for c in xs]
            ax.plot(range(len(xs)), ys, marker="o",
                    color=colors.get(lg, None), label=("EN (self)" if lg == "en" else lg),
                    linewidth=(2.6 if lg == "en" else 1.8),
                    linestyle=("--" if lg == "en" else "-"))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=35, ha="right", fontsize=8)
        ax.set_title("%s -- %s" % (d, hl), fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("headline metric")
    fig.suptitle("Mean-pooling + one English XGBoost: target languages converge to English "
                 "across the model-free config ladder", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=130)
    log("wrote %s" % OUT_PNG)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(datasets):
    doc = load_json() if not _SMOKE else {"datasets": {}, "meta": {}}
    doc.setdefault("datasets", {})
    doc["meta"] = {"configs": CONFIGS, "no_target": sorted(NO_TARGET),
                   "target_sample": sorted(TARGET_SAMPLE),
                   "xgb_common": XGB_COMMON, "seed": SEED, "smoke": _SMOKE,
                   "started": doc.get("meta", {}).get("started", time.strftime("%Y-%m-%d %H:%M:%S"))}
    for ds in datasets:
        name = ds["name"]
        if name in doc["datasets"] and not _SMOKE:
            log("%s: already present, skipping" % name)
            continue
        write_status({"phase": "loading", "dataset": name})
        log("=== %s: loaded (%d train / %d val / %d test) ==="
            % (name, len(ds["train"]), len(ds["val"]), len(ds["test"])))
        result = eval_dataset(ds)
        result["summary"] = summarize(result)
        doc["datasets"][name] = result
        save_json(doc)                       # incremental: survive interruption
        render()
    doc["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(doc)
    render()


def launch(passthrough):
    """Detached hidden relaunch through the repo's launch_hidden.vbs (no console window).

    The command is passed to the VBS *unquoted*: python.exe and this script live on
    space-free absolute paths, and embedded quotes would be escaped by the Windows
    command-line encoder in a way wscript/cmd does not undo.
    """
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    script = os.path.join(REPO, os.path.basename(__file__))
    assert " " not in sys.executable and " " not in script, "space in launch path"
    command = "%s %s %s" % (sys.executable, script, " ".join(passthrough))
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, command, LOG_PATH, REPO])
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    global _SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flores",
                    help="flores | sent | agg | mel | all (comma list)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args(argv)
    _SMOKE = args.smoke
    if _SMOKE:
        # never clobber the real outputs with a plumbing check
        global OUT_JSON, OUT_MD, OUT_PNG
        OUT_JSON = os.path.join(TMP, "proof_mean_alllangs_smoke.json")
        OUT_MD = os.path.join(TMP, "proof_mean_alllangs_smoke.md")
        OUT_PNG = os.path.join(TMP, "proof_mean_alllangs_smoke.png")

    if args.render:
        render()
        return

    # Detached hidden relaunch: pass everything except --launch through.
    if args.launch:
        passthrough = []
        if args.dataset:
            passthrough += ["--dataset", args.dataset]
        if args.smoke:
            passthrough += ["--smoke"]
        launch(passthrough)
        return

    sel = [s.strip() for s in args.dataset.split(",") if s.strip()]
    specs = []
    if "flores" in sel or "sent" in sel:
        specs.append(("sent", load_flores, "sent"))
    if "flores" in sel or "agg" in sel:
        specs.append(("agg", load_flores, "agg"))
    if "mel" in sel or "all" in sel:
        specs.append(("mel", None, None))

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    datasets = []
    for name, loader, arg in specs:
        datasets.append(load_mel() if name == "mel" else loader(arg))
    run(datasets)


if __name__ == "__main__":
    main()
