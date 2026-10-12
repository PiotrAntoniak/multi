"""toxi_proof_clean.py -- re-run of the out-of-the-box cross-lingual proof on the VERSIONED,
LEAKAGE-CORRECTED dataset `data/toxi_clean/` (so the headline numbers are backed by committed
clean data), using the new `validation` splits for threshold tuning.

Difference from `toxi_proof.py`
  * data : reads the committed parquet files `data/toxi_clean/<lang>_{train,validation,test}.parquet`
           (columns `text,is_toxic,lang`) for langs en,it,de,fr,es.  These are already pooled +
           globally deduped + re-split 80/10/10 with **0 cross-split overlap**, so NO in-script
           sampling and NO in-script dedup is needed -- the leakage fix is now in the data.
  * thr  : the F1-optimal decision threshold is tuned on the shipped **validation** split
           (EN model: EN validation; native models: their own language validation).
  * seeds: the detector is trained for seeds 0/1/2 (EN model AND native models) and every metric
           is reported as mean +/- sd, so the comparability verdict is not single-seed.

Everything else is the SAME pipeline as toxi_proof.py:
  EuroBERT-210m **masked-mean** embeddings (fp32, GPU, max_length 512, batch 64, length-sorted
  batching) -> `perlang_z` (L2-normalize then per-language z-score; the TARGET's stats come from the
  TARGET's OWN TRAIN rows) -> ONE English XGBoost(binary:logistic, hist, max_bin=64, cpu, n_jobs=2,
  depth=8, mcw=5, lr=0.1, n=400) on EN train, plus a native per-target model (train on target train).
  Evaluate on each target test; report the recall-first view and the target/EN AUC ratio (>=0.9?).

Outputs (repo root; CREATE/OVERWRITE only -- nothing existing is deleted or moved):
    toxi_proof_clean.json   full metrics payload (per-seed + aggregated mean/sd)
    toxi_proof_clean.md     report (per-target EN vs native table + verdict)
    toxi_proof_clean.png    per-target AUC ratio to EN, and FPR@R90 EN-vs-native
    models_toxi/<lang>_seed<s>.ubj   fresh trained XGBoost models (native UBJSON)
    models_toxi/manifest.json        sizes / hyperparams / seed / val-threshold / val+test metrics
    %TEMP%\\opencode\\toxi_proof_clean.pid / _status.json / .log
    %TEMP%\\opencode\\toxi_clean_cache\\*.npy   (embedding caches)

Usage:
    python toxi_proof_clean.py --smoke       # tiny sizes -> TEMP outputs (pipeline validation)
    python toxi_proof_clean.py --launch      # detached hidden (full run)
    python toxi_proof_clean.py                # foreground full run
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
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

from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             accuracy_score, precision_recall_curve)

REPO = os.path.dirname(os.path.abspath(__file__))
TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
OPCODE = os.path.join(TEMP, "opencode")
CACHE = os.path.join(OPCODE, "toxi_clean_cache")
os.makedirs(CACHE, exist_ok=True)

DATA_DIR = os.path.join(REPO, "data", "toxi_clean")

OUT_JSON = os.path.join(REPO, "toxi_proof_clean.json")
OUT_MD = os.path.join(REPO, "toxi_proof_clean.md")
OUT_PNG = os.path.join(REPO, "toxi_proof_clean.png")
MODELS_DIR = os.path.join(REPO, "models_toxi")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")
PID_PATH = os.path.join(OPCODE, "toxi_proof_clean.pid")
STATUS_PATH = os.path.join(OPCODE, "toxi_proof_clean_status.json")
LOG_NAME = "toxi_proof_clean.log"

MODEL_ID = "EuroBERT/EuroBERT-210m"
LANGS = ["en", "it", "de", "fr", "es"]
TARGETS = ["it", "de", "fr", "es"]
SEEDS = [0, 1, 2]
MAX_LEN, BATCH = 512, 64

# task-specified detector (seed is set per run)
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_bin=64, device="cpu",
                  n_jobs=2, max_depth=8, min_child_weight=5, learning_rate=0.1,
                  n_estimators=400)

FPR_GRID = [0.01, 0.05, 0.10]
FA_GRID = [5, 20, 50, 100]
RECALL_TARGET = 0.90

# The task-specified XGBoost config has no row/column subsampling, so it is SEED-DETERMINISTIC
# (verified: identical predictions for random_state 0/1/2).  To obtain a genuinely non-trivial
# mean+/-sd over seeds (so the verdict is not single-seed) while keeping every specified
# hyperparameter EXACT, each seed trains on a seed-dependent BOOTSTRAP resample (size = n_train,
# with replacement) of that model's train rows.  val/test rows are never resampled.
BOOTSTRAP = True
_LANG_OFF = {"en": 0, "it": 1, "de": 2, "fr": 3, "es": 4}


def _boot_idx(n, seed):
    """n bootstrap indices (with replacement), seeded by `seed`."""
    return np.random.default_rng(seed).integers(0, n, size=n)

_T0 = time.time()


def log(msg):
    line = "[%8.1fs] %s" % (time.time() - _T0, msg)
    print(line, flush=True)


def write_status(phase, extra=None):
    doc = {"phase": phase, "elapsed_s": round(time.time() - _T0, 1), "pid": os.getpid(),
           "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        doc.update(extra)
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
    except Exception as e:
        log("WARN: status write failed: %r" % (e,))


# ------------------------------------------------------------------ transforms
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_perlang_z(X_train):
    Xn = l2norm(X_train)
    mu = Xn.mean(0)
    sd = Xn.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def apply_perlang_z(X, mu, sd):
    return (l2norm(X) - mu) / sd


# ------------------------------------------------------------------ recall metrics
def _cum_curves(y, s):
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")
    ys = np.asarray(y, dtype=np.int64)[order]
    return int(ys.sum()), np.cumsum(ys), np.cumsum(1 - ys)


def recall_at_fp_budget(y, s, budget):
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    if n_pos == 0:
        return None
    j = int(np.searchsorted(cum_fp, budget, side="right")) - 1
    if j < 0:
        return 0.0
    return float(cum_tp[j] / n_pos)


def recall_at_fpr(y, s, fpr):
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    n_neg = len(y) - n_pos
    if n_neg == 0:
        return None
    return recall_at_fp_budget(y, s, int(np.floor(fpr * n_neg)))


def fpr_for_recall(y, s, target=RECALL_TARGET):
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    need = int(np.ceil(target * n_pos - 1e-9))
    j = int(np.searchsorted(cum_tp, need, side="left"))
    if j >= len(cum_tp):
        return None
    return float(cum_fp[j] / n_neg)


def f1_optimal_threshold(y, s):
    """Return (best_f1, threshold)."""
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2.0 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    if k >= len(thr):
        k = len(thr) - 1
    return float(f1[k]), float(thr[k])


def eval_cell(y, s, thr):
    """All metrics for one (labels, scores) cell at a fixed threshold."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    out = {
        "n_pos": n_pos, "n_neg": n_neg,
        "prevalence": float(n_pos / len(y)) if len(y) else None,
        "auc": float(roc_auc_score(y, s)) if n_pos and n_neg else None,
        "average_precision": float(average_precision_score(y, s)) if n_pos and n_neg else None,
    }
    pred = (s >= thr).astype(np.int64)
    out["threshold"] = float(thr)
    out["accuracy"] = float(accuracy_score(y, pred))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    for p in FPR_GRID:
        out["recall_at_fpr_%d" % int(round(p * 100))] = recall_at_fpr(y, s, p)
    for k in FA_GRID:
        out["recall_at_fa_%d" % k] = recall_at_fp_budget(y, s, k)
    out["fpr_for_recall_90"] = fpr_for_recall(y, s, RECALL_TARGET)
    return out


# ------------------------------------------------------------------ data
def load_parquet(lang, split, smoke=False):
    """Read data/toxi_clean/<lang>_<split>.parquet -> (texts, labels).  smoke = tiny head."""
    import pandas as pd
    name = split if split != "validation" else "validation"
    path = os.path.join(DATA_DIR, "%s_%s.parquet" % (lang, name))
    df = pd.read_parquet(path)
    if smoke:
        df = df.sample(n=min(250, len(df)), random_state=1)
    texts = df["text"].astype(str).tolist()
    labels = df["is_toxic"].astype(int).to_numpy()
    return texts, labels


# ------------------------------------------------------------------ embeddings
_EMB = {}


def get_model(device):
    if "tok" not in _EMB:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
        model = model.to(torch.float32)
        _EMB["tok"], _EMB["model"] = tok, model
    return _EMB["tok"], _EMB["model"]


def embed_mean(texts, device):
    """Masked-mean pool over the last hidden state; plain forward, fp32. Returns float32 (n,d).

    Batches are built over texts SORTED by character length (then scatter-restored to the input
    order) so padding does not stretch short texts to a long outlier's length.  Pure speed
    optimisation: each text is embedded independently -> vectors identical to unsorted batching.
    """
    import torch
    tok, model = get_model(device)
    n = len(texts)
    order = np.argsort(np.array([len(t) for t in texts], dtype=np.int64), kind="stable")
    sorted_texts = [texts[i] for i in order]
    acc = []
    with torch.inference_mode():
        for i in range(0, n, BATCH):
            batch = tok(sorted_texts[i:i + BATCH], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
            ids, mask = batch["input_ids"], batch["attention_mask"]
            h = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            m = mask.to(h.device).unsqueeze(-1).to(h.dtype)
            acc.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    e_sorted = np.concatenate(acc, 0).astype(np.float32)
    out = np.empty_like(e_sorted)
    out[order] = e_sorted
    return out


def ensure_emb(tag, texts, device):
    path = os.path.join(CACHE, "emb_%s.npy" % tag)
    if os.path.exists(path):
        return np.load(path)
    log("embedding %s (%d texts) ..." % (tag, len(texts)))
    emb = embed_mean(texts, device)
    np.save(path, emb)
    log("  %s -> %s" % (tag, emb.shape))
    return emb


# ------------------------------------------------------------------ per-seed run
_MODELS_OUT = MODELS_DIR          # redirected to TEMP in smoke mode (never clobber real models)
_MODELS_SUFFIX = ""


def _model_relpath(lang, seed):
    return os.path.relpath(os.path.join(_MODELS_OUT, "%s_seed%d%s.ubj" % (lang, seed, _MODELS_SUFFIX)),
                           REPO).replace("\\", "/")


def save_xgb(clf, lang, seed):
    """Persist a trained XGBoost model (native UBJSON format). Returns (repo-relative path, bytes)."""
    os.makedirs(_MODELS_OUT, exist_ok=True)
    path = os.path.join(_MODELS_OUT, "%s_seed%d%s.ubj" % (lang, seed, _MODELS_SUFFIX))
    clf.save_model(path)
    nbytes = os.path.getsize(path)
    log("  saved %s (%d bytes)" % (path, nbytes))
    return _model_relpath(lang, seed), nbytes


def run_seed(seed, emb, zstats, labels, device):
    """Train EN + native models at this seed; save them; return per-cell metrics + model metadata."""
    import xgboost as xgb

    def feats(lg, split):
        return apply_perlang_z(emb[(lg, split)], *zstats[lg])

    def fit(X, y, seed_key):
        if BOOTSTRAP:
            idx = _boot_idx(len(y), seed_key)
            X, y = X[idx], y[idx]
        clf = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS)
        clf.fit(X, y)
        return clf

    meta = []

    # English model
    clf_en = fit(feats("en", "train"), labels[("en", "train")], 10_000 + seed)
    en_val_s = clf_en.predict_proba(feats("en", "val"))[:, 1]
    en_f1opt, en_thr = f1_optimal_threshold(labels[("en", "val")], en_val_s)
    en_val = eval_cell(labels[("en", "val")], en_val_s, en_thr)
    en_val["val_f1_opt"] = en_f1opt
    en_test_s = clf_en.predict_proba(feats("en", "test"))[:, 1]
    en_ref = eval_cell(labels[("en", "test")], en_test_s, en_thr)
    en_ref["val_f1_opt"] = en_f1opt
    pth, nb = save_xgb(clf_en, "en", seed)
    meta.append({"name": "en_seed%d" % seed, "lang": "en", "seed": seed, "path": pth,
                 "bytes": nb, "n_train": int(len(labels[("en", "train")])),
                 "n_val": int(len(labels[("en", "val")])), "n_test": int(len(labels[("en", "test")])),
                 "val_threshold": en_thr, "val_f1_opt": en_f1opt,
                 "val": en_val, "test": en_ref,
                 "threshold_source": "en_validation.parquet (F1-optimal)"})

    cells = {}
    for lg in TARGETS:
        yt = labels[(lg, "test")]
        s_en = clf_en.predict_proba(feats(lg, "test"))[:, 1]
        clf_nat = fit(feats(lg, "train"), labels[(lg, "train")], 20000 + 100 * _LANG_OFF[lg] + seed)
        nat_val_s = clf_nat.predict_proba(feats(lg, "val"))[:, 1]
        nat_f1opt, nat_thr = f1_optimal_threshold(labels[(lg, "val")], nat_val_s)
        nat_val = eval_cell(labels[(lg, "val")], nat_val_s, nat_thr)
        nat_val["val_f1_opt"] = nat_f1opt
        s_nat = clf_nat.predict_proba(feats(lg, "test"))[:, 1]
        cells[lg] = {
            "en": eval_cell(yt, s_en, en_thr),
            "native": eval_cell(yt, s_nat, nat_thr),
        }
        cells[lg]["en"]["val_f1_opt"] = en_f1opt
        cells[lg]["native"]["val_f1_opt"] = nat_f1opt
        pth, nb = save_xgb(clf_nat, lg, seed)
        meta.append({"name": "%s_seed%d" % (lg, seed), "lang": lg, "seed": seed, "path": pth,
                     "bytes": nb, "n_train": int(len(labels[(lg, "train")])),
                     "n_val": int(len(labels[(lg, "val")])), "n_test": int(len(labels[(lg, "test")])),
                     "val_threshold": nat_thr, "val_f1_opt": nat_f1opt,
                     "val": nat_val, "test": cells[lg]["native"],
                     "threshold_source": "%s_validation.parquet (F1-optimal)" % lg})
        log("  seed=%d %s: EN auc=%.4f ap=%.4f fpr@r90=%s | native auc=%.4f ap=%.4f fpr@r90=%s"
            % (seed, lg, cells[lg]["en"]["auc"], cells[lg]["en"]["average_precision"],
               cells[lg]["en"]["fpr_for_recall_90"], cells[lg]["native"]["auc"],
               cells[lg]["native"]["average_precision"], cells[lg]["native"]["fpr_for_recall_90"]))
    return {"en_ref": en_ref, "cells": cells, "en_thr": en_thr, "en_val_f1_opt": en_f1opt,
            "meta": meta}


# ------------------------------------------------------------------ aggregation
SC_TOKENS = ["prevalence", "auc", "average_precision", "accuracy", "f1", "threshold",
             "val_f1_opt", "recall_at_fpr_1", "recall_at_fpr_5", "recall_at_fpr_10",
             "recall_at_fa_5", "recall_at_fa_20", "recall_at_fa_50", "recall_at_fa_100",
             "fpr_for_recall_90"]


def _agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mean": m, "sd": sd, "n": len(v)}


def aggregate(per_seed):
    """per_seed: list of run_seed dicts -> summary with mean/sd per metric."""
    en_ref = {}
    for k in SC_TOKENS + ["n_pos", "n_neg"]:
        en_ref[k] = _agg([ps["en_ref"].get(k) for ps in per_seed])
    targets = {}
    for lg in TARGETS:
        entry = {"en": {}, "native": {}, "ratios": {}}
        for k in SC_TOKENS + ["n_pos", "n_neg"]:
            entry["en"][k] = _agg([ps["cells"][lg]["en"].get(k) for ps in per_seed])
            entry["native"][k] = _agg([ps["cells"][lg]["native"].get(k) for ps in per_seed])
        # per-seed ratios (robust), then aggregate
        r_tgt_en, r_en_nat = [], []
        for ps in per_seed:
            a_t = ps["cells"][lg]["en"]["auc"]
            a_ref = ps["en_ref"]["auc"]
            a_nat = ps["cells"][lg]["native"]["auc"]
            r_tgt_en.append(a_t / a_ref if a_ref else None)
            r_en_nat.append(a_t / a_nat if a_nat else None)
        entry["ratios"]["target_over_en"] = _agg(r_tgt_en)
        entry["ratios"]["en_over_native"] = _agg(r_en_nat)
        # FPR@R90 ratio (native/en, lower better for EN means ratio>=1 good)
        entry["ratios"]["fpr_r90_en_over_native"] = _agg(
            [ps["cells"][lg]["en"]["fpr_for_recall_90"] / ps["cells"][lg]["native"]["fpr_for_recall_90"]
             if ps["cells"][lg]["native"]["fpr_for_recall_90"] else None for ps in per_seed])
        targets[lg] = entry
    return {"en_ref": en_ref, "targets": targets}


# ------------------------------------------------------------------ run
def run(device, smoke=False):
    global _MODELS_OUT, _MODELS_SUFFIX
    if smoke:                       # never clobber the real models_toxi/ artifacts during smoke
        _MODELS_OUT, _MODELS_SUFFIX = OPCODE, "_smoke"
    write_status("load data")
    emb, zstats, labels = {}, {}, {}
    for lg in LANGS:
        for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
            t, y = load_parquet(lg, split, smoke=smoke)
            labels[(lg, key)] = y
            emb[(lg, key)] = ensure_emb("%s_%s%s" % (lg, key, "_smoke" if smoke else ""), t, device)
        zstats[lg] = fit_perlang_z(emb[(lg, "train")])
        log("%s: n_train=%d n_val=%d n_test=%d" % (lg, len(labels[(lg, "train")]),
                                                   len(labels[(lg, "val")]), len(labels[(lg, "test")])))

    per_seed = []
    for seed in SEEDS:
        write_status("train seed=%d" % seed)
        per_seed.append(run_seed(seed, emb, zstats, labels, device))

    summary = aggregate(per_seed)

    pos_rates = {}
    for lg in LANGS:
        pos_rates[lg] = {}
        for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
            y = labels[(lg, key)]
            pos_rates[lg][split] = {"n": int(len(y)), "n_pos": int(y.sum()),
                                    "pos_rate": float(y.mean())}

    # ---- models manifest (fresh models trained on data/toxi_clean/, val-tuned thresholds)
    models = [m for ps in per_seed for m in ps["meta"]]
    manifest = {
        "data_dir": "data/toxi_clean",
        "dataset": "FredZhang7/toxi-text-3M (leakage-corrected)",
        "emb_model": MODEL_ID, "pooling": "mean", "max_len": MAX_LEN, "batch": BATCH,
        "transform": "perlang_z (L2-normalize then per-language z-score from own TRAIN rows)",
        "xgb_params": XGB_PARAMS, "seeds": SEEDS, "bootstrap_per_seed": BOOTSTRAP,
        "trained_fresh": True,
        "val_split_used_for_threshold": True,
        "threshold_rule": "F1-optimal on the model's own-language VALIDATION split",
        "splits": pos_rates,
        "n_models": len(models),
        "total_model_bytes": int(sum(m["bytes"] for m in models)),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "note": ("Every model here was trained FRESH in this run on data/toxi_clean/ parquet "
                 "splits (train rows), threshold tuned on the validation split, metrics on the "
                 "test split.  No previously persisted/leaked model was loaded or reused.  The "
                 "EN model's cross-lingual (out-of-the-box) per-target test metrics are in "
                 "toxi_proof_clean.json/md."),
    }
    if not smoke:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        log("wrote %s (%d models, %.2f MiB)"
            % (MANIFEST_PATH, len(models), manifest["total_model_bytes"] / 1024.0 / 1024.0))

    payload = {
        "config": {
            "data_dir": "data/toxi_clean", "dataset": "FredZhang7/toxi-text-3M (leakage-corrected)",
            "emb_model": MODEL_ID, "pooling": "mean",
            "max_len": MAX_LEN, "batch": BATCH, "dtype": "float32", "device": device,
            "transform": "perlang_z (L2-normalize then per-language z-score from own TRAIN text)",
            "xgb_params": XGB_PARAMS, "langs": LANGS, "targets": TARGETS, "seeds": SEEDS,
            "smoke": smoke, "fpr_grid": FPR_GRID, "fa_grid": FA_GRID, "recall_target": RECALL_TARGET,
            "bootstrap_per_seed": BOOTSTRAP,
            "models_dir": "models_toxi", "manifest": "models_toxi/manifest.json",
            "models_saved": None if smoke else len(models),
            "note": ("leakage fix is in the committed data: pooled + globally deduped + 80/10/10 "
                     "re-split with 0 cross-split overlap; thresholds tuned on the validation split; "
                     "mean+/-sd over seeds 0/1/2; models retrained fresh and saved to models_toxi/."),
            "seed_note": ("the task-specified XGBoost config is seed-DETERMINISTIC (no row/col "
                          "subsampling); to obtain a non-trivial mean+/-sd, each seed trains on a "
                          "seed-dependent bootstrap resample (size=n_train, with replacement) of its "
                          "train rows; val/test rows and every hyperparameter are unchanged."),
        },
        "per_split_positive_rates": pos_rates,
        "per_seed": per_seed,
        "summary": summary,
        "models": models,
    }
    json_path = OUT_JSON
    md_path = OUT_MD
    png_path = OUT_PNG
    if smoke:
        json_path = os.path.join(OPCODE, "toxi_proof_clean_smoke.json")
        md_path = os.path.join(OPCODE, "toxi_proof_clean_smoke.md")
        png_path = os.path.join(OPCODE, "toxi_proof_clean_smoke.png")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("wrote %s" % json_path)
    md = build_md(payload)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    log("wrote %s" % md_path)
    plot(payload, png_path)
    write_status("done", {"en_ref_auc_mean": summary["en_ref"]["auc"]["mean"]})
    return payload


# ------------------------------------------------------------------ report
def _f(v, nd=3):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def _ms(cell, key, nd=3):
    """Format 'mean+/-sd'."""
    d = cell.get(key)
    if not d or d.get("mean") is None:
        return "n/a"
    return "%s+/-%s" % (_f(d["mean"], nd), _f(d["sd"], nd))


def build_md(p):
    cfg = p["config"]
    summ = p["summary"]
    pr = p["per_split_positive_rates"]
    en = summ["en_ref"]
    L = []
    L.append("# toxi_proof_clean -- ONE English-trained XGBoost (EuroBERT-210m mean) on the "
             "leakage-corrected toxi-text-3M (`data/toxi_clean/`)")
    L.append("")
    L.append("Out-of-the-box cross-lingual validation on the **versioned, leakage-corrected** "
             "dataset `data/toxi_clean/` (pooled + globally deduped + 80/10/10 re-split, "
             "**0 cross-split overlap**).  A single **English-trained** `XGBClassifier` on "
             "**EuroBERT-210m masked-mean** embeddings (no MT, no second model) vs a **native** "
             "per-language model, evaluated recall-first.  All metrics are **mean+/-sd over seeds "
             "%s**." % cfg["seeds"])
    L.append("")
    L.append("Recipe: EuroBERT-210m **mean** pool (masked mean, plain forward, max_length %d, "
             "batch %d, **fp32**, GPU) -> `perlang_z` (L2-normalize then per-language z-score; "
             "target stats from the target's OWN train rows) -> `XGBClassifier(binary:logistic, "
             "hist, max_bin=64, device=cpu, n_jobs=2, max_depth=8, min_child_weight=5, lr=0.1, "
             "n_estimators=400)`.  F1-optimal threshold tuned on the **validation** split "
             "(EN val for the EN model; target val for the native model)."
             % (cfg["max_len"], cfg["batch"]))
    L.append("")
    L.append("> **Data.**  `data/toxi_clean/<lang>_{train,validation,test}.parquet`; the leakage fix "
             "is baked into the committed data, so this run does NO in-script sampling/dedup.  "
             "Test sizes are the full per-language test splits (en/it/de/fr/es = %s)."
             % ", ".join("%s:%d" % (lg, pr[lg]["test"]["n"]) for lg in LANGS))
    L.append("")
    L.append("> **Seed note.**  The task-specified XGBoost config has no row/column subsampling and "
             "is therefore seed-DETERMINISTIC (verified: identical predictions for random_state "
             "0/1/2).  To obtain a genuinely non-trivial mean+/-sd (so the verdict is not single-seed) "
             "each seed trains on a seed-dependent **bootstrap resample** (size=n_train, with "
             "replacement) of that model's train rows.  Validation/test rows and every hyperparameter "
             "are unchanged; thresholds are still tuned on the fixed validation split.")
    L.append("")
    n_models = cfg.get("models_saved")
    if n_models:
        L.append("> **Trained models (fresh).**  Every EN and native model was **trained fresh in "
                 "this run** on the `data/toxi_clean/` train parquet (threshold tuned on the "
                 "validation parquet) and **persisted** with XGBoost `save_model` (`ubj`) under "
                 "`models_toxi/` (%d models, `models_toxi/manifest.json` records lang/split sizes, "
                 "hyperparameters, seed, val-tuned threshold, and val+test metrics).  No previously "
                 "saved/leaked model was loaded or reused." % n_models)
        L.append("")
    L.append("## Per-split positive rates (imbalance)")
    L.append("")
    L.append("| lang | train n (pos%) | val n (pos%) | test n (pos%) |")
    L.append("|---|---|---|---|")
    for lg in LANGS:
        r = pr[lg]
        L.append("| %s | %d (%.3f) | %d (%.3f) | %d (%.3f) |"
                 % (lg, r["train"]["n"], r["train"]["pos_rate"],
                    r["validation"]["n"], r["validation"]["pos_rate"],
                    r["test"]["n"], r["test"]["pos_rate"]))
    L.append("")
    L.append("EN reference (EN-trained on EN test, mean over seeds): **AUC=%s**, AP=%s, "
             "n_pos/n_neg=%d/%d."
             % (_ms(en, "auc"), _ms(en, "average_precision"),
                int(en["n_pos"]["mean"] or 0), int(en["n_neg"]["mean"] or 0)))
    L.append("")
    L.append("## Per-target: EN-trained vs native (test, mean over seeds)")
    L.append("")
    L.append("Cells read `EN / native` as `mean+/-sd`.  Lower is better only for `FPR@R90`.")
    L.append("")
    hdr = ["metric", "it", "de", "fr", "es"]
    rows = [
        ("imbalance (pos rate)", "prevalence"),
        ("AUC", "auc"),
        ("AP (PR-AUC)", "average_precision"),
        ("accuracy", "accuracy"),
        ("F1 @ val-thr", "f1"),
        ("R@FPR1%", "recall_at_fpr_1"),
        ("R@FPR5%", "recall_at_fpr_5"),
        ("R@FPR10%", "recall_at_fpr_10"),
        ("R@FA5", "recall_at_fa_5"),
        ("R@FA20", "recall_at_fa_20"),
        ("R@FA50", "recall_at_fa_50"),
        ("R@FA100", "recall_at_fa_100"),
        ("FPR@R90", "fpr_for_recall_90"),
    ]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for name, key in rows:
        cells = [name]
        for lg in TARGETS:
            e = summ["targets"][lg]["en"]
            n = summ["targets"][lg]["native"]
            cells.append("%s / %s" % (_ms(e, key), _ms(n, key)))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Ratios / comparability (mean over seeds)")
    L.append("")
    L.append("`target/EN` = AUC(EN-trained on target test) / AUC(EN-trained on EN test); bar >= 0.9. "
             "`EN/native` = AUC(EN-trained on target) / AUC(native on target); >= 0.9 means the "
             "single English model matches the native one.")
    L.append("")
    L.append("| target | test n_pos/n_neg | target/EN AUC | EN/native AUC | FPR@R90 EN/native | "
             "R@FA100 EN/native |")
    L.append("|---|---|---|---|---|---|")
    for lg in TARGETS:
        c = summ["targets"][lg]
        L.append("| %s | %d/%d | %s | %s | %s / %s | %s / %s |"
                 % (lg, int(c["en"]["n_pos"]["mean"] or 0), int(c["en"]["n_neg"]["mean"] or 0),
                    _ms(c["ratios"], "target_over_en", 3),
                    _ms(c["ratios"], "en_over_native", 3),
                    _ms(c["en"], "fpr_for_recall_90"), _ms(c["native"], "fpr_for_recall_90"),
                    _ms(c["en"], "recall_at_fa_100"), _ms(c["native"], "recall_at_fa_100")))
    L.append("")
    L.append("## Verdict")
    L.append("")
    ratios = {lg: summ["targets"][lg]["ratios"]["target_over_en"]["mean"] for lg in TARGETS}
    enat = {lg: summ["targets"][lg]["ratios"]["en_over_native"]["mean"] for lg in TARGETS}
    n_ok = sum(1 for lg in TARGETS if ratios[lg] is not None and ratios[lg] >= 0.9)
    n_ok_en = sum(1 for lg in TARGETS if enat[lg] is not None and enat[lg] >= 0.9)
    ap_en = [summ["targets"][lg]["en"]["average_precision"]["mean"] for lg in TARGETS]
    ap_nt = [summ["targets"][lg]["native"]["average_precision"]["mean"] for lg in TARGETS]
    fpr_en = [summ["targets"][lg]["en"]["fpr_for_recall_90"]["mean"] for lg in TARGETS]
    fpr_nt = [summ["targets"][lg]["native"]["fpr_for_recall_90"]["mean"] for lg in TARGETS]
    L.append("- Target/EN AUC ratio >= 0.9 in **%d/%d** targets (ratios: %s)."
             % (n_ok, len(TARGETS), ", ".join("%s=%s" % (lg, _f(ratios[lg])) for lg in TARGETS)))
    L.append("- EN/native AUC ratio >= 0.9 in **%d/%d** targets (ratios: %s)."
             % (n_ok_en, len(TARGETS), ", ".join("%s=%s" % (lg, _f(enat[lg])) for lg in TARGETS)))
    L.append("- Mean AP: EN-trained **%.3f** vs native **%.3f**; mean FPR@R90: EN-trained **%s** vs "
             "native **%s** (smaller is better)."
             % (float(np.mean(ap_en)), float(np.mean(ap_nt)), _f(float(np.mean(fpr_en))),
                _f(float(np.mean(fpr_nt)))))
    if n_ok == len(TARGETS) and n_ok_en == len(TARGETS):
        L.append("- **Verdict: comparable** -- the single English-trained model matches the native "
                 "models on this real, unbalanced multilingual task (AUC and the recall-first view).")
    elif n_ok_en >= len(TARGETS) - 1:
        L.append("- **Verdict: largely comparable** -- the single English-trained model is within "
                 "reach of native on almost all targets (see the outlier).")
    else:
        L.append("- **Verdict: NOT comparable** -- the English-trained model is materially below "
                 "native on the recall-first view for multiple targets.")
    L.append("")
    return "\n".join(L)


def plot(p, out_png):
    summ = p["summary"]
    xs = TARGETS
    ratio_en = [summ["targets"][lg]["ratios"]["target_over_en"]["mean"] for lg in xs]
    ratio_en_sd = [summ["targets"][lg]["ratios"]["target_over_en"]["sd"] for lg in xs]
    ratio_nat = [summ["targets"][lg]["ratios"]["en_over_native"]["mean"] for lg in xs]
    ratio_nat_sd = [summ["targets"][lg]["ratios"]["en_over_native"]["sd"] for lg in xs]
    fpr_en = [summ["targets"][lg]["en"]["fpr_for_recall_90"]["mean"] for lg in xs]
    fpr_nat = [summ["targets"][lg]["native"]["fpr_for_recall_90"]["mean"] for lg in xs]
    x = np.arange(len(xs))
    w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].bar(x - w / 2, ratio_en, w, yerr=ratio_en_sd, capsize=3, label="target/EN (EN-trained)",
              color="#1f77b4")
    ax[0].bar(x + w / 2, ratio_nat, w, yerr=ratio_nat_sd, capsize=3, label="EN/native",
              color="#ff7f0e")
    ax[0].axhline(1.0, color="grey", ls=":", lw=1)
    ax[0].axhline(0.9, color="red", ls="--", lw=1, label="0.9 comparability bar")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(xs)
    ax[0].set_ylabel("AUC ratio")
    ax[0].set_title("AUC ratio (mean+/-sd over seeds)")
    ax[0].set_ylim(0, 1.15)
    ax[0].legend(fontsize=8)
    for xi, v in zip(x, ratio_en):
        if v is not None:
            ax[0].text(xi - w / 2, v + 0.015, "%.2f" % v, ha="center", fontsize=7)
    for xi, v in zip(x, ratio_nat):
        if v is not None:
            ax[0].text(xi + w / 2, v + 0.015, "%.2f" % v, ha="center", fontsize=7)

    ax[1].bar(x - w / 2, fpr_en, w, label="EN-trained", color="#1f77b4")
    ax[1].bar(x + w / 2, fpr_nat, w, label="native", color="#ff7f0e")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(xs)
    ax[1].set_ylabel("FPR needed for recall=0.90 (lower better)")
    ax[1].set_title("FPR@R90: EN-trained vs native (mean over seeds)")
    ax[1].legend(fontsize=8)
    for xi, v in zip(x, fpr_en):
        ax[1].text(xi - w / 2, (v or 0) + 0.004, _f(v, 2), ha="center", fontsize=7)
    for xi, v in zip(x, fpr_nat):
        ax[1].text(xi + w / 2, (v or 0) + 0.004, _f(v, 2), ha="center", fontsize=7)
    fig.suptitle("toxi_proof_clean: ONE EN-trained XGBoost (EuroBERT-210m mean) on "
                 "leakage-corrected toxi-text-3M", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------ main
def launch_detached():
    passthrough = [sys.executable, os.path.abspath(__file__)]
    logpath = os.path.join(OPCODE, LOG_NAME)
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline(passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, logpath, REPO], cwd=REPO)
    print("launched detached; log=%s" % logpath)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true", help="tiny sizes (pipeline validation)")
    ap.add_argument("--launch", action="store_true", help="relaunch detached hidden")
    args = ap.parse_args()

    if args.launch:
        launch_detached()
        return

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    dev = args.device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("start device=%s pid=%d smoke=%s seeds=%s" % (dev, os.getpid(), args.smoke, SEEDS))
    write_status("start", {"device": dev, "smoke": args.smoke})
    run(dev, smoke=args.smoke)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        write_status("error")
        raise
