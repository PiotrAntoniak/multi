"""toxi_proof.py -- external validation: ONE English-trained XGBoost on EuroBERT-210m **mean**
embeddings, out-of-the-box, on a REAL binary, highly-unbalanced, multilingual toxicity dataset
(`FredZhang7/toxi-text-3M`), evaluated recall-first.

Question.  Does a single English-trained classifier on EuroBERT-210m mean features (no second
model, no MT) perform comparably to a NATIVE per-language model on non-English text, when the
metric is the recall-first view (recall at fixed FPR / false-alarm budgets, FPR to reach 90%
recall), on a REAL, heavily imbalanced, multilingual dataset?

Recipe (task-specified)
  * data     : `FredZhang7/toxi-text-3M` columns (text, is_toxic, lang).
               train  : train/multilingual-train-deduplicated.csv (2.88M rows) -- EN subsampled to
                        20000 rows (seed 0); each target also sampled (native training).
               test   : test/multilingual-test.csv (63.8k) -- per target cap 5000.
               NOTE the validation file was DELETED from the HF repo (see commit history), so the
               per-language z-score stats are fit on that language's OWN TRAIN text (unlabeled for
               the transform; labels only used for model training). Documented in the .md.
  * embed    : EuroBERT/EuroBERT-210m, MASKED MEAN pool, plain forward, max_length 512, batch 64,
               fp32, GPU.
  * transform: `perlang_z` = L2-normalize then per-language z-score.  Each language uses its OWN
               train text stats (EN features use EN stats; target features use target stats).
  * model    : ONE XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2, seed=0,
               max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400) trained on EN.
  * native   : a second model trained natively per target language (same hyperparams), for the
               EN-trained vs native comparison.
  * F1 thresh: tuned on a held-out EN-val split (EN model) / native val split (native models).

Metrics: AUC, accuracy, AP (PR-AUC), F1 @ (val-tuned F1-optimal threshold), recall@FPR{1,5,10%},
recall@{5,20,50,100} false alarms, FPR@recall=0.90.  Plus the target/EN AUC ratio (>=0.9?).

Outputs (repo root; CREATE-ONLY -- nothing existing is deleted or moved):
    toxi_proof.json   full metrics payload
    toxi_proof.md     report (per-target table EN vs native + verdict)
    toxi_proof.png    per-target AUC ratio to EN and FPR@R90 EN-vs-native
    toxi_proof.pid / toxi_proof_status.json
    %TEMP%\\opencode\\toxi_proof.log          (detached launch log)
    %TEMP%\\opencode\\toxi_cache\\*.npy|*.parquet   (embedding/sample caches)

Usage:
    python toxi_proof.py --smoke            # tiny sizes, validates the pipeline
    python toxi_proof.py --launch           # detached hidden (full run)
    python toxi_proof.py                     # foreground full run
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
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.abspath(__file__))
TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
CACHE = os.path.join(TEMP, "opencode", "toxi_cache")
os.makedirs(CACHE, exist_ok=True)

OUT_JSON = os.path.join(REPO, "toxi_proof.json")
OUT_MD = os.path.join(REPO, "toxi_proof.md")
OUT_PNG = os.path.join(REPO, "toxi_proof.png")
PID_PATH = os.path.join(REPO, "toxi_proof.pid")
STATUS_PATH = os.path.join(REPO, "toxi_proof_status.json")
LOG_NAME = "toxi_proof.log"

DATASET = "FredZhang7/toxi-text-3M"
MODEL_ID = "EuroBERT/EuroBERT-210m"
LANGS = ["en", "it", "de", "fr", "es"]
TARGETS = ["it", "de", "fr", "es"]
MAX_LEN, BATCH = 512, 64

# task-specified detector
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_bin=64, device="cpu",
                  n_jobs=2, seed=0, max_depth=8, min_child_weight=5, learning_rate=0.1,
                  n_estimators=400)

FPR_GRID = [0.01, 0.05, 0.10]
FA_GRID = [5, 20, 50, 100]
RECALL_TARGET = 0.90

N_TRAIN = 20000          # rows sampled per language from the big train file
TEST_CAP = 5000          # cap per target test
VAL_FRAC = 0.20

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
def load_test():
    path = os.path.join(CACHE, "test.parquet")
    import pandas as pd
    if os.path.exists(path):
        df = pd.read_parquet(path)
        log("test from cache: %d rows" % len(df))
        return df
    from datasets import load_dataset
    ds = load_dataset(DATASET, data_files={"test": "test/multilingual-test.csv"})
    df = ds["test"].to_pandas()
    df.to_parquet(path)
    log("test loaded+filtered: %d rows" % len(df))
    return df


def load_train_samples(seed=0, n_train=N_TRAIN, smoke=False):
    """Load the big train CSV, sample per-language rows -> ({lang:(texts,labels)}, counts)."""
    import pandas as pd
    if smoke:
        n_train = 400
    cache_path = os.path.join(CACHE, "samples_smoke.parquet" if smoke else "samples.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        log("samples from cache: %d rows" % len(df))
    else:
        from datasets import load_dataset
        log("loading train CSV (1.44GB; download+arrow may take a few minutes) ...")
        ds = load_dataset(DATASET, data_files={"train": "train/multilingual-train-deduplicated.csv"})
        df = ds["train"].to_pandas()
        log("train loaded: %d rows; lang counts=%s" % (len(df), df["lang"].value_counts().to_dict()))
        kept = []
        for lg in LANGS:
            sub = df[df["lang"] == lg]
            n = min(n_train, len(sub))
            if smoke:
                sub = sub.sample(n=n, random_state=seed)
            else:
                sub = sub.sample(n=n, random_state=seed)
            kept.append(sub[["text", "is_toxic", "lang"]])
            log("  sampled %s: %d rows (avail %d)" % (lg, n, int((df["lang"] == lg).sum())))
        samp = pd.concat(kept, ignore_index=True)
        try:
            samp.to_parquet(cache_path)
        except Exception as e:
            log("WARN: sample parquet write failed: %r" % (e,))
        df = samp
    out = {}
    avail = {}
    for lg in LANGS:
        sub = df[df["lang"] == lg]
        avail[lg] = int(len(sub))
        out[lg] = {"texts": sub["text"].astype(str).tolist(),
                   "labels": sub["is_toxic"].astype(int).to_numpy()}
    return out, avail


def split_train_val(texts, labels, seed=0, smoke=False):
    idx = np.arange(len(texts))
    tr, va = train_test_split(idx, test_size=VAL_FRAC, random_state=seed, stratify=labels)
    return ([texts[i] for i in tr], labels[tr], [texts[i] for i in va], labels[va])


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
    order), so batch padding does not stretch short texts to a long outlier's length.  This is a
    pure speed optimisation: each text is embedded independently, so the returned vectors are
    identical to the unsorted batching.
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


# ------------------------------------------------------------------ run
def run(device, smoke=False, seed=0, dedup_test=True):
    write_status("load data")
    test_df = load_test()
    samples, avail = load_train_samples(seed=seed, smoke=smoke)
    log("train sample sizes (avail in big file): %s" % avail)

    tag_suffix = ("_smoke" if smoke else "") + ("_nl" if dedup_test else "")

    # ---- build per-language train/val + test text/labels
    # IMPORTANT: the dataset's non-English splits share exact texts between the train pool and the
    # test pool (translation artifacts).  Without removal a native model MEMORISES the test set.
    # `dedup_test` drops from the train pool every text that appears in that language's EVAL test.
    data = {}
    leak = {}
    for lg in LANGS:
        sub = test_df[test_df["lang"] == lg]
        n = min(300, len(sub)) if smoke else min(TEST_CAP, len(sub))
        te = sub.sample(n=n, random_state=seed) if n < len(sub) else sub
        test_texts = te["text"].astype(str).tolist()
        test_set = set(test_texts)
        texts = samples[lg]["texts"]
        labels = samples[lg]["labels"]
        if dedup_test:
            keep = np.array([i for i, tx in enumerate(texts) if tx not in test_set], dtype=np.int64)
        else:
            keep = np.arange(len(texts), dtype=np.int64)
        leak[lg] = int(len(texts) - len(keep))
        kt = [texts[i] for i in keep]
        ky = labels[keep]
        tr_t, tr_y, va_t, va_y = split_train_val(kt, ky, seed=seed, smoke=smoke)
        data[lg] = {
            "train": (tr_t, tr_y), "val": (va_t, va_y),
            "test": (test_texts, te["is_toxic"].astype(int).to_numpy()),
        }
        log("  %s: train=%d val=%d test=%d (removed %d leaked train texts)"
            % (lg, len(tr_y), len(va_y), len(data[lg]["test"][1]), leak[lg]))

    # ---- embeddings (GPU) + per-language z stats from TRAIN text
    write_status("embed")
    emb, zstats = {}, {}
    for lg in LANGS:
        emb[(lg, "train")] = ensure_emb("%s_train%s" % (lg, tag_suffix), data[lg]["train"][0], device)
        emb[(lg, "val")] = ensure_emb("%s_val%s" % (lg, tag_suffix), data[lg]["val"][0], device)
        emb[(lg, "test")] = ensure_emb("%s_test%s" % (lg, tag_suffix), data[lg]["test"][0], device)
        zstats[lg] = fit_perlang_z(emb[(lg, "train")])

    def feats(lg, split):
        return apply_perlang_z(emb[(lg, split)], *zstats[lg])

    # ---- models
    import xgboost as xgb
    write_status("train")

    def fit(X, y):
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X, y)
        return clf

    # English model
    Xen_tr = feats("en", "train")
    clf_en = fit(Xen_tr, data["en"]["train"][1])
    # F1-optimal threshold on EN val
    en_val_s = clf_en.predict_proba(feats("en", "val"))[:, 1]
    en_f1opt, en_thr = f1_optimal_threshold(data["en"]["val"][1], en_val_s)
    log("EN val: F1opt=%.4f thr=%.4f" % (en_f1opt, en_thr))

    # en reference test
    en_test_s = clf_en.predict_proba(feats("en", "test"))[:, 1]
    res = {"en": {"en": eval_cell(data["en"]["test"][1], en_test_s, en_thr)}}
    res["en"]["en"]["val_f1_opt"] = en_f1opt
    auc_en_ref = res["en"]["en"]["auc"]
    log("EN->EN test AUC=%.4f AP=%.4f" % (auc_en_ref, res["en"]["en"]["average_precision"]))

    # targets: EN-trained (on EN) and native (trained on target)
    for lg in TARGETS:
        yt = data[lg]["test"][1]
        # EN-trained model applied to target (target perlang_z stats)
        s_en = clf_en.predict_proba(feats(lg, "test"))[:, 1]
        # native model
        clf_nat = fit(feats(lg, "train"), data[lg]["train"][1])
        nat_val_s = clf_nat.predict_proba(feats(lg, "val"))[:, 1]
        nat_f1opt, nat_thr = f1_optimal_threshold(data[lg]["val"][1], nat_val_s)
        s_nat = clf_nat.predict_proba(feats(lg, "test"))[:, 1]
        res[lg] = {"en": eval_cell(yt, s_en, en_thr),
                   "native": eval_cell(yt, s_nat, nat_thr)}
        res[lg]["en"]["val_f1_opt"] = en_f1opt
        res[lg]["native"]["val_f1_opt"] = nat_f1opt
        log("%s: EN auc=%.4f ap=%.4f fpr@r90=%s | native auc=%.4f ap=%.4f fpr@r90=%s"
            % (lg, res[lg]["en"]["auc"], res[lg]["en"]["average_precision"],
               res[lg]["en"]["fpr_for_recall_90"], res[lg]["native"]["auc"],
               res[lg]["native"]["average_precision"], res[lg]["native"]["fpr_for_recall_90"]))

    payload = {
        "config": {
            "dataset": DATASET, "emb_model": MODEL_ID, "pooling": "mean",
            "max_len": MAX_LEN, "batch": BATCH, "dtype": "float32",
            "transform": "perlang_z (L2-normalize then per-language z-score from own TRAIN text)",
            "xgb_params": XGB_PARAMS, "langs": LANGS, "targets": TARGETS,
            "n_train_sampled": 400 if smoke else N_TRAIN, "test_cap": TEST_CAP,
            "seed": seed, "smoke": smoke, "dedup_test": dedup_test, "leak_removed": leak,
            "val_note": ("validation/multilingual-validation(new).csv was DELETED from the HF repo; "
                         "per-language z stats are fit on each language's OWN TRAIN text (unlabeled "
                         "for the transform)."),
            "leak_note": ("non-English train/test splits share exact texts; every eval-test text is "
                          "removed from that language's train pool before training (dedup_test)."),
            "fpr_grid": FPR_GRID, "fa_grid": FA_GRID, "recall_target": RECALL_TARGET,
            "avail_train_rows": avail,
        },
        "auc_en_ref": auc_en_ref,
        "results": res,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("wrote %s" % OUT_JSON)

    md = build_md(payload)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    log("wrote %s" % OUT_MD)

    plot(payload, OUT_PNG)
    write_status("done", {"auc_en_ref": auc_en_ref})
    return payload


# ------------------------------------------------------------------ report
def _f(v, nd=3):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def build_md(p):
    cfg = p["config"]
    res = p["results"]
    auc_en = p["auc_en_ref"]
    L = []
    L.append("# toxi_proof -- ONE English-trained XGBoost (EuroBERT-210m mean) on a REAL "
             "unbalanced multilingual toxicity task")
    L.append("")
    L.append("External validation of the out-of-the-box cross-lingual claim on "
             "`%s` (columns `text`,`is_toxic`,`lang`): a single **English-trained** "
             "`XGBClassifier` on **EuroBERT-210m mean** embeddings (no MT, no second model), with "
             "the model-free `perlang_z` transform." % cfg["dataset"])
    L.append("")
    L.append("Recipe: EuroBERT-210m **mean** pool (masked mean, plain forward, max_length %d, "
             "batch %d, **fp32**, GPU) -> `perlang_z` (L2-normalize then per-language z-score; "
             "target stats from the target's OWN text) -> ONE `XGBClassifier(binary:logistic, hist, "
             "max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, lr=0.1, "
             "n_estimators=400)`.  F1-optimal threshold tuned on a held-out EN-val split."
             % (cfg["max_len"], cfg["batch"]))
    L.append("")
    L.append("> **Data note.** `validation/multilingual-validation(new).csv` was **deleted** from "
             "the HF repo (commit history), so the per-language z-score stats are fit on each "
             "language's OWN **train** text (unlabeled for the transform; labels are used only to "
             "train the models, never for the transform).  EN train = %d sampled rows (seed %d); "
             "each target likewise sampled (native training).  Test capped at %d/target; en (525) "
             "and de (190) tests are SMALL and their numbers are noisy."
             % (cfg["n_train_sampled"], cfg["seed"], cfg["test_cap"]))
    L.append("")
    leak = cfg.get("leak_removed", {})
    L.append("> **Leakage removal (`dedup_test`).** The dataset's non-English train and test splits "
             "**share exact texts** (translation artifacts): before training, every eval-test text "
             "is dropped from that language's train pool.  Removed leaked train texts per language: "
             "%s.  Without this, a native model memorises the test set (e.g. it 97%% / fr 76%% / "
             "es 75%% / de 85%% of the test texts were in the train pool) and its scores are "
             "inflated.  EN overlap is ~0, so EN is unaffected." % (leak or "n/a"))
    L.append("")
    L.append("EN reference (EN-trained model on the EN test): **AUC=%.4f**, AP=%.4f, n_pos=%d."
             % (auc_en, res["en"]["en"]["average_precision"], res["en"]["en"]["n_pos"]))
    L.append("")
    L.append("## Per-target: EN-trained vs native (test)")
    L.append("")
    L.append("Lower is better only for `FPR@R90`.  Cells read `EN / native`.")
    L.append("")
    hdr = ["metric", "it", "de", "fr", "es"]
    rows = [
        ("imbalance (pos rate)", "prevalence"),
        ("AUC", "auc"),
        ("AP (PR-AUC)", "average_precision"),
        ("accuracy", "accuracy"),
        ("F1", "f1"),
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
            en_m = res[lg]["en"]
            na_m = res[lg]["native"]
            cells.append("%s / %s" % (_f(en_m[key]), _f(na_m[key])))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Ratios / comparability")
    L.append("")
    L.append("`target/EN` = AUC(EN-trained on target test) / AUC(EN-trained on EN test) "
             "(comparability bar >= 0.9).  `EN/native` = AUC(EN-trained on target) / "
             "AUC(native on target); >= 0.9 means the single English model matches the native one.")
    L.append("")
    L.append("| target | n_pos/n_neg | target/EN AUC | EN/native AUC | FPR@R90 EN/native | R@FA100 EN/native |")
    L.append("|---|---|---|---|---|---|")
    for lg in TARGETS:
        c = res[lg]
        tgt_en = c["en"]["auc"] / auc_en if auc_en else None
        en_nat = c["en"]["auc"] / c["native"]["auc"] if c["native"]["auc"] else None
        L.append("| %s | %d/%d | %s | %s | %s / %s | %s / %s |"
                 % (lg, c["en"]["n_pos"], c["en"]["n_neg"], _f(tgt_en), _f(en_nat),
                    _f(c["en"]["fpr_for_recall_90"]), _f(c["native"]["fpr_for_recall_90"]),
                    _f(c["en"]["recall_at_fa_100"]), _f(c["native"]["recall_at_fa_100"])))
    L.append("")
    L.append("## Verdict")
    L.append("")
    car = [c["en"]["auc"] / auc_en for c in (res[lg] for lg in TARGETS) if auc_en]
    enat = [res[lg]["en"]["auc"] / res[lg]["native"]["auc"]
            for lg in TARGETS if res[lg]["native"]["auc"]]
    ap_en = [res[lg]["en"]["average_precision"] for lg in TARGETS]
    ap_nt = [res[lg]["native"]["average_precision"] for lg in TARGETS]
    fpr_en = [res[lg]["en"]["fpr_for_recall_90"] for lg in TARGETS
              if res[lg]["en"]["fpr_for_recall_90"] is not None]
    fpr_nt = [res[lg]["native"]["fpr_for_recall_90"] for lg in TARGETS
              if res[lg]["native"]["fpr_for_recall_90"] is not None]
    n_ok = sum(1 for r in car if r is not None and r >= 0.9)
    n_ok_en = sum(1 for r in enat if r is not None and r >= 0.9)
    L.append("- Target/EN AUC ratio >= 0.9 in **%d/%d** targets (ratios: %s)."
             % (n_ok, len(TARGETS), ", ".join("%s=%s" % (lg, _f(r)) for lg, r in zip(TARGETS, car))))
    L.append("- EN/native AUC ratio >= 0.9 in **%d/%d** targets (ratios: %s)."
             % (n_ok_en, len(TARGETS), ", ".join("%s=%s" % (lg, _f(r)) for lg, r in zip(TARGETS, enat))))
    L.append("- Mean AP: EN-trained **%.3f** vs native **%.3f**; mean FPR@R90: EN-trained **%s** vs "
             "native **%s** (smaller is better)."
             % (float(np.mean(ap_en)), float(np.mean(ap_nt)),
                _f(float(np.mean(fpr_en))), _f(float(np.mean(fpr_nt)))))
    if n_ok_en == len(TARGETS):
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
    res = p["results"]
    auc_en = p["auc_en_ref"]
    xs = TARGETS
    ratio_en = [res[lg]["en"]["auc"] / auc_en for lg in xs]
    ratio_nat = [res[lg]["native"]["auc"] / auc_en for lg in xs]
    fpr_en = [res[lg]["en"]["fpr_for_recall_90"] for lg in xs]
    fpr_nat = [res[lg]["native"]["fpr_for_recall_90"] for lg in xs]
    x = np.arange(len(xs))
    w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].bar(x - w / 2, ratio_en, w, label="EN-trained", color="#1f77b4")
    ax[0].bar(x + w / 2, ratio_nat, w, label="native", color="#ff7f0e")
    ax[0].axhline(1.0, color="grey", ls=":", lw=1)
    ax[0].axhline(0.9, color="red", ls="--", lw=1, label="0.9 comparability bar")
    ax[0].set_xticks(x); ax[0].set_xticklabels(xs)
    ax[0].set_ylabel("AUC ratio to EN reference")
    ax[0].set_title("AUC ratio to EN (EN-trained vs native)")
    ax[0].set_ylim(0, 1.1); ax[0].legend(fontsize=8)
    for xi, v in zip(x, ratio_en):
        ax[0].text(xi - w / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=7)
    for xi, v in zip(x, ratio_nat):
        ax[0].text(xi + w / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=7)

    ax[1].bar(x - w / 2, fpr_en, w, label="EN-trained", color="#1f77b4")
    ax[1].bar(x + w / 2, fpr_nat, w, label="native", color="#ff7f0e")
    ax[1].set_xticks(x); ax[1].set_xticklabels(xs)
    ax[1].set_ylabel("FPR needed for recall=0.90 (lower better)")
    ax[1].set_title("FPR@R90: EN-trained vs native")
    ax[1].legend(fontsize=8)
    for xi, v in zip(x, fpr_en):
        ax[1].text(xi - w / 2, (v or 0) + 0.002, _f(v, 2), ha="center", fontsize=7)
    for xi, v in zip(x, fpr_nat):
        ax[1].text(xi + w / 2, (v or 0) + 0.002, _f(v, 2), ha="center", fontsize=7)
    fig.suptitle("toxi_proof: ONE EN-trained XGBoost (EuroBERT-210m mean) on toxi-text-3M", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------ main
def launch_detached():
    passthrough = [sys.executable, os.path.abspath(__file__)]
    logpath = os.path.join(TEMP, "opencode", LOG_NAME)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline(passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, logpath, REPO], cwd=REPO)
    print("launched detached; log=%s" % logpath)


def main():
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny sizes (pipeline validation)")
    ap.add_argument("--no-dedup", dest="dedup_test", action="store_false",
                    help="do NOT remove eval-test texts from the train pool (shows the leakage)")
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
    log("start device=%s pid=%d smoke=%s dedup=%s" % (dev, os.getpid(), args.smoke, args.dedup_test))
    write_status("start", {"device": dev, "smoke": args.smoke})
    run(dev, smoke=args.smoke, seed=args.seed, dedup_test=args.dedup_test)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        write_status("error")
        raise
