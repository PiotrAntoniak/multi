"""lang_matrix.py -- full NxN train-language -> test-language transfer matrices.

GOAL
----
Train a classifier on EVERY language L and evaluate it on EVERY language L' (held-out TEST
split) for every multilingual dataset, on BOTH EuroBERT backbones (210m, 610m) with MEAN
pooling.  Emit a square N x N matrix per (dataset, backbone): rows = trained-on language,
cols = evaluated-on language, cells = macro-F1 (primary) and AUC.  The diagonal (L' == L)
is the in-language score; the off-diagonal mean is the cross-lingual score.  A summary table
compares the 210m and 610m backbones.

DATASETS (all multilingual).  English-only sst2 / emotion are SKIPPED (noted in the report).
  * flores_sent : 12-class topic; CSV en/it/de/fr.
  * flores_agg  : 11-class topic (URL-aggregated); CSV en/it/de/fr.
  * mel         : MultiEURLEX 21-label multi-label; en/de/fr (multieurlex/*.jsonl).
  * toxic       : binary; en/it/de/fr/es (data/toxi_clean/*.parquet).
  * xnli        : 3-class NLI; premise + " </s> " + hypothesis; en/de/fr/es.
  * pawsx       : binary paraphrase; sentence1 + " </s> " + sentence2; en/de/fr/es.
  * sib200      : single-label topic; en/de/fr/es (Davlan/sib200 TSV).
  * textdetox   : binary toxicity; en/it/de/fr/es (split name == language).

Standard train/test splits are used where defined (validation file -> val).  textdetox has no
predefined split, so 80/10/10 per language (stratified, seed 0) is carved.  TRAIN and VAL are
capped (MAX_TRAIN=20000, MAX_VAL=5000) by a seed-0 permutation subset for tractability; TEST is
kept as the full standard split.  The cap only reduces training cost and is recorded in the
report.

EMBEDDINGS (new root ``lang_matrix_emb/<dataset>/<backbone>/<lang>_{train,val,test}.npy``)
------------------------------------------------------------------------------------------
EuroBERT masked-MEAN pooling over the plain tokenization: bf16 forward -> fp32 storage,
max_length 512, batch 64, length-sorted batching; resumable (one file per lang/split, written
atomically, skipped if present).  The bf16 forward convention is byte-identical to
``backbone_grid._embed_mean`` (same function), so the existing ``emb_bf16/`` caches MAY be
reused as SOURCES for the split-aligned datasets (fflores_sent, flores_agg, mel train/test,
toxic train/val/test); reused arrays are read and re-sliced into the new root -- the sources are
never modified.  Every other array is embedded fresh on the GPU.  Reuse is verified structurally
(2-D, fp32, width == backbone dim, row indices in range) and can be disabled with --no-reuse.

MODEL (XGBoost, fixed params -- no Optuna)
------------------------------------------
  multiclass : objective=multi:softprob, num_class=C (flores_sent/agg, xnli, sib200)
  binary     : objective=binary:logistic (pawsx, textdetox, toxic)
  multilabel : one binary:logistic per label, 21 labels (mel)
  params     : tree_method=hist, max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8,
               min_child_weight=5, learning_rate=0.1, n_estimators=400.
Trained on the train language's train(+val) rows; evaluated on every language's TEST rows.

FEATURE TRANSFORM -- the ``perlang_z`` convention (stated explicitly):
  For each language ell, L2-normalize then fit per-dimension mean/std on ell's OWN TRAIN rows
  and apply (L2 -> (x - mu_ell) / sd_ell) to ell's train/val/test rows.  So when the model is
  trained on language L it sees L's own stats, and when it is evaluated on language L' the model
  sees L''s own stats (each language self-normalizes).  This is the repo-standard ``perlang_z``
  (mean embeddings only; model-free; the target language contributes only its own unlabeled
  train texts through the same encoder to compute its stats).

OUTPUTS (create-only)
---------------------
  lang_matrix.py / lang_matrix.json / lang_matrix.md / lang_matrix_<dataset>.png
  embeddings under lang_matrix_emb/ ; scratch (log/pid/status/cell checkpoints) under
  %TEMP%\\opencode\\lang_matrix*.

USAGE
-----
  python lang_matrix.py --launch                 # detached hidden full run (PID+log/status)
  python lang_matrix.py                          # foreground full run
  python lang_matrix.py --render                 # (re)build md/png from an existing JSON
  python lang_matrix.py --smoke                  # tiny plumbing check (reuse caches, small caps)
"""
import os

# 2-thread runtime budget; set BEFORE numpy / xgboost import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

try:  # cp1252 consoles: never let a unicode print crash the run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TEMP = os.environ.get("TEMP", REPO)
OPCODE = os.path.join(TEMP, "opencode")
os.makedirs(OPCODE, exist_ok=True)

SRC_ROOT = os.path.join(REPO, "emb_bf16")                    # bf16-consistent source caches

# --- smoke / limit overrides (read by every process from the environment) ---
_SMOKE = os.environ.get("LM_SMOKE") == "1"
NO_REUSE = os.environ.get("LM_NO_REUSE") == "1"
_LIMIT = int(os.environ.get("LM_LIMIT", "0"))                 # extra train/val/test cap (0=off)
_ENV_EST = int(os.environ.get("LM_EST", "0"))                 # override n_estimators


def _configure_paths():
    """(Re)derive all output paths from _SMOKE.  Smoke is fully isolated under %TEMP%."""
    global EMB_ROOT, CELLS_DIR, LOG_PATH, PID_PATH, STATUS_PATH, OUT_JSON, OUT_MD, PNG_DIR
    base = "lang_matrix_smoke" if _SMOKE else "lang_matrix"
    if _SMOKE:
        sm = os.path.join(OPCODE, base)
        EMB_ROOT = os.path.join(sm, "emb")
        CELLS_DIR = os.path.join(sm, "cells")
        OUT_JSON = os.path.join(sm, "lang_matrix_smoke.json")
        OUT_MD = os.path.join(sm, "lang_matrix_smoke.md")
        PNG_DIR = sm
    else:
        EMB_ROOT = os.path.join(REPO, "lang_matrix_emb")
        CELLS_DIR = os.path.join(OPCODE, "lang_matrix_cells")
        OUT_JSON = os.path.join(REPO, "lang_matrix.json")
        OUT_MD = os.path.join(REPO, "lang_matrix.md")
        PNG_DIR = REPO
    LOG_PATH = os.path.join(OPCODE, base + ".log")
    PID_PATH = os.path.join(OPCODE, base + ".pid")
    STATUS_PATH = os.path.join(OPCODE, base + "_status.json")
    os.makedirs(CELLS_DIR, exist_ok=True)


_configure_paths()

BACKBONES = {
    "210m": {"id": "EuroBERT/EuroBERT-210m", "dim": 768},
    "610m": {"id": "EuroBERT/EuroBERT-610m", "dim": 1152},
}
BACKBONE_ORDER = ["210m", "610m"]

DATASET_ORDER = ["flores_sent", "flores_agg", "mel", "toxic",
                 "xnli", "pawsx", "sib200", "textdetox"]
SKIPPED = ["sst2 (English-only)", "emotion (English-only)"]

SEED = 0
MAX_TRAIN = 20000
MAX_VAL = 5000
EMB_BATCH = 64
EMB_MAXLEN = 512

XGB_FIXED = dict(tree_method="hist", max_bin=64, device="cpu", n_jobs=2, seed=SEED,
                 max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)

LANGS_FLORES = ["en", "it", "de", "fr"]
LANGS_MEL = ["en", "de", "fr"]
LANGS_5 = ["en", "it", "de", "fr", "es"]                       # toxic, textdetox
LANGS_4 = ["en", "de", "fr", "es"]                             # xnli, pawsx, sib200

_T0 = time.time()


def log(msg):
    print("[%8.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
    os.replace(tmp, path)


def write_status(state):
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        _atomic_json(STATUS_PATH, state)
    except Exception:                                        # noqa: BLE001
        pass


def cell_path(dataset, backbone, train_lang):
    return os.path.join(CELLS_DIR, "%s__%s__%s.json" % (dataset, backbone, train_lang))


def cap_idx(n, cap):
    """Seed-0 permutation subset (sorted) of range(n), or the full range when below the cap."""
    if cap and n > cap:
        return np.sort(np.random.RandomState(SEED).permutation(n)[:cap])
    return np.arange(n)


def _ntrain_cap():
    return _LIMIT if _LIMIT else MAX_TRAIN


def _nval_cap():
    return _LIMIT if _LIMIT else MAX_VAL


def _ntest_cap():
    return _LIMIT                                         # smoke/limit only; 0 -> full test


# ===========================================================================
# Embedding (EuroBERT, masked mean, bf16 forward -> fp32 storage)
# ===========================================================================
def _embed_mean(texts, tok, model, device, max_len=EMB_MAXLEN, batch=EMB_BATCH):
    """Masked mean over non-pad tokens of the plain forward (identical to backbone_grid)."""
    import torch
    n = len(texts)
    if n == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    order = np.argsort(np.array([len(t) for t in texts], dtype=np.int64), kind="stable")
    sorted_texts = [texts[i] for i in order]
    acc = []
    with torch.inference_mode():
        for i in range(0, n, batch):
            enc = tok(sorted_texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
            acc.append(pooled.astype(np.float32))
    e_sorted = np.concatenate(acc, 0).astype(np.float32)
    out = np.empty_like(e_sorted)
    out[order] = e_sorted
    return out


def _save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def resolve_source(srckey, bb, lang):
    root = SRC_ROOT
    dsname, sub = srckey.split("|", 1)
    if dsname == "flores_sent":
        return os.path.join(root, "flores_sent", bb, "mean", "emb_%s.npy" % lang)
    if dsname == "flores_agg":
        return os.path.join(root, "flores_agg", bb, "mean", "emb_%s.npy" % lang)
    if dsname == "mel":
        return os.path.join(root, "mel", bb, "emb_mean_%s_%s.npy" % (lang, sub))
    if dsname == "toxic":
        return os.path.join(root, "toxic", bb, "emb_%s_%s.npy" % (lang, sub))
    raise ValueError("unknown source key %r" % srckey)


def _load_model(bb, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    mid = BACKBONES[bb]["id"]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).to(torch.bfloat16).to(device).eval()
    log("loaded %s on %s (hidden=%d)" % (mid, device, model.config.hidden_size))
    return tok, model


def _free_model(model):
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass


def _meta_path(ds):
    return os.path.join(EMB_ROOT, ds, "meta.json")


def _emb_path(ds, bb, lang, split):
    return os.path.join(EMB_ROOT, ds, bb, "%s_%s.npy" % (lang, split))


# ===========================================================================
# Dataset loaders -- return a spec dict:
#   {"name","kind","n_classes","classes","langs",
#    "data": {lang: {split: {"texts":[...]|None, "y":[...], "reuse":(srckey,rows)|None}}}}
# Every spec's y arrays ALREADY reflect the capped row selection.
# ===========================================================================
def load_flores_sent():
    import pandas as pd
    cur = pd.read_csv(os.path.join(REPO, "flores200_en_it_de_fr.csv"))
    classes = sorted(cur["topic"].astype(str).unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[t] for t in cur["topic"].astype(str)], dtype=int)
    dev = np.where(cur["split"].values == "dev")[0]
    dt = np.where(cur["split"].values == "devtest")[0]
    perm = np.random.default_rng(SEED).permutation(dt)
    nvl = len(perm) // 2
    splits = {"train": dev, "val": perm[:nvl], "test": perm[nvl:]}
    data = {}
    for lg in LANGS_FLORES:
        data[lg] = {}
        for sp, idx in splits.items():
            data[lg][sp] = {"texts": None, "y": y[idx].tolist(),
                            "reuse": ("flores_sent|mean", idx)}
    return dict(name="flores_sent", kind="multiclass", n_classes=len(classes),
                classes=classes, langs=list(LANGS_FLORES), data=data)


def load_flores_agg():
    import pandas as pd
    cur = pd.read_csv(os.path.join(REPO, "flores200_en_it_de_fr.csv"))
    agg = pd.read_csv(os.path.join(REPO, "flores_agg.csv"))
    classes = sorted(cur["topic"].astype(str).unique())
    url_split = {}
    for url, grp in cur.groupby("URL"):
        ss = set(grp["split"])
        url_split[url] = ss.pop() if len(ss) == 1 else None
    split = agg["URL"].map(url_split).fillna("devtest").values
    c2i = {c: i for i, c in enumerate(classes)}
    y_full = np.array([c2i[t] for t in agg["topic"].astype(str)], dtype=int)
    excl = classes.index("business and economy")
    tr_raw = np.where(split == "dev")[0]
    dt = np.where(split == "devtest")[0]
    perm = np.random.default_rng(SEED).permutation(dt)
    nvl = len(perm) // 2
    val_raw, te_raw = perm[:nvl], perm[nvl:]

    def drop(idx):
        return idx[y_full[idx] != excl]

    tr, va, te = drop(tr_raw), drop(val_raw), drop(te_raw)
    kept = [i for i in range(len(classes)) if i != excl]
    remap = {old: new for new, old in enumerate(kept)}
    mask = y_full != excl
    y = np.full(len(y_full), -1, dtype=int)
    y[mask] = [remap[int(v)] for v in y_full[mask]]
    agg_classes = [classes[i] for i in kept]
    data = {lg: {} for lg in LANGS_FLORES}
    for sp, idx in (("train", tr), ("val", va), ("test", te)):
        for lg in LANGS_FLORES:
            data[lg][sp] = {"texts": None, "y": y[idx].tolist(),
                            "reuse": ("flores_agg|mean", idx)}
    return dict(name="flores_agg", kind="multiclass", n_classes=len(agg_classes),
                classes=agg_classes, langs=list(LANGS_FLORES), data=data)


def load_mel():
    def rows(name):
        out = []
        with open(os.path.join(REPO, "multieurlex", name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    tr_rows = {lg: rows("%s_train.jsonl" % lg) for lg in LANGS_MEL}
    va_rows = {lg: rows("%s_validation.jsonl" % lg) for lg in LANGS_MEL}
    te_rows = {lg: rows("%s_test.jsonl" % lg) for lg in LANGS_MEL}
    data = {}
    for lg in LANGS_MEL:
        ytr = np.asarray([r["multi_hot"] for r in tr_rows[lg]], dtype=int)
        yva = np.asarray([r["multi_hot"] for r in va_rows[lg]], dtype=int)
        yte = np.asarray([r["multi_hot"] for r in te_rows[lg]], dtype=int)
        tr = cap_idx(len(ytr), _ntrain_cap())
        va = cap_idx(len(yva), _nval_cap())
        te = cap_idx(len(yte), _ntest_cap())
        va_texts = [va_rows[lg][i]["text"] for i in va]
        data[lg] = {
            "train": {"texts": None, "y": ytr[tr].tolist(), "reuse": ("mel|train", tr)},
            "val": {"texts": va_texts, "y": yva[va].tolist(), "reuse": None},
            "test": {"texts": None, "y": yte[te].tolist(), "reuse": ("mel|test", te)},
        }
    return dict(name="mel", kind="multilabel", n_classes=int(ytr.shape[1]),
                classes=[str(i) for i in range(int(ytr.shape[1]))],
                langs=list(LANGS_MEL), data=data)


def load_toxic():
    import pandas as pd
    data = {}
    for lg in LANGS_5:
        data[lg] = {}
        for sp, sub in (("train", "train"), ("val", "validation"), ("test", "test")):
            df = pd.read_parquet(os.path.join(REPO, "data", "toxi_clean",
                                              "%s_%s.parquet" % (lg, sub)))
            y = df["is_toxic"].astype(int).to_numpy()
            cap = _ntrain_cap() if sp == "train" else (_nval_cap() if sp == "val"
                                                       else _ntest_cap())
            sel = cap_idx(len(y), cap)
            data[lg][sp] = {"texts": None, "y": y[sel].tolist(),
                            "reuse": ("toxic|%s" % sub, sel)}
    return dict(name="toxic", kind="binary", n_classes=2,
                classes=["non_toxic", "toxic"], langs=list(LANGS_5), data=data)


def _hf_pair_loader(ds_id, make_text, kind, n_classes, classes, langs):
    from datasets import load_dataset
    data = {}
    for lg in langs:
        data[lg] = {}
        for sp, splitname in (("train", "train"), ("val", "validation"), ("test", "test")):
            d = load_dataset(ds_id, lg, split=splitname)
            cap = _ntrain_cap() if sp == "train" else (_nval_cap() if sp == "val"
                                                       else _ntest_cap())
            sel = cap_idx(len(d), cap)
            dd = d.select(sel.tolist())
            data[lg][sp] = {"texts": make_text(dd), "y": np.asarray(dd["label"], int).tolist(),
                            "reuse": None}
    return dict(name=ds_id.split("/")[-1], kind=kind, n_classes=n_classes, classes=classes,
                langs=list(langs), data=data)


def load_xnli():
    def mk(dd):
        return [p + " </s> " + h for p, h in zip(dd["premise"], dd["hypothesis"])]
    spec = _hf_pair_loader("facebook/xnli", mk, "multiclass", 3,
                           ["entailment", "neutral", "contradiction"], LANGS_4)
    spec["name"] = "xnli"
    return spec


def load_pawsx():
    def mk(dd):
        return [a + " </s> " + b for a, b in zip(dd["sentence1"], dd["sentence2"])]
    spec = _hf_pair_loader("google-research-datasets/paws-x", mk, "binary", 2,
                           ["not_paraphrase", "paraphrase"], LANGS_4)
    spec["name"] = "pawsx"
    return spec


def load_sib200():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    codes = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn"}
    files, cats = {}, set()
    for lg, code in codes.items():
        def rd(f):
            return pd.read_csv(hf_hub_download("Davlan/sib200", "data/%s/%s.tsv" % (code, f),
                                               repo_type="dataset"), sep="\t")
        tr, va, te = rd("train"), rd("dev"), rd("test")
        files[lg] = (tr, va, te)
        cats |= set(tr["category"]) | set(va["category"]) | set(te["category"])
    cats = sorted(cats)
    c2i = {c: i for i, c in enumerate(cats)}
    data = {}
    for lg, (tr, va, te) in files.items():
        data[lg] = {}
        for sp, df in (("train", tr), ("val", va), ("test", te)):
            cap = _ntrain_cap() if sp == "train" else (_nval_cap() if sp == "val"
                                                       else _ntest_cap())
            sel = cap_idx(len(df), cap)
            sub = df.iloc[sel]
            data[lg][sp] = {"texts": sub["text"].astype(str).tolist(),
                            "y": np.array([c2i[c] for c in sub["category"]]).tolist(),
                            "reuse": None}
    return dict(name="sib200", kind="multiclass", n_classes=len(cats), classes=cats,
                langs=list(LANGS_4), data=data)


def load_textdetox():
    from datasets import load_dataset
    d = load_dataset("textdetox/multilingual_toxicity_dataset", "default")
    data = {}
    for lg in LANGS_5:
        s = d[lg]
        y = np.asarray(s["toxic"], int)
        tx = list(s["text"])
        tr, va, te = [], [], []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            rng = np.random.RandomState(SEED)
            rng.shuffle(ci)
            nt = max(1, int(round(0.10 * len(ci))))
            nv = max(1, int(round(0.10 * len(ci))))
            te += list(ci[:nt])
            va += list(ci[nt:nt + nv])
            tr += list(ci[nt + nv:])
        tr, va, te = np.array(tr), np.array(va), np.array(te)
        data[lg] = {}
        for sp, idx in (("train", tr), ("val", va), ("test", te)):
            cap = _ntrain_cap() if sp == "train" else (_nval_cap() if sp == "val"
                                                       else _ntest_cap())
            if cap and len(idx) > cap:
                idx = np.sort(np.random.RandomState(SEED).permutation(idx)[:cap])
            data[lg][sp] = {"texts": [tx[i] for i in idx], "y": y[idx].tolist(), "reuse": None}
    return dict(name="textdetox", kind="binary", n_classes=2,
                classes=["non_toxic", "toxic"], langs=list(LANGS_5), data=data)


LOADERS = {"flores_sent": load_flores_sent, "flores_agg": load_flores_agg, "mel": load_mel,
           "toxic": load_toxic, "xnli": load_xnli, "pawsx": load_pawsx, "sib200": load_sib200,
           "textdetox": load_textdetox}


def write_meta(spec):
    meta = {"name": spec["name"], "kind": spec["kind"], "n_classes": spec["n_classes"],
            "classes": spec["classes"], "langs": spec["langs"],
            "cap_train": _ntrain_cap(), "cap_val": _nval_cap(), "cap_test": _ntest_cap(),
            "seed": SEED,
            "n": {lg: {sp: len(spec["data"][lg][sp]["y"]) for sp in ("train", "val", "test")}
                  for lg in spec["langs"]},
            "y": {lg: {sp: spec["data"][lg][sp]["y"] for sp in ("train", "val", "test")}
                  for lg in spec["langs"]}}
    _atomic_json(_meta_path(spec["name"]), meta)
    return meta


def ensure_array(ds, bb, lang, split, entry, models, device, stats):
    path = _emb_path(ds, bb, lang, split)
    if os.path.exists(path):
        stats["present"] += 1
        return path
    # ---- reuse source (read + slice; never modify the source) ----
    if not NO_REUSE and entry.get("reuse") is not None:
        srckey, rows = entry["reuse"]
        src = resolve_source(srckey, bb, lang)
        if os.path.exists(src):
            A = np.load(src)
            rows = np.asarray(rows)
            if (A.ndim == 2 and A.shape[1] == BACKBONES[bb]["dim"] and len(rows) > 0
                    and int(rows.max()) < A.shape[0]):
                _save_npy(path, np.ascontiguousarray(A[rows], dtype=np.float32))
                stats["reused"] += 1
                log("  reuse   %-42s <- %s [%d rows]" %
                    (os.path.relpath(path, REPO), os.path.relpath(src, REPO), len(rows)))
                return path
            log("  REUSE-SHAPE-MISMATCH %s (A=%s rows<=%d) -> embedding fresh"
                % (os.path.relpath(src, REPO), A.shape,
                   int(rows.max()) if len(rows) else -1))
    # ---- embed fresh (GPU) ----
    texts = entry.get("texts")
    if texts is None:
        raise RuntimeError("no texts and no usable reuse for %s/%s/%s/%s" % (ds, bb, lang, split))
    if bb not in models:
        models[bb] = _load_model(bb, device)
    tok, model = models[bb]
    arr = _embed_mean(texts, tok, model, device)
    _save_npy(path, arr)
    stats["built"] += 1
    log("  embed   %-42s n=%-6d -> %s" %
        (os.path.relpath(path, REPO), len(texts), arr.shape))
    return path


def build_dataset_embeddings(name, backbones, device):
    spec = LOADERS[name]()
    meta = write_meta(spec)
    log("=== %s: kind=%s C=%d langs=%s n=%s" %
        (name, meta["kind"], meta["n_classes"], meta["langs"], meta["n"]))
    models, stats = {}, {"built": 0, "reused": 0, "present": 0, "todo": []}
    for bb in backbones:
        for lg in spec["langs"]:
            for sp in ("train", "val", "test"):
                p = _emb_path(name, bb, lg, sp)
                if not os.path.exists(p):
                    stats["todo"].append("%s/%s/%s_%s" % (name, bb, lg, sp))
                ensure_array(name, bb, lg, sp, spec["data"][lg][sp], models, device, stats)
        for tok, model in models.values():
            _free_model(model)
        models.clear()
    log("=== %s embeddings: built=%d reused=%d present=%d" %
        (name, stats["built"], stats["reused"], stats["present"]))
    return {"built": stats["built"], "reused": stats["reused"],
            "present": stats["present"], "todo": stats["todo"]}


# ===========================================================================
# Feature transform -- perlang_z (L2 + per-language train z-score)
# ===========================================================================
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_z(Xtrain):
    Z = l2norm(Xtrain)
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd < 1e-6, 1e-6, sd)
    return mu, sd


def apply_z(X, mu, sd):
    return np.ascontiguousarray((l2norm(X) - mu) / sd, dtype=np.float32)


# ===========================================================================
# XGBoost fit / score
# ===========================================================================
def _base_params(kind, n_classes):
    p = dict(XGB_FIXED)
    if _ENV_EST and _SMOKE:
        p["n_estimators"] = _ENV_EST
    if kind == "multiclass":
        p.update(objective="multi:softprob", num_class=int(n_classes))
    else:
        p.update(objective="binary:logistic")
    return p


def fit_model(kind, n_classes, X, y):
    import xgboost as xgb
    X = np.ascontiguousarray(X, dtype=np.float32)
    if kind == "multilabel":
        models = {}
        for c in range(n_classes):
            yc = np.asarray(y[:, c], dtype=int)
            if yc.min() == yc.max():
                continue
            m = xgb.XGBClassifier(**_base_params("binary", 2))
            m.fit(X, yc)
            models[c] = m
        return models
    clf = xgb.XGBClassifier(**_base_params(kind, n_classes))
    clf.fit(X, np.asarray(y, dtype=int))
    return clf


def score_model(kind, n_classes, model, X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if kind == "multilabel":
        S = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for c, m in model.items():
            S[:, c] = m.predict_proba(X)[:, 1]
        return S
    P = model.predict_proba(X)
    if kind == "binary":
        return P[:, 1]
    S = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        if 0 <= int(c) < n_classes:
            S[:, int(c)] = P[:, j]
    return S


def eval_metrics(kind, n_classes, y_true, S):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    out = {}
    if kind == "multilabel":
        y_true = np.asarray(y_true, dtype=int)
        pred = (S >= 0.5).astype(int)
        out["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        out["micro_f1"] = float(f1_score(y_true, pred, average="micro", zero_division=0))
        out["accuracy"] = float(accuracy_score(y_true, pred))
        aucs = []
        for c in range(n_classes):
            g = y_true[:, c]
            if g.min() == g.max():
                continue
            aucs.append(roc_auc_score(g, S[:, c]))
        out["auc"] = float(np.mean(aucs)) if aucs else float("nan")
        return out
    y = np.asarray(y_true, dtype=int)
    if kind == "binary":
        pred = (S >= 0.5).astype(int)
        out["accuracy"] = float(accuracy_score(y, pred))
        out["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))
        try:
            out["auc"] = float(roc_auc_score(y, S)) if y.min() != y.max() else float("nan")
        except Exception:                                    # noqa: BLE001
            out["auc"] = float("nan")
        return out
    pred = np.argmax(S, axis=1)
    out["accuracy"] = float(accuracy_score(y, pred))
    out["macro_f1"] = float(f1_score(y, pred, average="macro", labels=list(range(n_classes)),
                                     zero_division=0))
    try:
        out["auc"] = float(roc_auc_score(y, S, multi_class="ovr", average="macro",
                                         labels=list(range(n_classes))))
    except Exception:                                        # noqa: BLE001
        out["auc"] = float("nan")
    return out


# ===========================================================================
# One matrix row: train on L, evaluate on every language
# ===========================================================================
def run_row(spec):
    ds, bb, L = spec
    out_path = cell_path(ds, bb, L)
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                    # noqa: BLE001
            pass
    t0 = time.time()
    try:
        with open(_meta_path(ds), encoding="utf-8") as f:
            meta = json.load(f)
        langs, kind, C = meta["langs"], meta["kind"], int(meta["n_classes"])

        def getx(lang, sp):
            return np.load(_emb_path(ds, bb, lang, sp))

        stats = {lg: fit_z(getx(lg, "train")) for lg in langs}

        def T(lang, sp):
            return apply_z(getx(lang, sp), *stats[lang])

        ytr = np.concatenate([np.asarray(meta["y"][L]["train"], dtype=int),
                              np.asarray(meta["y"][L]["val"], dtype=int)])
        Xtr = np.vstack([T(L, "train"), T(L, "val")])
        model = fit_model(kind, C, Xtr, ytr)
        row = {}
        for E in langs:
            yte = np.asarray(meta["y"][E]["test"], dtype=int)
            S = score_model(kind, C, model, T(E, "test"))
            row[E] = eval_metrics(kind, C, yte, S)
            if kind == "multilabel":
                row[E]["n_test"] = int(len(yte))
            else:
                row[E]["n_test"] = int(len(yte))
        result = {"dataset": ds, "backbone": bb, "train_lang": L, "kind": kind,
                  "n_classes": C, "langs": langs, "row": row,
                  "elapsed_s": round(time.time() - t0, 1)}
        _atomic_json(out_path, result)
        diag = row.get(L, {})
        off = [row[e]["macro_f1"] for e in langs if e != L and np.isfinite(row[e]["macro_f1"])]
        log("row %-22s diag_macro=%.3f off_macro=%s (%.0fs)"
            % ("%s/%s/%s" % (ds, bb, L), diag.get("macro_f1", float("nan")),
               "%.3f" % np.mean(off) if off else "n/a", result["elapsed_s"]))
        return result
    except Exception as exc:                                 # noqa: BLE001
        result = {"dataset": ds, "backbone": bb, "train_lang": L, "state": "failed",
                  "error": repr(exc), "traceback": traceback.format_exc(),
                  "elapsed_s": round(time.time() - t0, 1)}
        log("row FAILED %s/%s/%s: %r" % (ds, bb, L, exc))
        return result


# ===========================================================================
# Assembly: matrices + summary
# ===========================================================================
def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                    # noqa: BLE001
            pass
    return default


def assemble(datasets):
    doc = _load_json(OUT_JSON, {})
    doc.setdefault("meta", {})
    doc.setdefault("datasets", {})
    doc["meta"].update({"backbones": BACKBONE_ORDER, "pooling": "mean",
                        "transform": "perlang_z (per-language train stats)",
                        "max_train": MAX_TRAIN, "max_val": MAX_VAL, "seed": SEED,
                        "xgb": {k: v for k, v in XGB_FIXED.items()},
                        "metric_primary": "macro_f1", "metric_secondary": "auc",
                        "skipped": SKIPPED, "smoke": _SMOKE})
    for ds in datasets:
        meta = _load_json(_meta_path(ds), None)
        if meta is None:
            continue
        langs, kind, C = meta["langs"], meta["kind"], int(meta["n_classes"])
        entry = {"kind": kind, "n_classes": C, "classes": meta["classes"], "langs": langs,
                 "n": meta["n"], "matrices": {}}
        complete = True
        for bb in BACKBONE_ORDER:
            mat_f1 = [[None] * len(langs) for _ in langs]
            mat_auc = [[None] * len(langs) for _ in langs]
            rows_present = True
            for i, L in enumerate(langs):
                res = _load_json(cell_path(ds, bb, L), None)
                if not res or res.get("state") == "failed" or "row" not in res:
                    rows_present = False
                    continue
                for j, E in enumerate(langs):
                    cell = res["row"].get(E, {})
                    mat_f1[i][j] = cell.get("macro_f1")
                    mat_auc[i][j] = cell.get("auc")
            if not rows_present:
                complete = False
                continue
            diag = [mat_f1[i][i] for i in range(len(langs))]
            off = [mat_f1[i][j] for i in range(len(langs)) for j in range(len(langs)) if i != j]
            diag_a = [mat_auc[i][i] for i in range(len(langs))]
            off_a = [mat_auc[i][j] for i in range(len(langs)) for j in range(len(langs))
                     if i != j]

            def _m(v):
                v = [x for x in v if x is not None and np.isfinite(x)]
                return float(np.mean(v)) if v else float("nan")
            entry["matrices"][bb] = {
                "macro_f1": mat_f1, "auc": mat_auc,
                "summary": {"diag_macro_f1": _m(diag), "offdiag_macro_f1": _m(off),
                            "diag_auc": _m(diag_a), "offdiag_auc": _m(off_a)}}
        if entry["matrices"]:
            doc["datasets"][ds] = entry
        log("assembled %s (complete=%s)" % (ds, complete))
    _atomic_json(OUT_JSON, doc)
    return doc


# ===========================================================================
# Markdown + plots
# ===========================================================================
def _f(v, nd=3):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except Exception:                                        # noqa: BLE001
        return "n/a"
    return ("n/a" if not np.isfinite(v) else ("%%.%df" % nd) % v)


def render():
    doc = _load_json(OUT_JSON, {})
    datasets = doc.get("datasets", {})
    order = [d for d in DATASET_ORDER if d in datasets]
    L = ["# Language transfer matrices (train L -> test L') -- EuroBERT mean pooling", ""]
    L.append("Every language is trained and every language is tested (held-out **test** split; "
             "train-on-`L` evaluate-on-`L'`). Features = EuroBERT **mean** pooling + the "
             "**`perlang_z`** transform (per language: L2-normalize, then fit mean/std on that "
             "language's own TRAIN rows and apply to its own rows -- each language "
             "self-normalizes; the trained model sees its own language's stats, the evaluated "
             "language uses its own). XGBoost fixed params (`hist, max_bin=64, cpu, n_jobs=2, "
             "seed=0, depth8, mcw5, lr0.1, n400`). Trained on train(+val); **macro-F1** is the "
             "primary cell, **AUC** the secondary. Rows = trained-on language, cols = "
             "evaluated-on language; **diagonal = in-language**. English-only datasets skipped: "
             + "; ".join(SKIPPED) + ".")
    L.append("")
    L.append("Train/val are capped at %d/%d rows per language (seed-0 permutation subset) for "
             "tractability; test is the full standard split. textdetox has no predefined split -> "
             "80/10/10 per language (stratified, seed 0)." % (MAX_TRAIN, MAX_VAL))
    L.append("")

    # ---- summary table: off-diagonal (cross-lingual) vs diagonal (in-language) ----
    L.append("## Summary -- mean cross-lingual (off-diagonal) vs in-language (diagonal) macro-F1")
    L.append("")
    L.append("| dataset | backbone | in-language macro-F1 (diag) | cross-lingual macro-F1 "
             "(off-diag) | gap (diag-off) | off-diag AUC | N |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for ds in order:
        e = datasets[ds]
        for bb in BACKBONE_ORDER:
            m = e["matrices"].get(bb)
            if not m:
                L.append("| %s | %s | n/a | n/a | n/a | n/a | %d |" % (ds, bb, len(e["langs"])))
                continue
            s = m["summary"]
            gap = (s["diag_macro_f1"] - s["offdiag_macro_f1"]
                   if np.isfinite(s["diag_macro_f1"]) and np.isfinite(s["offdiag_macro_f1"])
                   else float("nan"))
            L.append("| %s | %s | %s | %s | %s | %s | %d |"
                     % (ds, bb, _f(s["diag_macro_f1"]), _f(s["offdiag_macro_f1"]), _f(gap),
                        _f(s["offdiag_auc"]), len(e["langs"])))
    L.append("")

    # ---- per-dataset NxN matrices ----
    for ds in order:
        e = datasets[ds]
        langs = e["langs"]
        L.append("## %s (%s, %d classes) -- languages: %s"
                 % (ds, e["kind"], e["n_classes"], ", ".join(langs)))
        L.append("")
        for metric in ("macro_f1", "auc"):
            for bb in BACKBONE_ORDER:
                m = e["matrices"].get(bb)
                if not m:
                    continue
                M = m[metric]
                L.append("**%s / EuroBERT-%s**" % (metric, bb))
                L.append("")
                L.append("| trained \\ tested | " + " | ".join(langs) + " |")
                L.append("|" + "---|" * (len(langs) + 1))
                for i, Lg in enumerate(langs):
                    cells = []
                    for j in range(len(langs)):
                        v = M[i][j]
                        cells.append(("**%s**" % _f(v)) if i == j else _f(v))
                    L.append("| %s | " % Lg + " | ".join(cells) + " |")
                L.append("")
        L.append("")

    # ---- overall backbone comparison ----
    L.append("## Overall backbone comparison (mean over datasets)")
    L.append("")
    L.append("| backbone | mean in-language (diag) | mean cross-lingual (off-diag) | mean gap |")
    L.append("|---|---:|---:|---:|")
    for bb in BACKBONE_ORDER:
        dvals, ovals = [], []
        for ds in order:
            m = datasets[ds]["matrices"].get(bb)
            if not m:
                continue
            dvals.append(m["summary"]["diag_macro_f1"])
            ovals.append(m["summary"]["offdiag_macro_f1"])
        dvals = [v for v in dvals if np.isfinite(v)]
        ovals = [v for v in ovals if np.isfinite(v)]
        dm = float(np.mean(dvals)) if dvals else float("nan")
        om = float(np.mean(ovals)) if ovals else float("nan")
        L.append("| %s | %s | %s | %s |" % (bb, _f(dm), _f(om), _f(dm - om)))
    L.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("wrote %s" % OUT_MD)

    # ---- plots: 210m vs 610m side by side per dataset ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for ds in order:
            e = datasets[ds]
            langs = e["langs"]
            mats = [(bb, e["matrices"].get(bb)) for bb in BACKBONE_ORDER]
            if not any(m for _, m in mats):
                continue
            fig, axes = plt.subplots(1, 2, figsize=(4.6 * len(langs) + 3, 4.4 + 0.4 * len(langs)),
                                     squeeze=False)
            for ax, (bb, m) in zip(axes[0], mats):
                if not m:
                    ax.axis("off")
                    ax.set_title("EuroBERT-%s: n/a" % bb)
                    continue
                M = np.array([[np.nan if v is None else v for v in r] for r in m["macro_f1"]],
                             dtype=float)
                im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(langs)))
                ax.set_yticks(range(len(langs)))
                ax.set_xticklabels(langs)
                ax.set_yticklabels(langs)
                ax.set_xlabel("tested on")
                ax.set_ylabel("trained on")
                ax.set_title("EuroBERT-%s  (off-diag mean=%.3f, diag mean=%.3f)"
                             % (bb, m["summary"]["offdiag_macro_f1"],
                                m["summary"]["diag_macro_f1"]), fontsize=10)
                for i in range(len(langs)):
                    for j in range(len(langs)):
                        v = M[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, "%.2f" % v, ha="center", va="center",
                                    color="white" if v < 0.55 else "black", fontsize=8,
                                    fontweight="bold" if i == j else "normal")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle("%s: macro-F1 transfer matrix (rows=trained-on, cols=tested-on)" % ds,
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out = os.path.join(PNG_DIR, "lang_matrix_%s.png" % ds)
            fig.savefig(out, dpi=130)
            plt.close(fig)
            log("wrote %s" % out)
    except Exception as exc:                                 # noqa: BLE001
        log("plot skipped: %r" % (exc,))


# ===========================================================================
# Driver
# ===========================================================================
def run_all(args):
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if "all" in datasets:
        datasets = list(DATASET_ORDER)
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    os.environ["LM_SMOKE"] = "1" if _SMOKE else "0"
    os.environ["LM_NO_REUSE"] = "1" if NO_REUSE else "0"
    os.environ["LM_LIMIT"] = str(_LIMIT)
    os.environ["LM_EST"] = str(_ENV_EST)

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    state = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "smoke": _SMOKE, "phase": "start", "datasets": datasets, "backbones": backbones,
             "embeddings": {}, "rows_done": [], "rows_failed": [],
             "counts": {"rows_total": 0, "rows_done": 0, "rows_failed": 0}}
    write_status(state)
    log("pid=%d smoke=%s datasets=%s backbones=%s workers=%d reuse=%s"
        % (os.getpid(), _SMOKE, datasets, backbones, args.workers, not NO_REUSE))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Phase A: embeddings ----
    state["phase"] = "embeddings"
    write_status(state)
    if args.no_embed:
        log("--no-embed: skipping embedding phase")
    else:
        for ds in datasets:
            state["embeddings"][ds] = build_dataset_embeddings(ds, backbones, device)
            write_status(state)

    # ---- Phase B: matrix rows ----
    state["phase"] = "matrix"
    specs = [(ds, bb, L) for ds in datasets for bb in backbones
             for L in _load_json(_meta_path(ds), {}).get("langs", [])]
    state["counts"]["rows_total"] = len(specs)
    write_status(state)
    log("matrix rows to run: %d" % len(specs))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_row, s): s for s in specs}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:                         # noqa: BLE001
                res = {"dataset": s[0], "backbone": s[1], "train_lang": s[2],
                       "state": "failed", "error": repr(exc)}
            key = "%s/%s/%s" % s
            if res.get("state") == "failed":
                state["rows_failed"].append(key)
            else:
                state["rows_done"].append(key)
            state["counts"]["rows_done"] = len(state["rows_done"])
            state["counts"]["rows_failed"] = len(state["rows_failed"])
            write_status(state)

    # ---- Phase C: assemble + render ----
    state["phase"] = "outputs"
    write_status(state)
    assemble(datasets)
    render()
    state["phase"] = "done"
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["elapsed_s"] = round(time.time() - _T0, 1)
    write_status(state)
    log("ALL DONE in %.1fs" % (time.time() - _T0))
    return state


def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, "-u", os.path.abspath(__file__)]
                                  + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, LOG_PATH, REPO], cwd=REPO)
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    global _SMOKE, _LIMIT, _ENV_EST, NO_REUSE
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Full LxL cross-lingual transfer matrices.")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding phase")
    ap.add_argument("--no-reuse", action="store_true", help="never reuse emb_bf16 sources")
    ap.add_argument("--render", action="store_true", help="rebuild md/png from existing JSON")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--backbones", default=",".join(BACKBONE_ORDER))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit-rows", type=int, default=0, help="extra train/val/test cap")
    ap.add_argument("--n-estimators", type=int, default=0, help="override (smoke/debug)")
    args = ap.parse_args(argv)

    if args.launch:
        extra = [a for a in argv if a != "--launch"]
        launch_detached(extra)
        return 0
    if args.render:
        render()
        return 0

    _SMOKE = args.smoke
    NO_REUSE = args.no_reuse
    if _SMOKE:
        _ENV_EST = args.n_estimators or 60
        _LIMIT = args.limit_rows or 120
        args.workers = min(args.workers, 1)
    else:
        _LIMIT = args.limit_rows
        _ENV_EST = args.n_estimators
    _configure_paths()
    run_all(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        raise
