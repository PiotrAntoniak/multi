r"""backbone_grid.py -- single results table of XGBoost performance across
TWO backbones (EuroBERT-210m / EuroBERT-610m) x datasets x 4 methods, each tuned with
10 Optuna TPE trials, MEAN POOLING only.  Also rebuilds the 610m toxic precision/recall
figure `toxi_pr_610.png`.

Grid
----
  backbones : 210m (EuroBERT/EuroBERT-210m, 768-d), 610m (EuroBERT/EuroBERT-610m, 1152-d)
  datasets  : flores_sent (12-class), flores_agg (11-class), mel (MultiEURLEX 21-label),
              sst2 (binary), emotion (6-class), toxic (binary; en/it/de/fr/es)
  methods   : mean (raw), pc (GLOBAL PCA k=2 on the stacked-language train pool, zero the
              leading 2 components), inlp (~30-dir iterated null-space projection of a
              language-identity probe on the stacked train pool), D (language-shift:
              target features + D, D = mean_en_train - mean_target_train)
  model     : XGBoost, 10 Optuna TPE trials (seed 0) per cell, objective = val AUC (macro OVR).
              Trained on English, evaluated on English test ("EN self") and on every non-English
              target test; the cross-lingual number is the MEAN over non-English targets.

Embeddings (masked-mean pooling, forward in BFLOAT16, stored float32)
---------------------------------------------------------------------
EVERY embedding the grid uses is (re)built in bf16 under the NEW root ``emb_bf16/`` so the whole
table is bf16-consistent.  The committed fp32 caches are only read as SOURCES of text and are
never overwritten.  Layout::

  emb_bf16/flores_sent/<210m|610m>/mean/emb_<lang>.npy     (2009-era CSV: current 2006 rows)
  emb_bf16/flores_agg/<bb>/mean/emb_<lang>.npy
  emb_bf16/mel/<bb>/emb_mean_<lang>_{train,test}.npy       (from multieurlex/<lang>_*.jsonl)
  emb_bf16/sst2/<bb>/mean_{train,val,test}.npy             (from sst2.csv)
  emb_bf16/emotion/<bb>/mean_{train,val,test}.npy          (from emotion.csv)
  emb_bf16/toxic/<bb>/emb_<lang>_{train,validation,test}.npy  (from data/toxi_clean/*.parquet)

Flores splits come from `train_flores12.prepare_sentence` / `prepare_agg`; mel from its jsonl
(a 1000-row val slice of train, seed 0); sst2/emotion from their CSVs; toxic from the parquet.

Constraints: mean pooling only; GPU only for embedding; CPU XGB (n_jobs=2); resumable cells;
process pool <= 3 workers; no window; create-only (nothing deleted or moved).

Outputs (repo root): backbone_grid.json / backbone_grid.md / backbone_grid.png / toxi_pr_610.png
Logs / pid / status / per-cell checkpoints: %TEMP%\opencode\.

Usage:
  python backbone_grid.py --launch                 # detached hidden full run
  python backbone_grid.py                          # foreground full run
  python backbone_grid.py --smoke --no-embed       # tiny plumbing check (existing caches)
"""
import os

# 2-thread runtime budget; set before numpy / xgboost import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:                                            # noqa: BLE001
    pass
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TEMP = os.environ.get("TEMP", REPO)
OPCODE = os.path.join(TEMP, "opencode")
os.makedirs(OPCODE, exist_ok=True)
CELLS_DIR = os.path.join(OPCODE, "backbone_grid_cells")
os.makedirs(CELLS_DIR, exist_ok=True)

LOG_PATH = os.path.join(OPCODE, "backbone_grid.log")
PID_PATH = os.path.join(OPCODE, "backbone_grid.pid")
STATUS_PATH = os.path.join(OPCODE, "backbone_grid_status.json")
TOXI_CACHE = os.path.join(OPCODE, "toxi_clean_cache")

OUT_JSON = os.path.join(REPO, "backbone_grid.json")
OUT_MD = os.path.join(REPO, "backbone_grid.md")
OUT_PNG = os.path.join(REPO, "backbone_grid.png")
OUT_TOXI610 = os.path.join(REPO, "toxi_pr_610.png")

# bf16-consistent embedding cache root (create-only).  The committed fp32 caches
# (embeddings/, embeddings_610/, embeddings_agg*/, sst2_emb/, emotion_emb/, multieurlex/,
# multieurlex_610/, *_emb_610/, toxi_emb_*) are NEVER touched; the grid reads ONLY emb_bf16/.
BF16_ROOT = os.path.join(REPO, "emb_bf16")
EMB_DTYPE = "bfloat16"

BACKBONES = {
    "210m": {"id": "EuroBERT/EuroBERT-210m", "dim": 768},
    "610m": {"id": "EuroBERT/EuroBERT-610m", "dim": 1152},
}
BACKBONE_ORDER = ["210m", "610m"]
METHOD_ORDER = ["mean", "pc", "inlp", "D"]
METHOD_LABEL = {
    "mean": "mean (raw)",
    "pc": "pc (global PCA k=2)",
    "inlp": "inlp (~30-dir)",
    "D": "D (language-shift)",
}
DATASETS = ["flores_sent", "flores_agg", "mel", "sst2", "emotion", "toxic"]

N_TRIALS = 10
SEED = 0
MAX_BIN = 64
N_JOBS = 2
EMB_BATCH = 64
EMB_MAXLEN = 512

INLP_MAX_DIRS = 30
INLP_MAX_ITER = 100
INLP_C = 1.0

LANG_FLORES = ["en", "it", "de", "fr"]
LANG_MEL = ["en", "de", "fr"]
LANG_TOXI = ["en", "it", "de", "fr", "es"]

# Optional smoke overrides, read by every (spawned) process from the environment.
_SMOKE = os.environ.get("BG_SMOKE") == "1"
_LIMIT = int(os.environ.get("BG_LIMIT", "0"))            # 0 = no row limit
_ENV_TRIALS = int(os.environ.get("BG_TRIALS", "0"))

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def write_status(state):
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_json(STATUS_PATH, state)


def cell_path(dataset, backbone, method):
    return os.path.join(CELLS_DIR, "%s__%s__%s.json" % (dataset, backbone, method))


# ===========================================================================
# Embedding phase (GPU) -- create-only, resumable
# ===========================================================================
def _embed_mean(texts, tok, model, device, max_len=EMB_MAXLEN, batch=EMB_BATCH):
    """Masked mean over non-pad tokens of the plain forward.

    The backbone runs in bfloat16 (the model is bf16); the pooled last hidden state is
    upcast to float32 and the masked mean + save use float32 (bf16-speed forward, fp32
    stored arrays).
    """
    import torch
    n = len(texts)
    if n == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    order = np.argsort(np.array([len(t) for t in texts], dtype=np.int64), kind="stable")
    sorted_texts = [texts[i] for i in order]
    acc = []
    with torch.inference_mode():
        for i in range(0, n, batch):
            chunk = sorted_texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
            acc.append(pooled.astype(np.float32))
    e_sorted = np.concatenate(acc, 0).astype(np.float32)
    out = np.empty_like(e_sorted)
    out[order] = e_sorted
    return out


def ensure_npy(path, texts, tok, model, device):
    """Create `path` with the masked-mean embeddings of `texts` if it does not exist."""
    if os.path.exists(path):
        return np.load(path)
    log("  embed -> %s (%d texts)" % (os.path.relpath(path, REPO), len(texts)))
    arr = _embed_mean(texts, tok, model, device)
    tmp = path[:-4] + ".tmp.npy"          # keeps the .npy suffix so np.save does not append one
    np.save(tmp, arr.astype(np.float32))
    os.replace(tmp, path)
    return arr


def _read_jsonl_texts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["text"])
    return out


def _load_model(backbone, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    mid = BACKBONES[backbone]["id"]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).to(torch.bfloat16).to(device).eval()
    log("%s on %s (dtype=bf16, hidden=%d)" % (mid, device, model.config.hidden_size))
    return tok, model


def _free_model(model):
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass


def plan_missing(backbone):
    """Return the list of (out_path, texts) still to embed for this backbone.

    EVERY embedding the grid uses is (re)built in bf16 under ``emb_bf16/`` so the whole table
    is bf16-consistent; the committed fp32 caches are only read as SOURCES of text, never
    overwritten.  ``aggregate=True`` (per dataset) means one array covers that dataset.
    """
    import pandas as pd
    todo = []
    bb = backbone

    # ---- flores_sent / flores_agg : embed the CURRENT csv rows (already aligned to the
    #      split indices that train_flores12 derives from the same csv). ----
    sent_csv = pd.read_csv(os.path.join(REPO, "flores200_en_it_de_fr.csv"))
    agg_csv = pd.read_csv(os.path.join(REPO, "flores_agg.csv"))
    for ds, df in (("flores_sent", sent_csv), ("flores_agg", agg_csv)):
        for lg in LANG_FLORES:
            out = os.path.join(BF16_ROOT, ds, bb, "mean", "emb_%s.npy" % lg)
            if not os.path.exists(out):
                todo.append((out, df[lg].astype(str).tolist()))

    # ---- mel ----
    for lg in LANG_MEL:
        for split in ("train", "test"):
            out = os.path.join(BF16_ROOT, "mel", bb, "emb_mean_%s_%s.npy" % (lg, split))
            if not os.path.exists(out):
                src = os.path.join(REPO, "multieurlex", "%s_%s.jsonl" % (lg, split))
                todo.append((out, _read_jsonl_texts(src)))

    # ---- sst2 / emotion ----
    for ds, csv_name, text_col in (("sst2", "sst2.csv", "sentence"),
                                   ("emotion", "emotion.csv", "text")):
        df = pd.read_csv(os.path.join(REPO, csv_name))
        for split, col in (("train", "train"), ("val", "validation"), ("test", "test")):
            out = os.path.join(BF16_ROOT, ds, bb, "mean_%s.npy" % split)
            if not os.path.exists(out):
                todo.append((out, df[df["split"] == col][text_col].astype(str).tolist()))

    # ---- toxic (embedded fresh in bf16 for BOTH backbones) ----
    for lg in LANG_TOXI:
        for split in ("train", "validation", "test"):
            out = os.path.join(BF16_ROOT, "toxic", bb, "emb_%s_%s.npy" % (lg, split))
            if not os.path.exists(out):
                df = pd.read_parquet(os.path.join(REPO, "data", "toxi_clean",
                                                  "%s_%s.parquet" % (lg, split)))
                todo.append((out, df["text"].astype(str).tolist()))
    return todo


def build_embeddings(backbones, device):
    """Ensure every needed embedding cache exists (GPU for the missing ones)."""
    for bb in backbones:
        todo = plan_missing(bb)
        if not todo:
            log("embeddings[%s]: nothing to build" % bb)
            continue
        log("embeddings[%s]: %d array(s) to build" % (bb, len(todo)))
        tok, model = _load_model(bb, device)
        for out, texts in todo:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            ensure_npy(out, texts, tok, model, device)
        _free_model(model)


# ===========================================================================
# Data loading (per process; each cell loads its dataset from disk)
# ===========================================================================
_DCACHE = {}


def _limit_rows(a):
    return a[:_LIMIT] if (_LIMIT and len(a) > _LIMIT) else a


def _load_flores(ds, backbone):
    import train_flores12 as T
    sent = T.prepare_sentence()
    classes = sent["classes"]
    if ds == "flores_sent":
        base = os.path.join(BF16_ROOT, "flores_sent", backbone, "mean")
        X = {lg: np.load(os.path.join(base, "emb_%s.npy" % lg)) for lg in LANG_FLORES}
        n_rows = len(sent["cur"])
        tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
        y = sent["y"]
        langs, n_classes, kind = LANG_FLORES, 12, "multiclass"
    else:
        agg = T.prepare_agg(sent["cur"], classes)
        base = os.path.join(BF16_ROOT, "flores_agg", backbone, "mean")
        X = {lg: np.load(os.path.join(base, "emb_%s.npy" % lg)) for lg in LANG_FLORES}
        n_rows = len(agg["agg"])
        tr, va, te = agg["train_idx"], agg["val_idx"], agg["test_idx"]
        y = agg["y"]
        langs, n_classes, kind = LANG_FLORES, agg["n_classes"], "multiclass"
    for lg in langs:
        if X[lg].shape[0] != n_rows:
            raise ValueError("%s/%s: %s has %d rows, expected %d"
                             % (ds, backbone, lg, X[lg].shape[0], n_rows))
    tr, va, te = _limit_rows(tr), _limit_rows(va), _limit_rows(te)
    Xtr = {lg: np.asarray(X[lg][tr]) for lg in langs}
    Xva = {lg: np.asarray(X[lg][va]) for lg in langs}
    Xte = {lg: np.asarray(X[lg][te]) for lg in langs}
    ytr = {lg: y[tr] for lg in langs}
    yva = {lg: y[va] for lg in langs}
    yte = {lg: y[te] for lg in langs}
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=ytr, yva=yva, yte=yte,
                langs=langs, targets=[l for l in langs if l != "en"],
                kind=kind, n_classes=n_classes)


def _load_mel(backbone):
    root = os.path.join(BF16_ROOT, "mel", backbone)
    Xtr, Xte = {}, {}
    for lg in LANG_MEL:
        Xtr[lg] = np.load(os.path.join(root, "emb_mean_%s_train.npy" % lg))
        Xte[lg] = np.load(os.path.join(root, "emb_mean_%s_test.npy" % lg))

    def _labels(name):
        Y = []
        with open(os.path.join(REPO, "multieurlex", name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    Y.append(json.loads(line)["multi_hot"])
        return np.asarray(Y, dtype=np.int8)

    Ytr = _labels("en_train.jsonl")
    Yte = _labels("en_test.jsonl")
    ntr = len(Ytr)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(ntr)
    n_val = min(1000, max(1, ntr // 5))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    test_idx = np.arange(len(Yte))
    tr_idx, val_idx, test_idx = _limit_rows(tr_idx), _limit_rows(val_idx), _limit_rows(test_idx)
    d = dict(
        Xtr={lg: np.asarray(Xtr[lg][tr_idx]) for lg in LANG_MEL},
        Xva={lg: np.asarray(Xtr[lg][val_idx]) for lg in LANG_MEL},
        Xte={lg: np.asarray(Xte[lg][test_idx]) for lg in LANG_MEL},
        langs=LANG_MEL, targets=["de", "fr"], kind="multilabel", n_classes=21)
    d["ytr"] = {lg: Ytr[tr_idx] for lg in LANG_MEL}
    d["yva"] = {lg: Ytr[val_idx] for lg in LANG_MEL}
    d["yte"] = {lg: Yte[test_idx] for lg in LANG_MEL}
    return d


def _load_csv_binary(ds, backbone):
    import pandas as pd
    if ds == "sst2":
        df = pd.read_csv(os.path.join(REPO, "sst2.csv"))
        text_col, kind, n_classes = "sentence", "binary", None
    else:
        df = pd.read_csv(os.path.join(REPO, "emotion.csv"))
        text_col, kind, n_classes = "text", "multiclass", 6
    out_dir = os.path.join(BF16_ROOT, ds, backbone)
    tr_df = df[df["split"] == "train"]
    va_df = df[df["split"] == "validation"]
    te_df = df[df["split"] == "test"]
    tr_i = _limit_rows(np.arange(len(tr_df)))
    va_i = _limit_rows(np.arange(len(va_df)))
    te_i = _limit_rows(np.arange(len(te_df)))
    Xtr = np.load(os.path.join(out_dir, "mean_train.npy"))[tr_i]
    Xva = np.load(os.path.join(out_dir, "mean_val.npy"))[va_i]
    Xte = np.load(os.path.join(out_dir, "mean_test.npy"))[te_i]
    return dict(Xtr={"en": np.asarray(Xtr)}, Xva={"en": np.asarray(Xva)},
                Xte={"en": np.asarray(Xte)},
                ytr={"en": tr_df["label"].astype(int).to_numpy()[tr_i]},
                yva={"en": va_df["label"].astype(int).to_numpy()[va_i]},
                yte={"en": te_df["label"].astype(int).to_numpy()[te_i]},
                langs=["en"], targets=[], kind=kind, n_classes=n_classes)


def _load_toxic(backbone):
    import pandas as pd
    root = os.path.join(BF16_ROOT, "toxic", backbone)
    Xtr, Xva, Xte, ytr, yva, yte = {}, {}, {}, {}, {}, {}
    for lg in LANG_TOXI:
        df = pd.read_parquet(os.path.join(REPO, "data", "toxi_clean",
                                          "%s_train.parquet" % lg))
        dv = pd.read_parquet(os.path.join(REPO, "data", "toxi_clean",
                                          "%s_validation.parquet" % lg))
        dt = pd.read_parquet(os.path.join(REPO, "data", "toxi_clean",
                                          "%s_test.parquet" % lg))
        tr = _limit_rows(np.arange(len(df)))
        va = _limit_rows(np.arange(len(dv)))
        te = _limit_rows(np.arange(len(dt)))
        a = np.load(os.path.join(root, "emb_%s_train.npy" % lg))
        b = np.load(os.path.join(root, "emb_%s_validation.npy" % lg))
        c = np.load(os.path.join(root, "emb_%s_test.npy" % lg))
        Xtr[lg], Xva[lg], Xte[lg] = a[tr], b[va], c[te]
        ytr[lg] = df["is_toxic"].astype(int).to_numpy()[tr]
        yva[lg] = dv["is_toxic"].astype(int).to_numpy()[va]
        yte[lg] = dt["is_toxic"].astype(int).to_numpy()[te]
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=ytr, yva=yva, yte=yte,
                langs=LANG_TOXI, targets=["it", "de", "fr", "es"], kind="binary", n_classes=None)


def load_dataset(ds, backbone):
    key = (ds, backbone)
    if key in _DCACHE:
        return _DCACHE[key]
    if ds in ("flores_sent", "flores_agg"):
        d = _load_flores(ds, backbone)
    elif ds == "mel":
        d = _load_mel(backbone)
    elif ds in ("sst2", "emotion"):
        d = _load_csv_binary(ds, backbone)
    elif ds == "toxic":
        d = _load_toxic(backbone)
    else:
        raise ValueError("unknown dataset %r" % ds)
    d["dataset"] = ds
    d["backbone"] = backbone
    d["dim"] = BACKBONES[backbone]["dim"]
    if len(_DCACHE) > 2:                                     # keep memory bounded
        _DCACHE.clear()
    _DCACHE[key] = d
    return d


# ===========================================================================
# Feature transforms
# ===========================================================================
def _fit_inlp(Xtr, langs):
    """Iterated null-space projection of a language-identity probe on the stacked train pool."""
    if len(langs) < 2:
        return np.zeros((0, Xtr[langs[0]].shape[1]))
    from sklearn.linear_model import LogisticRegression
    mats, ys = [], []
    for i, lg in enumerate(langs):
        mats.append(np.asarray(Xtr[lg], dtype=np.float64))
        ys.append(np.full(len(Xtr[lg]), i))
    X = np.vstack(mats)
    y = np.concatenate(ys)
    d = X.shape[1]
    Xcur = X.copy()
    Qs = []
    for _r in range(1, 40):
        clf = LogisticRegression(max_iter=INLP_MAX_ITER, C=INLP_C)
        clf.fit(Xcur, y)
        W = np.asarray(clf.coef_, dtype=np.float64)
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
        Q = Vt[:int((S > 1e-8).sum())].copy()
        if Q.shape[0] == 0:
            break
        Qs.append(Q)
        Xcur = Xcur - (Xcur @ Q.T) @ Q
        if sum(q.shape[0] for q in Qs) >= INLP_MAX_DIRS:
            break
    Vall = np.vstack(Qs) if Qs else np.zeros((0, d))
    if Vall.shape[0]:
        _, Sv, Vt = np.linalg.svd(Vall, full_matrices=False)
        basis = Vt[:int((Sv > 1e-8).sum())].copy()
    else:
        basis = np.zeros((0, d))
    return basis[:INLP_MAX_DIRS]


def fit_transform(method, Xtr, langs, targets):
    if method == "mean":
        return ("mean", None)
    if method == "pc":
        from sklearn.decomposition import PCA
        stack = np.vstack([np.asarray(Xtr[lg], dtype=np.float64) for lg in langs])
        p = PCA(n_components=2, random_state=0).fit(stack)
        return ("pc", (p.mean_, p.components_))
    if method == "inlp":
        return ("inlp", _fit_inlp(Xtr, langs))
    if method == "D":
        mu_en = np.asarray(Xtr["en"], dtype=np.float64).mean(0)
        D = {lg: (mu_en - np.asarray(Xtr[lg], dtype=np.float64).mean(0)) for lg in targets}
        return ("D", D)
    raise ValueError("unknown method %r" % method)


def apply_transform(kind, state, X, lang):
    if kind == "mean":
        return np.asarray(X, dtype=np.float32)
    if kind == "pc":
        mean, comps = state
        X = np.asarray(X, dtype=np.float64)
        return (X - ((X - mean) @ comps.T) @ comps).astype(np.float32)
    if kind == "inlp":
        V = state
        X = np.asarray(X, dtype=np.float64)
        if V.shape[0] == 0:
            return X.astype(np.float32)
        return (X - (X @ V.T) @ V).astype(np.float32)
    if kind == "D":
        if lang == "en":
            return np.asarray(X, dtype=np.float32)
        return (np.asarray(X, dtype=np.float64) + state[lang]).astype(np.float32)
    raise ValueError(kind)


# ===========================================================================
# XGBoost / Optuna
# ===========================================================================
def _search_space(trial):
    try:
        from xgboost_topic_optuna import _optuna_search_space
        return _optuna_search_space(trial)
    except Exception:                                        # noqa: BLE001
        return dict(
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )


def _base_params(kind, n_classes):
    p = dict(tree_method="hist", max_bin=MAX_BIN, device="cpu", n_jobs=N_JOBS, seed=SEED)
    if kind == "multilabel":
        p.update(objective="binary:logistic")
    elif kind == "binary":
        p.update(objective="binary:logistic")
    else:
        p.update(objective="multi:softprob", num_class=int(n_classes))
    return p


def _fit(kind, n_classes, params, X, y):
    import xgboost as xgb
    X = np.ascontiguousarray(X, dtype=np.float32)
    if kind == "multilabel":
        models = {}
        for c in range(n_classes):
            yc = np.asarray(y[:, c], dtype=int)
            if yc.min() == yc.max():                         # degenerate concept -> constant
                continue
            p = _base_params(kind, n_classes)
            p.update(params)
            m = xgb.XGBClassifier(**p)
            m.fit(X, yc)
            models[c] = m
        return models
    p = _base_params(kind, n_classes)
    p.update(params)
    clf = xgb.XGBClassifier(**p)
    clf.fit(X, np.asarray(y, dtype=int))
    return clf


def _scores(kind, model, X, n_classes):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if kind == "multilabel":
        S = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for c, m in model.items():
            S[:, c] = m.predict_proba(X)[:, 1]
        return S
    P = model.predict_proba(X)
    if kind == "binary":
        return P[:, 1]
    # multiclass: map columns back to 0..C-1
    S = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        if 0 <= int(c) < n_classes:
            S[:, int(c)] = P[:, j]
    return S


def _auc_macro(kind, n_classes, y_true, S):
    from sklearn.metrics import roc_auc_score
    if kind == "multilabel":
        aucs = []
        for c in range(n_classes):
            g = np.asarray(y_true[:, c], dtype=int)
            if g.min() == g.max():
                continue
            aucs.append(roc_auc_score(g, S[:, c]))
        return float(np.mean(aucs)) if aucs else float("nan")
    if kind == "binary":
        y = np.asarray(y_true, dtype=int)
        if y.min() == y.max():
            return float("nan")
        return float(roc_auc_score(y, S))
    try:
        return float(roc_auc_score(y_true, S, multi_class="ovr", average="macro",
                                   labels=list(range(n_classes))))
    except Exception:                                        # noqa: BLE001
        return float("nan")


def _eval(kind, n_classes, y_true, S):
    from sklearn.metrics import f1_score, accuracy_score
    out = {}
    if kind == "multilabel":
        pred = (S >= 0.5).astype(int)
        out["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        out["micro_f1"] = float(f1_score(y_true, pred, average="micro", zero_division=0))
        out["subset_acc"] = float(accuracy_score(y_true, pred))
    elif kind == "binary":
        pred = (S >= 0.5).astype(int)
        y = np.asarray(y_true, dtype=int)
        out["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))
        out["micro_f1"] = float(f1_score(y, pred, average="micro", zero_division=0))
        out["accuracy"] = float(accuracy_score(y, pred))
    else:
        pred = np.argmax(S, axis=1)
        out["macro_f1"] = float(f1_score(y_true, pred, average="macro",
                                         labels=list(range(n_classes)), zero_division=0))
        out["micro_f1"] = float(f1_score(y_true, pred, average="micro", zero_division=0))
        out["accuracy"] = float(accuracy_score(y_true, pred))
    out["auc"] = _auc_macro(kind, n_classes, y_true, S)
    return out


def _tune(kind, n_classes, Xtr, ytr, Xva, yva, trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _search_space(trial)
        model = _fit(kind, n_classes, params, Xtr, ytr)
        S = _scores(kind, model, Xva, n_classes)
        auc = _auc_macro(kind, n_classes, yva, S)
        if not np.isfinite(auc):
            m = _eval(kind, n_classes, yva, S)
            trial.set_user_attr("fallback_macro_f1", m["macro_f1"])
            return float(m["macro_f1"])
        return float(auc)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    return dict(study.best_params), float(study.best_value)


# ===========================================================================
# One cell
# ===========================================================================
def run_cell_task(spec):
    dataset, backbone, method = spec
    out_path = cell_path(dataset, backbone, method)
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                    # noqa: BLE001
            pass
    t0 = time.time()
    trials = _ENV_TRIALS or N_TRIALS
    try:
        d = load_dataset(dataset, backbone)
        langs, targets = d["langs"], d["targets"]
        kind, n_classes = d["kind"], d["n_classes"]
        tkind, state = fit_transform(method, d["Xtr"], langs, targets)

        Xtr_en = apply_transform(tkind, state, d["Xtr"]["en"], "en")
        Xva_en = apply_transform(tkind, state, d["Xva"]["en"], "en")
        best_params, best_value = _tune(kind, n_classes, Xtr_en, d["ytr"]["en"],
                                        Xva_en, d["yva"]["en"], trials)
        model = _fit(kind, n_classes, best_params, Xtr_en, d["ytr"]["en"])

        en = _eval(kind, n_classes, d["yte"]["en"],
                   _scores(kind, model, apply_transform(tkind, state, d["Xte"]["en"], "en"),
                           n_classes))
        per_target = {}
        for lg in targets:
            Xte_lg = apply_transform(tkind, state, d["Xte"][lg], lg)
            per_target[lg] = _eval(kind, n_classes, d["yte"][lg],
                                   _scores(kind, model, Xte_lg, n_classes))

        keys = ["macro_f1", "micro_f1", "auc"] + (["subset_acc"] if kind == "multilabel" else [])
        xl = {}
        for k in keys:
            vals = [per_target[lg][k] for lg in targets if np.isfinite(per_target[lg][k])]
            xl[k] = float(np.mean(vals)) if vals else float("nan")

        result = {
            "dataset": dataset, "backbone": backbone, "method": method,
            "kind": kind, "n_classes": n_classes, "langs": langs, "targets": targets,
            "dim": d["dim"],
            "n_train": int(len(d["ytr"]["en"])), "n_val": int(len(d["yva"]["en"])),
            "n_test": int(len(d["yte"]["en"])),
            "transform": tkind,
            "inlp_dirs": int(state.shape[0]) if tkind == "inlp" else None,
            "best_params": best_params, "best_value": best_value, "n_trials": trials,
            "en": en, "per_target": per_target, "xl_mean": xl,
            "elapsed_s": round(time.time() - t0, 1),
        }
        _atomic_json(out_path, result)
        log("cell %-26s %-18s en_macro=%.3f xl_macro=%.3f xl_auc=%.3f (%.0fs)"
            % ("%s/%s/%s" % (dataset, backbone, method), "", en["macro_f1"],
               xl["macro_f1"], xl["auc"], result["elapsed_s"]))
        return result
    except Exception as exc:                                # noqa: BLE001
        result = {"dataset": dataset, "backbone": backbone, "method": method,
                  "state": "failed", "error": repr(exc),
                  "traceback": traceback.format_exc(),
                  "elapsed_s": round(time.time() - t0, 1)}
        log("cell FAILED %s/%s/%s: %r" % (dataset, backbone, method, exc))
        return result


# ===========================================================================
# toxi_pr_610.png
# ===========================================================================
def _perlang_z_fit(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Z = X / n
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def _perlang_z_apply(X, mu, sd):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n - mu) / sd


def _thr_f1_opt(y, s):
    from sklearn.metrics import f1_score
    grid = np.round(np.arange(0.0, 1.0000001, 0.01), 2)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def build_toxi_pr_610():
    """2x2 precision/recall-vs-threshold figure on the 610m toxic embeddings."""
    import pandas as pd
    import xgboost as xgb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(BF16_ROOT, "toxic", "610m")
    data_dir = os.path.join(REPO, "data", "toxi_clean")
    pair = ["en", "it"]
    raw = {}
    for lg in pair:
        tr = pd.read_parquet(os.path.join(data_dir, "%s_train.parquet" % lg))
        va = pd.read_parquet(os.path.join(data_dir, "%s_validation.parquet" % lg))
        te = pd.read_parquet(os.path.join(data_dir, "%s_test.parquet" % lg))
        raw[lg] = {
            "train": (np.load(os.path.join(root, "emb_%s_train.npy" % lg)),
                      tr["is_toxic"].astype(int).to_numpy()),
            "val": (np.load(os.path.join(root, "emb_%s_validation.npy" % lg)),
                    va["is_toxic"].astype(int).to_numpy()),
            "test": (np.load(os.path.join(root, "emb_%s_test.npy" % lg)),
                     te["is_toxic"].astype(int).to_numpy()),
        }
    zstats = {lg: _perlang_z_fit(raw[lg]["train"][0]) for lg in pair}

    def feats(lg, split):
        mu, sd = zstats[lg]
        return _perlang_z_apply(raw[lg][split][0], mu, sd).astype(np.float32)

    params = dict(objective="binary:logistic", tree_method="hist", max_bin=MAX_BIN,
                  device="cpu", n_jobs=N_JOBS, seed=0, max_depth=8, min_child_weight=5,
                  learning_rate=0.1, n_estimators=400)
    clfs, vthr = {}, {}
    for lg in pair:
        m = xgb.XGBClassifier(**params)
        m.fit(feats(lg, "train"), raw[lg]["train"][1])
        clfs[lg] = m
        sv = m.predict_proba(feats(lg, "val"))[:, 1]
        vthr[lg] = _thr_f1_opt(raw[lg]["val"][1], sv)

    cells = {}
    for ml in pair:
        for tl in pair:
            s = clfs[ml].predict_proba(feats(tl, "test"))[:, 1]
            cells["%s_model_on_%s" % (ml, tl)] = {"s": s, "y": raw[tl]["test"][1],
                                                  "val_thr": vthr[ml]}

    grid = np.round(np.arange(0.0, 1.0000001, 0.01), 2)
    colors = {"en_model_on_en": "#1f77b4", "en_model_on_it": "#d62728",
              "it_model_on_en": "#ff7f0e", "it_model_on_it": "#2ca02c"}
    labels = {"en_model_on_en": "EN model / EN test", "en_model_on_it": "EN model / IT test",
              "it_model_on_en": "IT model / EN test", "it_model_on_it": "IT model / IT test"}

    def sweep(cell, which):
        y, s = np.asarray(cell["y"]), np.asarray(cell["s"])
        n_pos = int(y.sum())
        out = []
        for t in grid:
            pred = s >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            p = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
            r = tp / n_pos if n_pos > 0 else np.nan
            out.append(p if which == "precision" else r)
        return np.array(out)

    order = ["en_model_on_en", "en_model_on_it", "it_model_on_en", "it_model_on_it"]
    layout = {
        (0, 0): ("precision", ["en_model_on_en"]),
        (0, 1): ("precision", ["en_model_on_it", "it_model_on_it"]),
        (1, 0): ("recall", ["en_model_on_en"]),
        (1, 1): ("recall", ["en_model_on_it", "it_model_on_it"]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey="row")
    for (r, c), (which, names) in layout.items():
        ax = axes[r, c]
        for nm in names:
            arr = sweep(cells[nm], which)
            ax.plot(grid, arr, color=colors[nm], lw=2.0, label=labels[nm])
            vt = cells[nm]["val_thr"]
            j = int(np.argmin(np.abs(grid - vt)))
            if np.isfinite(arr[j]):
                ax.plot([vt], [arr[j]], marker="o", ms=8, mfc="white", mec=colors[nm],
                        mew=1.8, zorder=6)
            ax.axvline(vt, color=colors[nm], ls=":", lw=1.0, alpha=0.6)
        ax.set_title(("Precision" if which == "precision" else "Recall") + " vs threshold",
                     fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.legend(fontsize=9.5, loc="upper right" if which == "recall" else "lower right",
                  framealpha=0.92)
    for c in (0, 1):
        axes[1, c].set_xlabel("decision threshold  t")
    axes[0, 0].set_ylabel("precision")
    axes[1, 0].set_ylabel("recall")
    fig.suptitle("toxi_pr_610: Precision (top) / Recall (bottom) vs decision threshold  --  "
                 "EuroBERT-610m mean + perlang_z (toxic clean)\n"
                 "left = EN model on EN test;  right = EN model on IT test + IT model on IT "
                 "test;  circles = val-tuned F1-optimal threshold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_TOXI610, dpi=130)
    plt.close(fig)
    log("wrote %s" % OUT_TOXI610)
    return cells


# ===========================================================================
# Outputs
# ===========================================================================
def _cell(result):
    if not result or result.get("state") == "failed":
        return None
    return result


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except Exception:                                        # noqa: BLE001
        return "n/a"
    return "n/a" if not np.isfinite(v) else ("%%.%df" % nd) % v


def _table(results, backbones, datasets, methods, metric, nd=3):
    """One markdown table: rows = dataset x backbone x method, columns = methods side by side.

    Instead of wide columns, emit long-form: dataset | backbone | method | EN self | XL mean.
    """
    lines = ["| dataset | backbone | method | EN self %s | XL-mean %s |" % (metric, metric),
             "|---|---|---|---:|---:|"]
    for ds in datasets:
        for bb in backbones:
            for m in methods:
                r = results.get((ds, bb, m))
                if not r or r.get("state") == "failed":
                    lines.append("| %s | %s | %s | n/a | n/a |" % (ds, bb, METHOD_LABEL[m]))
                    continue
                en = r["en"].get(metric)
                xl = r["xl_mean"].get(metric)
                lines.append("| %s | %s | %s | %s | %s |"
                             % (ds, bb, METHOD_LABEL[m], _fmt(en, nd), _fmt(xl, nd)))
    return lines


def write_outputs(results, cfg, backbones, datasets, methods):
    payload = {"config": cfg, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "cells": {"%s|%s|%s" % k: v for k, v in results.items()}}
    _atomic_json(OUT_JSON, payload)

    L = ["# backbone_grid -- XGBoost across EuroBERT-210m / 610m x datasets x methods", ""]
    L.append("Mean pooling only; XGBoost with %d Optuna TPE trials (seed %d) per cell; "
             "objective = val AUC (macro OVR). Trained on English, evaluated on the English "
             "test (EN self) and on every non-English target test; XL-mean = mean over "
             "non-English targets. B = backbone." % (cfg["n_trials"], SEED))
    L.append("")
    for metric, nd in (("macro_f1", 3), ("auc", 3), ("micro_f1", 3)):
        L.append("## %s" % metric)
        L.append("")
        L += _table(results, backbones, datasets, methods, metric, nd)
        L.append("")
    if any(r and r.get("kind") == "multilabel" for r in results.values()):
        L.append("## subset_acc (mel only)")
        L.append("")
        L += _table(results, backbones, datasets, methods, "subset_acc", 3)
        L.append("")

    L.append("## EN self / XL-mean side by side (macro-F1)")
    L.append("")
    L.append("| dataset | method | 210m EN | 210m XL | 610m EN | 610m XL |")
    L.append("|---|---|---:|---:|---:|---:|")
    for ds in datasets:
        for m in methods:
            r2 = results.get((ds, "210m", m))
            r6 = results.get((ds, "610m", m))
            L.append("| %s | %s | %s | %s | %s | %s |" % (
                ds, METHOD_LABEL[m],
                _fmt(r2["en"].get("macro_f1")) if r2 and r2.get("state") != "failed" else "n/a",
                _fmt(r2["xl_mean"].get("macro_f1")) if r2 and r2.get("state") != "failed" else "n/a",
                _fmt(r6["en"].get("macro_f1")) if r6 and r6.get("state") != "failed" else "n/a",
                _fmt(r6["xl_mean"].get("macro_f1")) if r6 and r6.get("state") != "failed" else "n/a"))
    L.append("")

    L.append("## Failed cells")
    L.append("")
    failed = [k for k, v in results.items() if not v or v.get("state") == "failed"]
    if failed:
        for k in failed:
            L.append("- %s: %s" % ("/".join(k), results[k].get("error")))
    else:
        L.append("- none")
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("wrote %s" % OUT_MD)

    # simple grouped bar chart: 210m vs 610m XL-mean macro-F1 per (dataset, method)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [(ds, m) for ds in datasets for m in methods]
        x = np.arange(len(rows))
        w = 0.38
        v210 = [results.get((ds, "210m", m), {}).get("xl_mean", {}).get("macro_f1", np.nan)
                if results.get((ds, "210m", m), {}).get("state") != "failed" else np.nan
                for ds, m in rows]
        v610 = [results.get((ds, "610m", m), {}).get("xl_mean", {}).get("macro_f1", np.nan)
                if results.get((ds, "610m", m), {}).get("state") != "failed" else np.nan
                for ds, m in rows]
        fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(rows)), 5))
        ax.bar(x - w / 2, v210, w, label="210m", color="#1f77b4")
        ax.bar(x + w / 2, v610, w, label="610m", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(["%s\n%s" % (ds, METHOD_LABEL[m].split()[0]) for ds, m in rows],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("XL-mean macro-F1")
        ax.set_title("backbone_grid: cross-lingual mean macro-F1 (EN-trained), 210m vs 610m")
        ax.legend()
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=130)
        plt.close(fig)
        log("wrote %s" % OUT_PNG)
    except Exception as exc:                                # noqa: BLE001
        log("plot skipped: %r" % (exc,))


# ===========================================================================
# Driver
# ===========================================================================
def run_all(args):
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    os.environ["BG_SMOKE"] = "1" if _SMOKE else "0"
    os.environ["BG_LIMIT"] = str(_LIMIT)
    os.environ["BG_TRIALS"] = str(_ENV_TRIALS)

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    state = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "smoke": _SMOKE, "phase": "start", "cells_done": [], "cells_failed": []}
    write_status(state)
    log("pid=%d smoke=%s trials=%d datasets=%s backbones=%s methods=%s workers=%d"
        % (os.getpid(), _SMOKE, _ENV_TRIALS or N_TRIALS, datasets, backbones, methods,
           args.workers))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.no_embed:
        state["phase"] = "embeddings"
        write_status(state)
        build_embeddings(backbones, device)

    state["phase"] = "grid"
    write_status(state)
    specs = [(ds, bb, m) for ds in datasets for bb in backbones for m in methods]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell_task, s): s for s in specs}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:                        # noqa: BLE001
                res = {"dataset": s[0], "backbone": s[1], "method": s[2],
                       "state": "failed", "error": repr(exc)}
            results[s] = res
            key = "%s/%s/%s" % s
            (state["cells_failed"] if res.get("state") == "failed"
             else state["cells_done"]).append(key)
            state["phase"] = "grid"
            write_status(state)

    state["phase"] = "toxi_pr_610"
    write_status(state)
    try:
        if "610m" in backbones and not args.no_toxi610:
            build_toxi_pr_610()
    except Exception as exc:                                # noqa: BLE001
        log("toxi_pr_610 FAILED: %r" % (exc,))

    cfg = {"datasets": datasets, "backbones": backbones, "methods": methods,
           "n_trials": _ENV_TRIALS or N_TRIALS, "seed": SEED, "pooling": "mean",
           "embedding_dtype": "bfloat16 forward -> float32 arrays", "emb_root": "emb_bf16",
           "optuna": "TPE seed 0", "objective": "val AUC (macro OVR)",
           "xgb": "hist cpu max_bin=%d n_jobs=%d" % (MAX_BIN, N_JOBS),
           "smoke": _SMOKE, "row_limit": _LIMIT}
    state["phase"] = "outputs"
    write_status(state)
    write_outputs(results, cfg, backbones, datasets, methods)
    state["phase"] = "done"
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["elapsed_s"] = round(time.time() - _T0, 1)
    write_status(state)
    log("ALL DONE in %.1fs" % (time.time() - _T0))
    return results


# ===========================================================================
# Detached launch
# ===========================================================================
def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, "-u", os.path.abspath(__file__)]
                                  + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, LOG_PATH, REPO], cwd=REPO)
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    global _SMOKE, _LIMIT, _ENV_TRIALS
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Two-backbone XGBoost grid (mean pooling).")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding-build phase")
    ap.add_argument("--no-toxi610", action="store_true", help="skip toxi_pr_610.png")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--backbones", default=",".join(BACKBONE_ORDER))
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)

    if args.launch:
        extra = [a for a in argv if a != "--launch"]
        launch_detached(extra)
        return 0

    _SMOKE = args.smoke
    _ENV_TRIALS = args.trials
    _LIMIT = args.limit_rows
    if _SMOKE:
        _ENV_TRIALS = min(_ENV_TRIALS, 2)
        _LIMIT = _LIMIT or 1500
        args.workers = min(args.workers, 2)
    run_all(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        raise
