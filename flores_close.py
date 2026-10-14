"""Close the English -> non-English transfer gap on FLORES (sent 12-class + agg 11-class).

Goal
----
With EuroBERT-210m (mean-pooled, 768-d) features + XGBoost, find the combination that best
CLOSES the gap between the English-trained model evaluated on a non-English target test set
and the *in-language ceiling* (that target language trained on its own train rows and
evaluated on its own test rows).  Every method always TRAINS on English (falling back to the
in-language model only for the translate-train/alignment variants that need the target) and is
evaluated on en / it / de / fr.

Protocol
--------
* sent : ``train_flores12.prepare_sentence()`` train=dev (997) / val=first 504 devtest /
         test=last 505 devtest; ``load_sent_embeddings('mean', kept, 2009)``.
* agg  : ``train_flores12.prepare_agg()`` (11 classes; "business and economy" dropped and the
         kept labels remapped to 0..10) / ``load_agg_embeddings('mean', 562)``.
* points are parallel across languages, so split indices apply row-for-row to every language.

Methods (all train EN; evaluate en/it/de/fr)
--------------------------------------------
feature transforms : raw, l2_thenZ, pc(k=2), pc(k=3), inlp, pc+inlp, pc+l2_thenZ, inlp+l2_thenZ.
  pc(k)  : PCA once on the STACKED-language TRAIN pool, zero the leading k components, inverse
           back to embedding space, then re-fit XGBoost (== remove the top-k principal dirs).
  inlp   : iterated null-space projection of a language-identity probe on the stacked TRAIN pool.
  l2_thenZ: L2-normalize then per-language TRAIN z-score (own-language stats).
kNN transfer : cosine kNN of each target test item against the EN train gallery, k in
           {1,3,5,10,20}, majority + similarity-weighted; reported per transform, best k.
translate-train : MT the EN train text into the target with Helsinki-NLP/opus-mt-en-{it,de,fr},
           embed with EuroBERT-210m mean pooling, train EN+translated, evaluate on target test.
           (GPU allowed just for this step; MT + embeddings cached under %TEMP%.)
nonlinear alignment : ``affine_moment`` (DiTTO-style unlabeled affine standardization of the
           target onto EN using target TRAIN stats) and ``mlp_align`` (1-hidden-layer MLP
           target->EN fitted on the parallel TRAIN rows, no labels).
ensembles : average probabilities of the best 2-3 feature-transform methods (+ translate-train).

Thresholds
----------
Every method reports the F1-optimal OVR threshold(s) tuned on EN val (translate-train tunes on
the target val, which is clean): a global micro-F1-optimal t, a global macro-F1-optimal t, and
per-class thresholds.  F1 is reported at those thresholds (plus argmax and macro OVR AUC).

Outputs
-------
``flores_close.json`` + ``flores_close.md`` ONLY (plus the operational .log/.pid/.status files
of a detached run).  CPU, n_jobs=2, 2 threads, no windows.

Usage
-----
    python flores_close.py --launch        # detached (hidden) full run
    python flores_close.py               # foreground full run
    python flores_close.py --smoke       # tiny plumbing check
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import json
import subprocess
import sys
import time
import warnings

import numpy as np

try:                                                    # keep the log readable
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:                                       # noqa: BLE001
    pass

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T                       # noqa: E402  (data prep + alignment)
import lang_neutralize as ln                      # noqa: E402

CACHE = os.path.join(os.environ.get("TEMP", REPO), "opencode", "flores_close_cache")
os.makedirs(CACHE, exist_ok=True)

OUT_JSON = os.path.join(REPO, "flores_close.json")
OUT_MD = os.path.join(REPO, "flores_close.md")
LOG_PATH = os.path.join(REPO, "flores_close.log")
PID_PATH = os.path.join(REPO, "flores_close.pid")
STATUS_PATH = os.path.join(REPO, "flores_close_status.json")

LANGS = ["en", "it", "de", "fr"]
TARGETS = ["it", "de", "fr"]
EMB_MODEL_ID = "EuroBERT/EuroBERT-210m"
MT_MODEL_ID = {"it": "Helsinki-NLP/opus-mt-en-it",
               "de": "Helsinki-NLP/opus-mt-en-de",
               "fr": "Helsinki-NLP/opus-mt-en-fr"}

FEATURE_METHODS = ["raw", "l2_thenZ", "pc2", "pc3", "inlp", "pc+inlp",
                   "pc+l2_thenZ", "inlp+l2_thenZ"]
KNN_KS = [1, 3, 5, 10, 20]
GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 2)

# Fixed XGBoost config (same for every method / ceiling -> fair comparison).
XGB_PARAMS = dict(objective="multi:softprob", tree_method="hist", device="cpu",
                  max_bin=64, n_jobs=2, seed=0, n_estimators=200, max_depth=6,
                  learning_rate=0.1, subsample=0.8, colsample_bytree=0.4,
                  reg_lambda=1.0, min_child_weight=1, gamma=0.0)

INLP_ROUNDS = 6
INLP_MAX_ITER = 200
INLP_C = 1.0
MLP_EPOCHS = 200
MLP_HIDDEN = 256

_T0 = time.time()
_SMOKE = False


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, STATUS_PATH)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataset(name, emb_root=None, emb_agg_root=None):
    sent = T.prepare_sentence()
    classes = sent["classes"]
    agg = T.prepare_agg(sent["cur"], classes)
    if name == "sent":
        X = T.load_sent_embeddings("mean", sent["kept"], len(sent["git"]))
        cur = sent["cur"]
        y = sent["y"]
        tr = sent["train_idx"]
        va = sent["val_idx"]
        te = sent["test_idx"]
        n_classes = 12
        cls = classes
    else:
        X = T.load_agg_embeddings("mean", len(agg["agg"]))
        cur = agg["agg"]
        y = agg["y"]
        tr = agg["train_idx"]
        va = agg["val_idx"]
        te = agg["test_idx"]
        n_classes = agg["n_classes"]
        cls = agg["classes"]
    texts = {lg: [("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v))
                  for v in cur[lg].tolist()] for lg in LANGS}
    # sanity: y defined on all split rows
    trval = np.concatenate([tr, va])
    assert y[te].min() >= 0 and y[trval].min() >= 0
    return {"name": name, "X": {lg: np.asarray(X[lg], dtype=np.float64) for lg in LANGS},
            "y": np.asarray(y, dtype=int), "train": tr, "val": va, "test": te,
            "trval": trval, "n_classes": n_classes, "classes": cls, "texts": texts,
            "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}}


# ---------------------------------------------------------------------------
# Feature transforms.  A transform returns {lang: (n, d) float64}.
# ---------------------------------------------------------------------------
def fit_pc(stacked, k):
    from sklearn.decomposition import PCA
    p = PCA(n_components=int(k), random_state=0).fit(stacked)
    return p.mean_, p.components_          # (d,), (k, d)


def apply_pc(X, mean, comps):
    return X - ((X - mean) @ comps.T) @ comps


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_l2z(X_by_lang, train_idx):
    stats = {}
    for lg in LANGS:
        Z = l2norm(X_by_lang[lg][train_idx])
        mu = Z.mean(0)
        sd = Z.std(0)
        sd[sd < 1e-6] = 1e-6
        stats[lg] = (mu, sd)
    return stats


def apply_l2z(X_by_lang, stats):
    out = {}
    for lg in LANGS:
        mu, sd = stats[lg]
        out[lg] = (l2norm(X_by_lang[lg]) - mu) / sd
    return out


def inlp_basis(X_by_lang, train_idx, n_rounds=INLP_ROUNDS):
    """Iterated null-space projection of a language-identity probe on the stacked TRAIN pool."""
    from sklearn.linear_model import LogisticRegression
    mats, ys = [], []
    for i, lg in enumerate(LANGS):
        mats.append(X_by_lang[lg][train_idx])
        ys.append(np.full(len(train_idx), i))
    Xs = np.vstack(mats)
    ys = np.concatenate(ys)
    d = Xs.shape[1]
    Xcur = Xs.copy()
    Qs = []
    ranks = []
    for r in range(1, n_rounds + 1):
        clf = LogisticRegression(max_iter=INLP_MAX_ITER, C=INLP_C, n_jobs=2)
        clf.fit(Xcur, ys)
        W = np.asarray(clf.coef_, dtype=np.float64)
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
        Q = Vt[:int((S > 1e-8).sum())].copy()
        Qs.append(Q)
        ranks.append(int(Q.shape[0]))
        Xcur = ln.project_out(Xcur, Q)
        if Q.shape[0] == 0:
            break
    Vall = np.vstack(Qs) if Qs else np.zeros((0, d))
    if Vall.shape[0]:
        _, Sv, Vt = np.linalg.svd(Vall, full_matrices=False)
        basis = Vt[:int((Sv > 1e-8).sum())].copy()
    else:
        basis = np.zeros((0, d))
    return basis, ranks


def apply_proj(X, V):
    return ln.project_out(X, V)


def build_transforms(D, cache_status=None):
    """Return {method: {lang: features}} for every FEATURE_METHODS entry."""
    tr, va = D["train"], D["val"]
    trval = D["trval"]
    X = D["X"]
    stacked_tr = np.vstack([X[lg][tr] for lg in LANGS])
    out = {"raw": X}

    # l2_thenZ
    out["l2_thenZ"] = apply_l2z(X, fit_l2z(X, trval))

    # pc(k)
    for k, nm in ((2, "pc2"), (3, "pc3")):
        mean, comps = fit_pc(stacked_tr, k)
        out[nm] = {lg: apply_pc(X[lg], mean, comps) for lg in LANGS}

    # inlp
    basis, ranks = inlp_basis(X, trval)
    if cache_status is not None:
        cache_status["inlp"] = {"n_removed": int(basis.shape[0]), "ranks": ranks}
    out["inlp"] = {lg: apply_proj(X[lg], basis) for lg in LANGS}

    # pc(2)+inlp
    mean, comps = fit_pc(stacked_tr, 2)
    pc2 = {lg: apply_pc(X[lg], mean, comps) for lg in LANGS}
    basis2, _ = inlp_basis(pc2, trval)
    out["pc+inlp"] = {lg: apply_proj(pc2[lg], basis2) for lg in LANGS}
    if cache_status is not None:
        cache_status["pc+inlp"] = {"n_removed": int(basis2.shape[0])}

    # pc(2)+l2_thenZ
    out["pc+l2_thenZ"] = apply_l2z(pc2, fit_l2z(pc2, trval))

    # inlp+l2_thenZ
    out["inlp+l2_thenZ"] = apply_l2z(out["inlp"], fit_l2z(out["inlp"], trval))
    return out


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def fit_xgb(Xtr, ytr, n_classes):
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    p["num_class"] = int(n_classes)
    if _SMOKE:
        p["n_estimators"] = 60
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xtr, ytr)
    return clf


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def prf_ovr(y, P, t):
    """Micro + macro F1 of the OVR decisions ``class c positive iff P[:,c] >= t``."""
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
        a = int((pp & yy).sum())
        b = int((pp & ~yy).sum())
        cc = int((~pp & yy).sum())
        f1s.append((2.0 * a / (2.0 * a + b + cc)) if (2 * a + b + cc) > 0 else 0.0)
    return micro, float(np.mean(f1s)), f1s


def prf_ovr_perclass(y, P, ts):
    n, C = P.shape
    Y = np.zeros((n, C), dtype=bool)
    Y[np.arange(n), y] = True
    Ppos = P >= np.asarray(ts)[None, :]
    tp = int((Ppos & Y).sum())
    fp = int((Ppos & ~Y).sum())
    fn = int((~Ppos & Y).sum())
    micro = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    f1s = []
    for c in range(C):
        pp, yy = Ppos[:, c], Y[:, c]
        a = int((pp & yy).sum())
        b = int((pp & ~yy).sum())
        cc = int((~pp & yy).sum())
        f1s.append((2.0 * a / (2.0 * a + b + cc)) if (2 * a + b + cc) > 0 else 0.0)
    return micro, float(np.mean(f1s)), f1s


def auc_macro(y, P, n_classes):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, P, multi_class="ovr", average="macro",
                                   labels=list(range(n_classes))))
    except Exception:
        return float("nan")


def best_t_global(yv, Pv, mode):
    bt, bv = 0.5, -1.0
    for t in GRID:
        mic, mac, _ = prf_ovr(yv, Pv, t)
        v = mic if mode == "micro" else mac
        if v > bv:
            bv, bt = v, float(t)
    return bt


def best_t_perclass(yv, Pv, n_classes):
    ts = np.full(Pv.shape[1], 0.5)
    for c in range(Pv.shape[1]):
        bt, bv = 0.5, -1.0
        for t in GRID:
            _, _, f1s = prf_ovr(yv, Pv, t)
            if f1s[c] > bv:
                bv, bt = f1s[c], float(t)
        ts[c] = bt
    return ts


def evaluate(y_true, P, y_val, P_val, n_classes):
    from sklearn.metrics import accuracy_score, f1_score
    t_micro = best_t_global(y_val, P_val, "micro")
    t_macro = best_t_global(y_val, P_val, "macro")
    tvec = best_t_perclass(y_val, P_val, n_classes)
    mic_m, mac_m, _ = prf_ovr(y_true, P, t_micro)
    mic_M, mac_M, _ = prf_ovr(y_true, P, t_macro)
    mic_pc, mac_pc, _ = prf_ovr_perclass(y_true, P, tvec)
    pred = P.argmax(1)
    return {
        "auc": auc_macro(y_true, P, n_classes),
        "acc": float(accuracy_score(y_true, pred)),
        "argmax_macro_f1": float(f1_score(y_true, pred, average="macro",
                                          labels=list(range(n_classes)), zero_division=0)),
        "t_micro": float(t_micro), "micro_f1": float(mic_m), "macro_f1_at_tmicro": float(mac_m),
        "t_macro": float(t_macro), "macro_f1": float(mac_M), "micro_f1_at_tmacro": float(mic_M),
        "perclass_t": [float(v) for v in tvec],
        "tm_micro_f1": float(mic_pc), "tm_macro_f1": float(mac_pc),
    }


# ---------------------------------------------------------------------------
# kNN retrieval transfer
# ---------------------------------------------------------------------------
def knn_scores(galleryX, gallery_y, queryX, k, weighted):
    """Return (queryX.shape[0], C) score matrix for cosine kNN."""
    G = galleryX / np.maximum(np.linalg.norm(galleryX, axis=1, keepdims=True), 1e-12)
    Q = queryX / np.maximum(np.linalg.norm(queryX, axis=1, keepdims=True), 1e-12)
    C = int(gallery_y.max()) + 1
    sims = Q @ G.T                                       # (q, n_gallery)
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]   # (q, k)
    scores = np.zeros((sims.shape[0], C))
    for r in range(sims.shape[0]):
        nb = idx[r]
        s = sims[r, nb]
        if weighted:
            w = np.clip(s, 0.0, None)
            if w.sum() <= 0:
                w = np.ones_like(w)
            for j, c in enumerate(gallery_y[nb]):
                scores[r, int(c)] += w[j]
        else:
            for c in gallery_y[nb]:
                scores[r, int(c)] += 1.0
    # normalise to a probability-like vector
    ssum = scores.sum(1, keepdims=True)
    ssum[ssum == 0] = 1.0
    return scores / ssum


def run_knn(D, TX):
    """{method: {lang: {k_variant: metrics}}}; only on a light subset of transforms."""
    methods = ["raw", "l2_thenZ", "pc2", "pc3", "inlp", "pc+inlp"] if not _SMOKE else ["raw", "pc2"]
    out = {}
    y = D["y"]
    te = D["test"]
    trval = D["trval"]
    tr = D["train"]
    ks = KNN_KS if not _SMOKE else [5]
    for m in methods:
        Xf = TX[m]
        out[m] = {}
        for lg in LANGS:
            out[m][lg] = {}
            for k in ks:
                for wname, w in (("maj", False), ("wt", True)):
                    key = "%s%d" % ("w" if w else "m", k)
                    Pte = knn_scores(Xf["en"][trval], y[trval], Xf[lg][te], k, w)
                    # tune threshold on EN val using gallery = EN train only (no leakage)
                    Pva = knn_scores(Xf["en"][tr], y[tr], Xf["en"][D["val"]], k, w)
                    ev = evaluate(y[te], Pte, y[D["val"]], Pva, D["n_classes"])
                    ev["k"] = int(k)
                    ev["weighted"] = bool(w)
                    out[m][lg][key] = ev
    return out


def knn_best(out, metric="micro_f1"):
    """Return per transform and overall the best (method, variant) by mean over target langs."""
    ranked = []
    for m, per_lang in out.items():
        variants = list(next(iter(per_lang.values())).keys())
        for v in variants:
            vals = [per_lang[lg][v][metric] for lg in TARGETS]
            ranked.append({"method": m, "variant": v, "mean_target_%s" % metric: float(np.mean(vals))})
    ranked.sort(key=lambda r: r["mean_target_%s" % metric], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# translate-train
# ---------------------------------------------------------------------------
def _load_mt(lang):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    mid = MT_MODEL_ID[lang]
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mid, local_files_only=True)
    return tok, mdl


def _load_embedder():
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(EMB_MODEL_ID, local_files_only=True)
    mdl = AutoModel.from_pretrained(EMB_MODEL_ID, local_files_only=True)
    return tok, mdl


def _embed(texts, tok, mdl, device, batch=64, max_len=511):
    import torch
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(device)
        with torch.inference_mode():
            h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def translate_and_embed(D, lang):
    """MT the EN train+val text into `lang`, embed with EuroBERT mean; cached."""
    tag = "%s_%s" % (D["name"], lang)
    epath = os.path.join(CACHE, "tt_emb_%s.npy" % tag)
    if os.path.isfile(epath) and not _SMOKE:
        return np.load(epath)
    import torch
    trval = D["trval"]
    en_texts = [D["texts"]["en"][i] for i in trval]
    if _SMOKE:
        en_texts = en_texts[:40]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mt_tok, mt_mdl = _load_mt(lang)
    mt_mdl = mt_mdl.to(dev).eval()
    trans = []
    bs = 64
    for i in range(0, len(en_texts), bs):
        b = mt_tok(en_texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                   max_length=512).to(dev)
        with torch.inference_mode():
            g = mt_mdl.generate(**b, max_new_tokens=256, num_beams=1, do_sample=False)
        trans.extend(mt_tok.batch_decode(g, skip_special_tokens=True))
    del mt_mdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    etok, emdl = _load_embedder()
    emdl = emdl.to(dev).eval()
    E = _embed(trans, etok, emdl, dev)
    del emdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    if not _SMOKE:
        np.save(epath, E)
    return E


def run_translate_train(D):
    """{lang: metrics}; train on EN train+val + translated train+val; eval on target test."""
    out = {}
    y = D["y"]
    te = D["test"]
    va = D["val"]
    ytrval = y[D["trval"]]
    Xen_trval = D["X"]["en"][D["trval"]]
    for lg in TARGETS:
        MT = translate_and_embed(D, lg)
        n = min(len(MT), len(Xen_trval))
        Xtr = np.vstack([Xen_trval[:n], MT[:n]])
        ytr = np.concatenate([ytrval[:n], ytrval[:n]])
        clf = fit_xgb(Xtr, ytr, D["n_classes"])
        Pte = clf.predict_proba(D["X"][lg][te])
        Pva = clf.predict_proba(D["X"][lg][va])          # clean target val for thresholds
        out[lg] = evaluate(y[te], Pte, y[va], Pva, D["n_classes"])
        log("  translate-train %s micro_f1=%.4f macro_f1=%.4f auc=%.4f"
            % (lg, out[lg]["micro_f1"], out[lg]["macro_f1"], out[lg]["auc"]))
    return out


# ---------------------------------------------------------------------------
# Nonlinear / affine alignment
# ---------------------------------------------------------------------------
def map_affine_moment(X_en, X_t, tr):
    """DiTTO-style unlabeled affine: standardize target -> EN (own TRAIN stats)."""
    mu_e, sd_e = X_en[tr].mean(0), X_en[tr].std(0)
    mu_t, sd_t = X_t[tr].mean(0), X_t[tr].std(0)
    sd_e[sd_e < 1e-6] = 1e-6
    sd_t[sd_t < 1e-6] = 1e-6
    return lambda Z: (Z - mu_t) / sd_t * sd_e + mu_e


def _mlp_fit(Xsrc, Xdst, epochs=MLP_EPOCHS, hidden=MLP_HIDDEN):
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    d = Xsrc.shape[1]
    mu_s, sd_s = Xsrc.mean(0), Xsrc.std(0)
    mu_d, sd_d = Xdst.mean(0), Xdst.std(0)
    sd_s[sd_s < 1e-6] = 1e-6
    sd_d[sd_d < 1e-6] = 1e-6
    S = ((Xsrc - mu_s) / sd_s).astype(np.float32)
    Tn = ((Xdst - mu_d) / sd_d).astype(np.float32)
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(S)
    Yt = torch.from_numpy(Tn)
    n = len(S)
    bs = 256
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
    net.eval()

    def apply(Z):
        Zs = ((Z - mu_s) / sd_s).astype(np.float32)
        with torch.inference_mode():
            P = net(torch.from_numpy(Zs)).numpy().astype(np.float64)
        return P * sd_d + mu_d
    return apply


def run_alignment(D):
    """{'affine_moment': {lang: metrics}, 'mlp_align': {lang: metrics}} for target langs."""
    out = {"affine_moment": {}, "mlp_align": {}}
    y = D["y"]
    te = D["test"]
    va = D["val"]
    tr = D["trval"]
    Xen = D["X"]["en"]
    # EN self model (used for the EN column of alignment methods)
    en_clf = fit_xgb(Xen[tr], y[tr], D["n_classes"])
    Pen = en_clf.predict_proba(Xen[te])
    Pen_va = en_clf.predict_proba(Xen[va])
    en_metrics = evaluate(y[te], Pen, y[va], Pen_va, D["n_classes"])
    for m in ("affine_moment", "mlp_align"):
        out[m]["en"] = en_metrics
    for lg in TARGETS:
        Xt = D["X"][lg]
        # affine moment (unlabeled)
        fA = map_affine_moment(Xen, Xt, tr)
        PA = en_clf.predict_proba(fA(Xt[te]))
        PAva = en_clf.predict_proba(fA(Xt[va]))
        out["affine_moment"][lg] = evaluate(y[te], PA, y[va], PAva, D["n_classes"])
        # mlp align (parallel, label-free)
        fM = _mlp_fit(Xt[tr], Xen[tr], epochs=(30 if _SMOKE else MLP_EPOCHS))
        PM = en_clf.predict_proba(fM(Xt[te]))
        PMva = en_clf.predict_proba(fM(Xt[va]))
        out["mlp_align"][lg] = evaluate(y[te], PM, y[va], PMva, D["n_classes"])
        log("  align %s affine micro=%.4f mlp micro=%.4f"
            % (lg, out["affine_moment"][lg]["micro_f1"], out["mlp_align"][lg]["micro_f1"]))
    return out


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------
def run_ensembles(D, feature_probs, feature_val_probs, base_results, method_names,
                  tt_probs=None, tt_val_probs=None):
    """Average probs of the top-3 feature methods (+ translate-train for target langs)."""
    ranked = sorted(method_names,
                    key=lambda m: np.mean([base_results[m][lg]["micro_f1"] for lg in TARGETS]),
                    reverse=True)
    top = ranked[:3]
    ens = {lg: np.mean([feature_probs[m][lg] for m in top], axis=0) for lg in LANGS}
    valprobs = {lg: np.mean([feature_val_probs[m][lg] for m in top], axis=0) for lg in LANGS}

    def eval_ens(probs, valprobs_l, tag):
        res = {}
        for lg in LANGS:
            res[lg] = evaluate(D["y"][D["test"]], probs[lg], D["y"][D["val"]],
                               valprobs_l[lg], D["n_classes"])
        return {"members": list(top), "tag": tag, "metrics": res}

    out = {"ens_top3": eval_ens(ens, valprobs, "top3 feature methods")}

    if tt_probs:
        ens2 = {lg: ens[lg] for lg in LANGS}
        val2 = {lg: valprobs[lg] for lg in LANGS}
        for lg in TARGETS:
            ens2[lg] = (ens[lg] + tt_probs[lg]) / 2.0
            if tt_val_probs and lg in tt_val_probs:
                val2[lg] = (valprobs[lg] + tt_val_probs[lg]) / 2.0
        out["ens_top3+tt"] = eval_ens(ens2, val2, "top3 feature methods + translate-train")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def gap_closure(raw_gap, meth_gap):
    if raw_gap <= 1e-9:
        return float("nan")
    return (raw_gap - meth_gap) / raw_gap


def build_report(all_results):
    """Compose the md + json payload from per-dataset results."""
    md = []
    md.append("# FLORES English -> non-English gap closure (EuroBERT-210m + XGBoost)\n")
    md.append("In-language ceiling = the target language trained on its own train+val rows and "
              "tested on its own test rows (raw features). Every method trains on English. "
              "Gap = ceiling - method, averaged over it/de/fr. Primary metric = OVR micro-F1 at "
              "the F1-optimal global threshold (tuned on EN val; translate-train on target val). "
              "Micro/macro-F1 and macro OVR AUC are all reported.\n")

    payload = {"config": {"emb_model": EMB_MODEL_ID, "xgb_params": XGB_PARAMS,
                          "feature_methods": FEATURE_METHODS, "knn_ks": KNN_KS,
                          "threshold_grid": 0.02, "inlp_rounds": INLP_ROUNDS,
                          "mlp_epochs": MLP_EPOCHS, "mlp_hidden": MLP_HIDDEN},
               "datasets": {}}

    primary = "micro_f1"
    for ds, R in all_results.items():
        ceiling = R["ceiling"]
        methods = R["methods"]                      # method -> {lang: metrics}
        cnames = list(methods.keys())
        md.append("\n## %s (n_classes=%d, splits %s)\n" % (ds, R["n_classes"], R["sizes"]))
        md.append("**In-language ceiling** (micro-F1 / macro-F1 / AUC): "
                  + "  ".join("%s %.4f/%.4f/%.4f" % (lg, ceiling[lg]["micro_f1"],
                                                      ceiling[lg]["macro_f1"], ceiling[lg]["auc"])
                              for lg in LANGS) + "\n")

        # per-method table (primary metric + macro f1 + auc + gaps)
        md.append("| method | EN micro | it micro | de micro | fr micro | mean tgt micro "
                  "| mean gap (micro) | gap-closed % | it/de/fr macro-F1 | AUC mean |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        rows = []
        for m in cnames:
            mm = methods[m]
            en = mm["en"][primary]
            tgt = [mm[lg][primary] for lg in TARGETS]
            gap = float(np.mean([ceiling[lg][primary] - mm[lg][primary] for lg in TARGETS]))
            macro_t = [mm[lg]["macro_f1"] for lg in TARGETS]
            auc_t = [mm[lg]["auc"] for lg in TARGETS]
            rows.append({"method": m, "en": en, "tgt": tgt, "mean_tgt": float(np.mean(tgt)),
                         "gap": gap, "macro_t": macro_t, "auc_t": auc_t})
            md.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s | %.4f |"
                      % (m, en, tgt[0], tgt[1], tgt[2], np.mean(tgt), gap, "-",
                         "/".join("%.3f" % v for v in macro_t), np.mean(auc_t)))

        raw = next(r for r in rows if r["method"] == "raw")
        for r in rows:
            r["closure"] = gap_closure(raw["gap"], r["gap"])

        best = min(rows, key=lambda r: r["gap"])
        payload["datasets"][ds] = {
            "n_classes": R["n_classes"], "sizes": R["sizes"],
            "classes": R["classes"],
            "ceiling": {lg: ceiling[lg] for lg in LANGS},
            "methods": {m: {lg: methods[m][lg] for lg in LANGS} for m in cnames},
            "knn_best": R.get("knn_best", []),
            "translations_meta": R.get("translations_meta", {}),
            "inlp_meta": R.get("inlp_meta", {}),
            "rows": rows,
            "best_by_gap": best["method"],
            "raw_gap": raw["gap"],
        }
        md.append("\n* raw (baseline) mean target micro-F1 gap = %.4f.*\n" % raw["gap"])

        # closure table
        md.append("| method | mean tgt micro | macro-F1 it/de/fr | AUC it/de/fr | gap(ceiling-method) "
                  "| gap closed vs raw |")
        md.append("|---|---:|---|---|---:|---:|")
        for r in sorted(rows, key=lambda x: x["gap"]):
            cl = r["closure"]
            flag = " **(>=90%)**" if (np.isfinite(cl) and cl >= 0.90) else ""
            md.append("| %s | %.4f | %s | %s | %.4f | %s%s |"
                      % (r["method"], r["mean_tgt"],
                         "/".join("%.3f" % v for v in r["macro_t"]),
                         "/".join("%.3f" % v for v in r["auc_t"]),
                         r["gap"], ("%.1f%%" % (100 * cl)) if np.isfinite(cl) else "n/a", flag))
        md.append("")

    # overall best combination
    best_overall = None
    best_gap = None
    for ds, R in all_results.items():
        p = payload["datasets"][ds]
        for r in p["rows"]:
            if best_gap is None or r["gap"] < best_gap:
                best_gap = r["gap"]
                best_overall = (ds, r["method"], r["gap"], r["closure"], r["mean_tgt"])
    md.append("\n## Best combination\n")
    if best_overall:
        ds, m, g, cl, mt = best_overall
        md.append("Best single combination: **%s** on **%s** (mean target micro-F1 gap = %.4f, "
                  "gap closed vs raw = %s, mean target micro-F1 = %.4f).\n"
                  % (m, ds, g, ("%.1f%%" % (100 * cl)) if np.isfinite(cl) else "n/a", mt))
    payload["best_overall"] = ({"dataset": best_overall[0], "method": best_overall[1],
                                "gap": best_overall[2], "closure": best_overall[3],
                                "mean_target_micro_f1": best_overall[4]}
                               if best_overall else None)
    return md, payload


# ---------------------------------------------------------------------------
# Driver for one dataset
# ---------------------------------------------------------------------------
def run_dataset(ds):
    log("==== DATASET %s ====" % ds)
    D = build_dataset(ds)
    log("%s: sizes=%s n_classes=%d" % (ds, D["sizes"], D["n_classes"]))
    base = {"n_classes": D["n_classes"], "sizes": D["sizes"], "classes": D["classes"]}
    cache_status = {}
    TX = build_transforms(D, cache_status)
    base["inlp_meta"] = cache_status
    y = D["y"]
    tr, va, te = D["trval"], D["val"], D["test"]
    nc = D["n_classes"]

    methods = {}
    per_method_probs = {}          # test probs for ensemble
    per_method_val_probs = {}      # EN-val probs for ensemble threshold tuning

    # ---- feature transforms ----
    for m in FEATURE_METHODS:
        Xf = TX[m]
        clf = fit_xgb(Xf["en"][tr], y[tr], nc)
        per_method_probs[m] = {}
        per_method_val_probs[m] = {}
        for lg in LANGS:
            P = clf.predict_proba(Xf[lg][te])
            Pv = clf.predict_proba(Xf["en"][va])       # threshold tuning always on EN val
            per_method_probs[m][lg] = P
            per_method_val_probs[m][lg] = Pv
            methods[m] = methods.get(m, {})
            methods[m][lg] = evaluate(y[te], P, y[va], Pv, nc)
        log("  [%s] raw-en micro=%.4f | tgt micro=%s"
            % (m, methods[m]["en"]["micro_f1"],
               "/".join("%.4f" % methods[m][lg]["micro_f1"] for lg in TARGETS)))

    # ---- ceilings (raw features, per language) ----
    ceiling = {}
    for lg in LANGS:
        clf = fit_xgb(TX["raw"][lg][tr], y[tr], nc)
        P = clf.predict_proba(TX["raw"][lg][te])
        Pv = clf.predict_proba(TX["raw"][lg][va])
        ceiling[lg] = evaluate(y[te], P, y[va], Pv, nc)
        log("  ceiling %s micro=%.4f macro=%.4f auc=%.4f"
            % (lg, ceiling[lg]["micro_f1"], ceiling[lg]["macro_f1"], ceiling[lg]["auc"]))
    base["ceiling"] = ceiling

    # ---- kNN ----
    knn = run_knn(D, TX)
    base["knn_best"] = knn_best(knn)[:6]
    # expose the best kNN variant as a first-class method row
    kb = knn_best(knn)
    if kb:
        bm, bv = kb[0]["method"], kb[0]["variant"]
        methods["knn(%s,%s)" % (bm, bv)] = {lg: knn[bm][lg][bv] for lg in LANGS}
    base["knn_detail"] = {m: {lg: list(vs.keys()) for lg, vs in per.items()}
                          for m, per in knn.items()}

    # ---- translate-train ----
    tt_probs = {}
    tt_val = {}
    ytrval = y[tr]
    Xen_trval = D["X"]["en"][tr]
    for lg in TARGETS:
        try:
            MT = translate_and_embed(D, lg)
            n = min(len(MT), len(Xen_trval))
            Xtr = np.vstack([Xen_trval, MT[:n]])
            ytr = np.concatenate([ytrval, ytrval[:n]])
            clf = fit_xgb(Xtr, ytr, nc)
            Pte = clf.predict_proba(D["X"][lg][te])
            Pva = clf.predict_proba(D["X"][lg][va])
            tt_probs[lg] = Pte
            tt_val[lg] = Pva
            ev = evaluate(y[te], Pte, y[va], Pva, nc)
            log("  translate-train %s micro=%.4f macro=%.4f auc=%.4f"
                % (lg, ev["micro_f1"], ev["macro_f1"], ev["auc"]))
        except Exception as exc:                        # noqa: BLE001
            log("  translate-train %s FAILED: %r" % (lg, exc))
    if tt_probs:
        methods["translate-train"] = {}
        for lg in TARGETS:
            methods["translate-train"][lg] = evaluate(y[te], tt_probs[lg], y[va], tt_val[lg], nc)
        methods["translate-train"]["en"] = methods["raw"]["en"]
        for lg in TARGETS:
            per_method_probs.setdefault("translate-train", {})[lg] = tt_probs[lg]
            per_method_val_probs.setdefault("translate-train", {})[lg] = tt_val[lg]

    # ---- alignment ----
    try:
        al = run_alignment(D)
        for m in ("affine_moment", "mlp_align"):
            methods[m] = al[m]
            per_method_probs[m] = {} if m not in per_method_probs else per_method_probs[m]
    except Exception as exc:                            # noqa: BLE001
        log("  alignment FAILED: %r" % (exc,))

    # ---- ensembles ----
    try:
        ens = run_ensembles(D, per_method_probs, per_method_val_probs, methods,
                            FEATURE_METHODS, tt_probs=tt_probs or None,
                            tt_val_probs=tt_val or None)
        for m, rec in ens.items():
            methods[m] = rec["metrics"]
    except Exception as exc:                            # noqa: BLE001
        log("  ensembles FAILED: %r" % (exc,))
        import traceback
        traceback.print_exc()

    base["methods"] = methods
    base["translations_meta"] = {"mt_models": MT_MODEL_ID, "cache": CACHE}
    return base


# ---------------------------------------------------------------------------
# Launch / main
# ---------------------------------------------------------------------------
def launch_detached(extra):
    vbs = os.path.join(CACHE, "launch_flores_close.vbs")
    logp = LOG_PATH
    py = sys.executable
    script = os.path.abspath(__file__)
    cmd = 'cmd /c ""%s" "%s" %s > "%s" 2>&1"' % (py, script, " ".join(extra), logp)
    with open(vbs, "w", encoding="ascii") as f:
        f.write('Set sh = CreateObject("WScript.Shell")\r\n')
        f.write('sh.Run "%s", 0, False\r\n' % cmd.replace('"', '""'))
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs], cwd=REPO)
    print("launched detached via %s (log=%s)" % (vbs, logp))


def main(argv=None):
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    global _SMOKE
    ap = argparse.ArgumentParser(description="Close the EN->non-EN FLORES gap (EuroBERT+XGB).")
    ap.add_argument("--launch", action="store_true", help="spawn a detached hidden copy and exit")
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check")
    ap.add_argument("--datasets", default="sent,agg", help="comma list of sent,agg")
    args = ap.parse_args(argv)

    if args.launch:
        extra = [a for a in (argv or sys.argv[1:]) if a != "--launch"]
        launch_detached(extra)
        return 0
    _SMOKE = args.smoke

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    state = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "smoke": bool(_SMOKE), "datasets": {}}
    write_status(state)

    dss = ["sent"] if _SMOKE else [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}
    try:
        for ds in dss:
            state["dataset"] = ds
            write_status(state)
            all_results[ds] = run_dataset(ds)
            state["datasets"][ds] = {"methods": list(all_results[ds]["methods"].keys())}
            write_status(state)
        md, payload = build_report(all_results)
        payload["elapsed_s"] = round(time.time() - _T0, 1)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["elapsed_s"] = round(time.time() - _T0, 1)
        write_status(state)
        log("DONE in %.1fs -> %s , %s" % (time.time() - _T0, OUT_JSON, OUT_MD))
    except Exception as exc:                            # noqa: BLE001
        import traceback
        traceback.print_exc()
        state["error"] = repr(exc)
        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_status(state)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
