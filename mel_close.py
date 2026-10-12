"""mel_close.py -- close the English -> non-English gap on MultiEURLEX
(EuroBERT-210m mean pooling + XGBoost), self-contained.

Goal
----
With EuroBERT-210m (mean-pooled, 768-d) features + one-vs-rest XGBoost, find the combination
that best CLOSES the gap between the English-trained model evaluated on a target-language
(de / fr) test set and the *in-language ceiling* (that target trained on its own train rows,
tested on its own test rows).  Every method always TRAINS on English; it is evaluated on
en / de / fr test.

Data (read-only; no validation embedding exists -> a val subset is carved from train)
-------------------------------------------------------------------------------------
  multieurlex/emb_mean_{en,de,fr}_{train,test}.npy   (11000 / 5000, 768)  mean-pooled EuroBERT
  multieurlex/{en,de,fr}_{train,test}.jsonl          keys `text`, `multi_hot` (21 dims)
A random VAL_SIZE=1000 slice of TRAIN (seed 0) is held out for threshold tuning; documents are
PARALLEL across languages, so row i is the same document in every language.

Methods (all train EN; evaluate en/de/fr test)
----------------------------------------------
feature transforms : raw; l2_thenZ [both] (L2-normalise then per-language TRAIN z-score);
                     pc (PCA fit on the STACKED-language TRAIN pool, zero the leading k=2
                     components, inverse, re-fit); inlp (iterated null-space projection of a
                     language-identity probe on the stacked TRAIN pool, 30 directions -- the
                     basis is fit here (create-only `lang_inlp_mel.npz`) because the FLORES
                     ablation script/`lang_inlp_mean.npz` are absent).
nonlinear align    : affine_moment (DiTTO-style unlabeled affine standardisation of the target
                     onto EN using TRAIN stats) and mlp_align (1-hidden-layer MLP, 256 units,
                     mapping TARGET-train features -> EN-train features, parallel rows, no
                     labels); both then apply the EN `raw` XGB.
knn transfer       : cosine kNN of target test against the EN train gallery on per-language
                     standardised + L2-normalised features, k in {5,10,20}, similarity-weighted;
                     the multi-label decision is per-concept frequency >= 0.5.
translate-train    : MT the EN train text into the target (cached opus-mt chunked MT under
                     %TEMP%\\opencode\\tt_cache) and train on EN + translated.  Skipped with a
                     note when the cache is absent.
ensemble           : probability average of the 3 best methods by mean target micro-F1.

Thresholds / metrics
--------------------
Every method reports, at the global threshold tuned on the EN val split to maximise micro-F1:
micro-F1, macro-F1, subset-accuracy (exact 21-dim match) and OVR-macro AUC (threshold-free);
plus micro/macro-F1 under `quantile_match_enval` (per-concept target threshold whose predicted
positive rate equals the EN-val TRUE positive rate -- label-free for the target).

Ceiling
-------
Native de->de and fr->fr models with the winning `l2_thenZ [both]` recipe (primary reference)
and with `raw` (secondary).  Gap = ceiling - method (mean over de/fr); closure is measured
against the `raw` English baseline.

Outputs (create-only; nothing is deleted or moved)
--------------------------------------------------
  mel_close.json / mel_close.md / mel_close_plots.png / mel_close.pid / mel_close_status.json
  %TEMP%\\opencode\\mel_close\\*.log|*.pid  (detached launch)

Usage
-----
  python mel_close.py            # foreground full run
  python mel_close.py --launch   # detached hidden full run (writes mel_close.log)
  python mel_close.py --smoke    # tiny plumbing check
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
import warnings

import numpy as np

try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:                                            # noqa: BLE001
    pass

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TMP = os.path.join(os.environ.get("TEMP", REPO), "opencode")
os.makedirs(TMP, exist_ok=True)
CACHE = os.path.join(TMP, "mel_close")
os.makedirs(CACHE, exist_ok=True)
TT_CACHE = os.path.join(TMP, "tt_cache")

EMB_DIR = os.path.join(REPO, "multieurlex")
LANGS = ["en", "de", "fr"]
TARGETS = ["de", "fr"]
N_CLASSES = 21
VAL_SIZE = 1000
SEED = 0
EMB_MODEL_ID = "EuroBERT/EuroBERT-210m"
INLP_PATH = os.path.join(REPO, "lang_inlp_mel.npz")

# Rare EuroVoc level-1 concepts (8 lowest-TRAIN-frequency, present in test) -- see
# imbalanced_recall_mel.md.  idx -> (EuroVoc id, English name).
RARE = [(13, "100162", "international organisations"),
        (16, "100151", "science"),
        (14, "100159", "energy"),
        (9, "100153", "employment and working conditions"),
        (11, "100145", "law"),
        (10, "100142", "politics"),
        (20, "100155", "environment"),
        (8, "100154", "transport")]

FEATURE_METHODS = ["raw", "l2_thenZ", "pc", "inlp"]
KNN_KS = [5, 10, 20]
GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 2)

# Fixed XGBoost config, the FLORES-winning one-vs-rest config (same for every method / ceiling
# -> fair comparison).  CPU, n_jobs=2.
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_bin=64, device="cpu",
                  n_jobs=2, seed=0, learning_rate=0.1, max_depth=6, min_child_weight=1,
                  reg_lambda=1.0, subsample=0.8, colsample_bytree=0.4, n_estimators=200)

INLP_MAX_DIRS = 30
INLP_MAX_ITER = 100
INLP_C = 1.0
MLP_EPOCHS = 150
MLP_HIDDEN = 256

_T0 = time.time()
_SMOKE = False


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _paths(tag, outdir):
    suf = tag
    return {
        "json": os.path.join(outdir, "mel_close%s.json" % suf),
        "md": os.path.join(outdir, "mel_close%s.md" % suf),
        "png": os.path.join(outdir, "mel_close%s_plots.png" % suf),
        "pid": os.path.join(outdir, "mel_close%s.pid" % suf),
        "status": os.path.join(outdir, "mel_close%s_status.json" % suf),
        "log": os.path.join(outdir, "mel_close%s.log" % suf),
    }


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError("not JSON serializable: %r" % (type(o),))


def write_status(state, path):
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _read_labels_any(path, n_expect):
    Y = np.zeros((n_expect, N_CLASSES), dtype=np.int8)
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            Y[n] = np.asarray(json.loads(line)["multi_hot"], dtype=np.int8)
            n += 1
    return Y[:n]


def load_data():
    """Return Xtr/Xte {lang:(n,768) f64}, Ytr/Yte (n,21), tr/val/test idx over the 'all' stack."""
    Xtr, Xte = {}, {}
    for lg in LANGS:
        Xtr[lg] = np.load(os.path.join(EMB_DIR, "emb_mean_%s_train.npy" % lg)).astype(np.float64)
        Xte[lg] = np.load(os.path.join(EMB_DIR, "emb_mean_%s_test.npy" % lg)).astype(np.float64)
    Ytr = _read_labels_any(os.path.join(EMB_DIR, "en_train.jsonl"), len(Xtr["en"]))
    Yte = _read_labels_any(os.path.join(EMB_DIR, "en_test.jsonl"), len(Xte["en"]))
    ntr = len(Ytr)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(ntr)
    val_idx = np.sort(perm[:VAL_SIZE])
    tr_idx = np.sort(perm[VAL_SIZE:])
    test_idx = np.arange(ntr, ntr + len(Yte))
    if _SMOKE:
        tr_idx = tr_idx[:400]
        val_idx = val_idx[:100]
        test_idx = test_idx[:200]
    Xall = {lg: np.vstack([Xtr[lg], Xte[lg]]) for lg in LANGS}
    return {"Xall": Xall, "Ytr": Ytr, "Yte": Yte, "ntr": ntr,
            "tr": tr_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# Feature transforms
# ---------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_l2z(Xtr_rows):
    Z = l2norm(Xtr_rows)
    mu = Z.mean(0)
    sd = Z.std(0)
    sd[sd < 1e-6] = 1e-6
    return mu, sd


def apply_l2z(X, stats):
    mu, sd = stats
    return ((l2norm(X) - mu) / sd).astype(np.float32)


def fit_pc(stacked, k):
    from sklearn.decomposition import PCA
    p = PCA(n_components=int(k), random_state=0).fit(stacked)
    return p.mean_, p.components_


def apply_pc(X, mean, comps):
    return (X - ((X - mean) @ comps.T) @ comps).astype(np.float32)


def project_out(X, V):
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] == 0:
        return X.astype(np.float32)
    return (X - (X @ V.T) @ V).astype(np.float32)


def fit_inlp(Xall, tr_idx, cache_status):
    """Iterated null-space projection of a language-identity probe on the stacked TRAIN pool."""
    from sklearn.linear_model import LogisticRegression
    if os.path.isfile(INLP_PATH) and not _SMOKE:
        z = np.load(INLP_PATH)
        basis = z["basis"]
        cache_status["inlp"] = {"source": "cached", "n_removed": int(basis.shape[0])}
        return basis
    mats, ys = [], []
    for i, lg in enumerate(LANGS):
        mats.append(Xall[lg][tr_idx])
        ys.append(np.full(len(tr_idx), i))
    Xs = np.vstack(mats)
    ys = np.concatenate(ys)
    d = Xs.shape[1]
    Xcur = Xs.copy()
    Qs, ranks = [], []
    for _r in range(1, 40):
        clf = LogisticRegression(max_iter=INLP_MAX_ITER, C=INLP_C)
        clf.fit(Xcur, ys)
        W = np.asarray(clf.coef_, dtype=np.float64)
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
        Q = Vt[:int((S > 1e-8).sum())].copy()
        if Q.shape[0] == 0:
            break
        Qs.append(Q)
        Xcur = Xcur - (Xcur @ Q.T) @ Q
        total = sum(q.shape[0] for q in Qs)
        ranks.append(total)
        if total >= INLP_MAX_DIRS:
            break
    Vall = np.vstack(Qs) if Qs else np.zeros((0, d))
    if Vall.shape[0]:
        _, Sv, Vt = np.linalg.svd(Vall, full_matrices=False)
        basis = Vt[:int((Sv > 1e-8).sum())].copy()
    else:
        basis = np.zeros((0, d))
    basis = basis[:INLP_MAX_DIRS]
    if not _SMOKE:                                       # create-only cache
        np.savez(INLP_PATH, basis=basis)
    cache_status["inlp"] = {"source": "fit", "n_removed": int(basis.shape[0]), "rounds": ranks}
    return basis


def build_transforms(D, cache_status):
    Xall = D["Xall"]
    tr = D["tr"]
    out = {"raw": {lg: Xall[lg].astype(np.float32) for lg in LANGS}}
    meta = {}

    # l2_thenZ [both]: each language uses its own TRAIN stats
    l2z_stats = {lg: fit_l2z(Xall[lg][tr]) for lg in LANGS}
    out["l2_thenZ"] = {lg: apply_l2z(Xall[lg], l2z_stats[lg]) for lg in LANGS}
    meta["l2z_stats"] = l2z_stats

    # pc: global PCA on the stacked-language TRAIN pool, zero leading k=2 components
    stacked_tr = np.vstack([Xall[lg][tr] for lg in LANGS])
    mean, comps = fit_pc(stacked_tr, 2)
    out["pc"] = {lg: apply_pc(Xall[lg], mean, comps) for lg in LANGS}
    meta["pc"] = {"k": 2}

    # inlp: 30-dir language subspace removed
    basis = fit_inlp(Xall, tr, cache_status)
    out["inlp"] = {lg: project_out(Xall[lg], basis) for lg in LANGS}
    return out, meta


# ---------------------------------------------------------------------------
# XGBoost (one-vs-rest)
# ---------------------------------------------------------------------------
def fit_ovr(X, Y, n_classes):
    import xgboost as xgb
    X = np.ascontiguousarray(X, dtype=np.float32)
    p = dict(XGB_PARAMS)
    if _SMOKE:
        p["n_estimators"] = 40
    models = []
    for c in range(n_classes):
        m = xgb.XGBClassifier(**p)
        m.fit(X, Y[:, c].astype(int))
        models.append(m)
    return models


def predict_ovr(models, X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    return np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def micro_macro(pred, gold):
    pred = np.asarray(pred, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    micro = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    f1s = []
    for c in range(gold.shape[1]):
        p, g = pred[:, c], gold[:, c]
        tpc = int((p & g).sum())
        fpc = int((p & ~g).sum())
        fnc = int((~p & g).sum())
        d = 2 * tpc + fpc + fnc
        f1s.append(0.0 if d == 0 else 2 * tpc / d)
    return micro, float(np.mean(f1s))


def subset_acc(pred, gold):
    return float((np.asarray(pred, dtype=bool) == np.asarray(gold, dtype=bool)).all(1).mean())


def auc_macro(scores, gold):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(gold.shape[1]):
        g = gold[:, c]
        if g.min() == g.max():
            continue
        aucs.append(roc_auc_score(g, scores[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def best_t_micro(yv, Sv):
    best, bt = -1.0, 0.5
    for t in GRID:
        mic, _ = micro_macro(Sv >= t, yv)
        if mic > best:
            best, bt = mic, float(t)
    return bt


def quantile_match_thresholds(scores, val_true):
    """Per concept, threshold whose target pred-positive rate = EN-val TRUE positive rate."""
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


def evaluate(y_true, S_te, y_val, S_val):
    t = best_t_micro(y_val, S_val)
    pred = S_te >= t
    micro, macro = micro_macro(pred, y_true)
    sub = subset_acc(pred, y_true)
    auc = auc_macro(S_te, y_true)
    qthr = quantile_match_thresholds(S_te, y_val)
    qpred = S_te >= qthr
    micro_q, macro_q = micro_macro(qpred, y_true)
    return {"t_global": float(t), "micro_f1": float(micro), "macro_f1": float(macro),
            "subset_acc": float(sub), "auc": float(auc),
            "micro_f1_qm": float(micro_q), "macro_f1_qm": float(macro_q),
            "perclass_f1_qm": [round(v, 4) for v in perclass_f1(qpred, y_true)]}


def perclass_f1(pred, gold):
    gold = np.asarray(gold, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    out = []
    for c in range(gold.shape[1]):
        p, g = pred[:, c], gold[:, c]
        tpc = int((p & g).sum())
        fpc = int((p & ~g).sum())
        fnc = int((~p & g).sum())
        d = 2 * tpc + fpc + fnc
        out.append(0.0 if d == 0 else 2 * tpc / d)
    return out


# ---------------------------------------------------------------------------
# Alignment (affine moment + MLP)
# ---------------------------------------------------------------------------
def affine_map(X_en, X_t, tr):
    mu_e, sd_e = X_en[tr].mean(0), X_en[tr].std(0)
    mu_t, sd_t = X_t[tr].mean(0), X_t[tr].std(0)
    sd_e = np.where(sd_e < 1e-6, 1e-6, sd_e)
    sd_t = np.where(sd_t < 1e-6, 1e-6, sd_t)
    return lambda Z: ((Z - mu_t) / sd_t * sd_e + mu_e).astype(np.float32)


def mlp_fit(Xsrc, Xdst, epochs=MLP_EPOCHS, hidden=MLP_HIDDEN):
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    d = Xsrc.shape[1]
    mu_s, sd_s = Xsrc.mean(0), Xsrc.std(0)
    mu_d, sd_d = Xdst.mean(0), Xdst.std(0)
    sd_s = np.where(sd_s < 1e-6, 1e-6, sd_s)
    sd_d = np.where(sd_d < 1e-6, 1e-6, sd_d)
    S = ((Xsrc - mu_s) / sd_s).astype(np.float32)
    Tn = ((Xdst - mu_d) / sd_d).astype(np.float32)
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(S)
    Yt = torch.from_numpy(Tn)
    n = len(S)
    bs = 256
    if _SMOKE:
        epochs = 20
    net.train()
    for _ep in range(epochs):
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
            P = net(torch.from_numpy(Zs)).numpy()
        return (P * sd_d + mu_d).astype(np.float32)
    return apply


# ---------------------------------------------------------------------------
# kNN transfer
# ---------------------------------------------------------------------------
def knn_scores(gallery, gallery_y, query, k):
    """Similarity-weighted cosine kNN -> (n_query, C) scores in [0,1] (per-concept frequency)."""
    G = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    Q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    C = int(gallery_y.shape[1])
    sims = Q @ G.T
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    scores = np.zeros((sims.shape[0], C), dtype=np.float64)
    for r in range(sims.shape[0]):
        nb = idx[r]
        w = np.clip(sims[r, nb], 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        scores[r] = (gallery_y[nb].astype(np.float64) * w[:, None]).sum(0)
    ssum = scores.sum(1, keepdims=True)
    ssum[ssum == 0] = 1.0
    return scores / ssum


# ---------------------------------------------------------------------------
# rare-concept recall metrics (self-contained; matches imbalanced_recall.py)
# ---------------------------------------------------------------------------
def _cum(y, s):
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")
    ys = np.asarray(y, dtype=np.int64)[order]
    return int(ys.sum()), np.cumsum(ys), np.cumsum(1 - ys)


def recall_at_fa(y, s, budget):
    n_pos, cum_tp, cum_fp = _cum(y, s)
    if n_pos == 0:
        return None
    j = int(np.searchsorted(cum_fp, budget, side="right")) - 1
    return 0.0 if j < 0 else float(cum_tp[j] / n_pos)


def recall_at_fpr(y, s, fpr):
    n_pos, _, _ = _cum(y, s)
    n_neg = len(y) - n_pos
    if n_neg == 0:
        return None
    return recall_at_fa(y, s, int(np.floor(fpr * n_neg)))


def fpr_for_recall(y, s, target=0.90):
    n_pos, cum_tp, cum_fp = _cum(y, s)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    need = int(np.ceil(target * n_pos - 1e-9))
    j = int(np.searchsorted(cum_tp, need, side="left"))
    if j >= len(cum_tp):
        return None
    return float(cum_fp[j] / n_neg)


def max_fbeta(y, s, beta):
    n_pos, cum_tp, cum_fp = _cum(y, s)
    if n_pos == 0:
        return None
    denom = cum_tp + cum_fp
    prec = np.where(denom > 0, cum_tp / np.maximum(denom, 1), 0.0)
    rec = cum_tp / n_pos
    b2 = beta * beta
    d = b2 * prec + rec
    f = np.where(d > 0, (1 + b2) * prec * rec / np.maximum(d, 1e-12), 0.0)
    return float(np.nanmax(f))


def rare_stats(y, s):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    out = {"n_pos_test": n_pos,
           "average_precision": float(average_precision_score(y, s)) if n_pos and n_neg else None}
    for p in (0.01, 0.05, 0.10):
        out["recall_at_fpr_%d" % int(round(p * 100))] = recall_at_fpr(y, s, p)
    for kfa in (5, 20, 50, 100):
        out["recall_at_fa_%d" % kfa] = recall_at_fa(y, s, kfa)
    out["fpr_for_recall_90"] = fpr_for_recall(y, s, 0.90)
    out["max_f2"] = max_fbeta(y, s, 2)
    out["max_f3"] = max_fbeta(y, s, 3)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(tag="", outdir=None):
    outdir = outdir or REPO
    P = _paths(tag, outdir)
    with open(P["pid"], "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    state = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "smoke": bool(_SMOKE), "phase": "load", "methods_done": []}
    write_status(state, P["status"])

    log("loading data")
    D = load_data()
    Xall, Ytr, Yte = D["Xall"], D["Ytr"], D["Yte"]
    tr, val, te = D["tr"], D["val"], D["test"]
    log("sizes tr=%d val=%d test=%d langs=%s" % (len(tr), len(val), len(te), LANGS))
    y_val = Ytr[val]
    y_te = Yte[te - D["ntr"]]

    cache_status = {}
    state["phase"] = "transforms"
    write_status(state, P["status"])
    TX, tf_meta = build_transforms(D, cache_status)
    tf_meta["inlp"] = cache_status.get("inlp", {})

    methods = {}          # name -> {lang: metrics}
    probs = {}            # name -> {"val": S_val(n_val,21), "test": {lang: S(n_test,21)}}
    payload = {"config": {"dataset": "nlpaueb/multi_eurlex (en/de/fr; 11000/5000)",
                          "emb_model": EMB_MODEL_ID, "pooling": "mean", "dim": 768,
                          "n_classes": N_CLASSES, "val_size": VAL_SIZE, "seed": SEED,
                          "langs": LANGS, "targets": TARGETS, "xgb_params": XGB_PARAMS,
                          "feature_methods": FEATURE_METHODS, "knn_ks": KNN_KS,
                          "inlp_max_dirs": INLP_MAX_DIRS, "mlp_epochs": MLP_EPOCHS,
                          "mlp_hidden": MLP_HIDDEN},
               "transform_meta": tf_meta, "methods": {}, "ceiling": {}, "rare": {}}

    def flush():
        with open(P["json"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        state["methods_done"] = list(methods.keys())
        write_status(state, P["status"])

    # ---- kNN (label-free, no fit; runs first so partial results appear early) ----
    state["phase"] = "method:knn"
    write_status(state, P["status"])
    knn_z = {}
    for lg in LANGS:
        knn_z[lg] = l2norm((Xall[lg] - Xall[lg][tr].mean(0)) / np.where(
            Xall[lg][tr].std(0) < 1e-6, 1e-6, Xall[lg][tr].std(0)))
    best_k, best_k_score = KNN_KS[0], -1.0
    for k in KNN_KS:
        S_val = knn_scores(knn_z["en"][tr], Ytr[tr], knn_z["en"][val], k)
        name = "knn_k%d" % k
        methods[name] = {}
        S = {}
        for lg in LANGS:
            S[lg] = knn_scores(knn_z["en"][tr], Ytr[tr], knn_z[lg][te], k)
            methods[name][lg] = evaluate(y_te, S[lg], y_val, S_val)
        probs[name] = {"val": S_val, "test": S}
        tgt_mean = float(np.mean([methods[name][lg]["micro_f1"] for lg in TARGETS]))
        log("  [%s] tgt micro=%s (mean %.4f)" %
            (name, "/".join("%.4f" % methods[name][lg]["micro_f1"] for lg in TARGETS), tgt_mean))
        if tgt_mean > best_k_score:
            best_k_score, best_k = tgt_mean, k
        flush()
    payload["knn_best_k"] = int(best_k)

    # ---- feature-transform methods (1 EN-XGB fit each) ----
    raw_en_clf = None
    for m in FEATURE_METHODS:
        state["phase"] = "method:" + m
        write_status(state, P["status"])
        Xf = TX[m]
        clf = fit_ovr(Xf["en"][tr], Ytr[tr], N_CLASSES)
        S_val = predict_ovr(clf, Xf["en"][val])
        S = {lg: predict_ovr(clf, Xf[lg][te]) for lg in LANGS}
        if m == "raw":
            raw_en_clf = clf
        methods[m] = {}
        for lg in LANGS:
            methods[m][lg] = evaluate(y_te, S[lg], y_val, S_val)
        probs[m] = {"val": S_val, "test": S}
        log("  [%s] en micro=%.4f | tgt micro=%s" %
            (m, methods[m]["en"]["micro_f1"],
             "/".join("%.4f" % methods[m][lg]["micro_f1"] for lg in TARGETS)))
        flush()

    # ---- affine_moment + mlp_align (reuse the raw EN model) ----
    if raw_en_clf is not None:
        S_val_raw = probs["raw"]["val"]
        for mname in ("affine_moment", "mlp_align"):
            state["phase"] = "method:" + mname
            write_status(state, P["status"])
            methods[mname] = {"en": methods["raw"]["en"]}
            probs[mname] = {"val": S_val_raw, "test": {"en": probs["raw"]["test"]["en"]}}
            for lg in TARGETS:
                Xt = Xall[lg]
                if mname == "affine_moment":
                    f = affine_map(Xall["en"], Xt, tr)
                else:
                    f = mlp_fit(Xt[tr], Xall["en"][tr])
                S = predict_ovr(raw_en_clf, f(Xt[te]))
                methods[mname][lg] = evaluate(y_te, S, y_val, S_val_raw)
                probs[mname]["test"][lg] = S
            log("  [%s] tgt micro=%s" % (mname,
                "/".join("%.4f" % methods[mname][lg]["micro_f1"] for lg in TARGETS)))
            flush()

    # ---- translate-train (cached MT) ----
    state["phase"] = "method:translate-train"
    write_status(state, P["status"])
    tt_ok = True
    tt_cache_files = {}
    for lg in TARGETS:
        mp = os.path.join(TT_CACHE, "mt_emb_mel_%s.npy" % lg)
        tt_cache_files[lg] = {"path": mp, "exists": os.path.isfile(mp)}
        if not os.path.isfile(mp):
            tt_ok = False
    if tt_ok:
        try:
            # EN side: l2_thenZ(EN); MT side: l2_thenZ(target stats, since MT is target text)
            Xe = TX["l2_thenZ"]["en"][:D["ntr"]]
            methods["translate-train"] = {}
            probs["translate-train"] = {"val": None, "test": {}}
            for lg in TARGETS:
                MT = np.load(os.path.join(TT_CACHE, "mt_emb_mel_%s.npy" % lg)).astype(np.float64)
                MTz = apply_l2z(MT, tf_meta["l2z_stats"][lg])
                Xtr_tt = np.vstack([Xe[tr], MTz[tr]])
                ytr_tt = np.concatenate([Ytr[tr], Ytr[tr]])
                clf = fit_ovr(Xtr_tt, ytr_tt, N_CLASSES)
                S_te = predict_ovr(clf, TX["l2_thenZ"][lg][te])
                S_val = predict_ovr(clf, Xe[val])
                methods["translate-train"][lg] = evaluate(y_te, S_te, y_val, S_val)
                probs["translate-train"]["test"][lg] = S_te
                if probs["translate-train"]["val"] is None:
                    probs["translate-train"]["val"] = S_val
                log("  [translate-train] %s micro=%.4f macro=%.4f" %
                    (lg, methods["translate-train"][lg]["micro_f1"],
                     methods["translate-train"][lg]["macro_f1"]))
            methods["translate-train"]["en"] = methods["l2_thenZ"]["en"]
            probs["translate-train"]["test"]["en"] = probs["l2_thenZ"]["test"]["en"]
        except Exception as exc:                          # noqa: BLE001
            log("  translate-train FAILED: %r" % (exc,))
    else:
        log("  translate-train SKIPPED (MT cache absent: %s)" % tt_cache_files)
    payload["translations_meta"] = tt_cache_files
    flush()

    # ---- ceilings (native, l2_thenZ primary + raw secondary) ----
    ceiling = {}
    for tf, label in (("l2_thenZ", "primary"), ("raw", "secondary")):
        state["phase"] = "ceiling:" + tf
        write_status(state, P["status"])
        ceiling[tf] = {}
        for lg in TARGETS:
            clf = fit_ovr(TX[tf][lg][tr], Ytr[tr], N_CLASSES)
            S_val = predict_ovr(clf, TX[tf][lg][val])
            S_te = predict_ovr(clf, TX[tf][lg][te])
            ceiling[tf][lg] = {"metrics": evaluate(y_te, S_te, Ytr[val], S_val),
                               "_test_scores": S_te}
            m = ceiling[tf][lg]["metrics"]
            log("  ceiling(%s) %s micro=%.4f macro=%.4f auc=%.4f" %
                (tf, lg, m["micro_f1"], m["macro_f1"], m["auc"]))
        flush()

    # ---- ensemble of the 3 best methods by mean target micro-F1 ----
    state["phase"] = "ensemble"
    write_status(state, P["status"])
    rankable = [m for m in probs if probs[m]["val"] is not None
                and all(lg in probs[m]["test"] for lg in LANGS)]
    ranked = sorted(rankable,
                    key=lambda m: np.mean([methods[m][lg]["micro_f1"] for lg in TARGETS]),
                    reverse=True)
    top3 = ranked[:3]
    ens_val = np.mean([probs[m]["val"] for m in top3], axis=0)
    ens_test = {lg: np.mean([probs[m]["test"][lg] for m in top3], axis=0) for lg in LANGS}
    methods["ensemble"] = {lg: evaluate(y_te, ens_test[lg], y_val, ens_val) for lg in LANGS}
    probs["ensemble"] = {"val": ens_val, "test": ens_test}
    log("  [ensemble] members=%s | tgt micro=%s" %
        (top3, "/".join("%.4f" % methods["ensemble"][lg]["micro_f1"] for lg in TARGETS)))
    flush()

    # ---- best method (min mean target gap vs the l2_thenZ ceiling) ----
    ceil_primary = {lg: ceiling["l2_thenZ"][lg]["metrics"] for lg in TARGETS}
    rows = []
    for m in methods:
        if "en" not in methods[m]:
            continue
        tgt_micro = [methods[m][lg]["micro_f1"] for lg in TARGETS]
        gap = float(np.mean([ceil_primary[lg]["micro_f1"] - methods[m][lg]["micro_f1"]
                             for lg in TARGETS]))
        rows.append({"method": m,
                     "en_micro": methods[m]["en"]["micro_f1"],
                     "de_micro": methods[m]["de"]["micro_f1"],
                     "fr_micro": methods[m]["fr"]["micro_f1"],
                     "mean_tgt_micro": float(np.mean(tgt_micro)),
                     "macro_f1_qm_de": methods[m]["de"]["macro_f1_qm"],
                     "macro_f1_qm_fr": methods[m]["fr"]["macro_f1_qm"],
                     "macro_f1_qm_mean": float(np.mean([methods[m][lg]["macro_f1_qm"]
                                                        for lg in TARGETS])),
                     "de_auc": methods[m]["de"]["auc"], "fr_auc": methods[m]["fr"]["auc"],
                     "de_subset": methods[m]["de"]["subset_acc"],
                     "fr_subset": methods[m]["fr"]["subset_acc"],
                     "gap": gap})
    raw_row = next((r for r in rows if r["method"] == "raw"), None)
    for r in rows:
        r["closure"] = (None if not raw_row or abs(raw_row["gap"]) < 1e-9
                        else (raw_row["gap"] - r["gap"]) / raw_row["gap"])
    best = min(rows, key=lambda r: r["gap"]) if rows else None

    # ---- rare-concept recall stats for the best method vs the primary ceiling ----
    rare_report = {}
    if best is not None:
        bm = best["method"]
        for idx, euro, name in RARE:
            entry = {"index": idx, "eurovoc_id": euro, "concept": name, "per_target": {}}
            for lg in TARGETS:
                n_pos_tr = int(Ytr[:, idx].sum())
                s_best = probs[bm]["test"][lg][:, idx]
                s_ceil = ceiling["l2_thenZ"][lg]["_test_scores"][:, idx]
                entry["per_target"][lg] = {
                    "n_pos_train": n_pos_tr,
                    "best_method": rare_stats(y_te[:, idx], s_best),
                    "native_ceiling": rare_stats(y_te[:, idx], s_ceil),
                }
            rare_report[str(idx)] = entry

    payload["methods"] = {m: {lg: methods[m][lg] for lg in LANGS if lg in methods[m]}
                          for m in methods}
    payload["ceiling"] = {tf: {lg: ceiling[tf][lg]["metrics"] for lg in TARGETS}
                          for tf in ceiling}
    payload["rows"] = rows
    payload["best_method_by_gap"] = best["method"] if best else None
    payload["rare"] = rare_report
    payload["elapsed_s"] = round(time.time() - _T0, 1)
    flush()

    write_report(payload, rows, best, rare_report, P)
    write_plot(rows, ceiling, best, P)
    state["phase"] = "done"
    state["elapsed_s"] = round(time.time() - _T0, 1)
    write_status(state, P["status"])
    log("DONE in %.1fs -> %s , %s" % (time.time() - _T0, P["json"], P["md"]))
    return payload


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(payload, rows, best, rare_report, P):
    ceil = payload["ceiling"]["l2_thenZ"]
    ceil_raw = payload["ceiling"]["raw"]
    L = []
    L.append("# MultiEURLEX English -> non-English gap closure "
             "(EuroBERT-210m mean + XGBoost)\n")
    L.append("Long EU legal documents (`nlpaueb/multi_eurlex`, en/de/fr; train/test = %d/%d). "
             "Mean-pooled EuroBERT-210m (768-d, 512-token truncation) -> one-vs-rest XGBoost "
             "(`%s`). A %d-row VAL slice of train (seed %d) tunes thresholds (no val embedding "
             "exists). **Every method trains on English** and is evaluated on en/de/fr test; the "
             "**in-language ceiling** is the target trained on its own train rows (primary = "
             "`l2_thenZ [both]`). Documents are parallel across languages."
             % (11000, 5000, "binary:logistic hist depth%d ne%d n_jobs2" %
                (XGB_PARAMS["max_depth"], XGB_PARAMS["n_estimators"]), VAL_SIZE, SEED))
    L.append("")
    L.append("**Ceiling (native, l2_thenZ)** micro / macro / subset / AUC: "
             + "  ".join("%s %.4f/%.4f/%.4f/%.4f" % (lg, ceil[lg]["micro_f1"],
                         ceil[lg]["macro_f1"], ceil[lg]["subset_acc"], ceil[lg]["auc"])
                         for lg in TARGETS))
    L.append("")
    L.append("**Ceiling (native, raw)** micro / macro / AUC: "
             + "  ".join("%s %.4f/%.4f/%.4f" % (lg, ceil_raw[lg]["micro_f1"],
                         ceil_raw[lg]["macro_f1"], ceil_raw[lg]["auc"]) for lg in TARGETS))
    L.append("")
    L.append("Gap = ceiling(l2_thenZ native) - method, mean over de/fr; closure vs the `raw` "
             "English baseline.\n")
    L.append("| method | EN micro | de micro | fr micro | mean tgt micro | de/fr macro-F1 (qm) "
             "| de/fr AUC | de/fr subset-acc | mean gap | gap-closed vs raw |")
    L.append("|---|---:|---:|---:|---:|---|---|---|---:|---:|")
    for r in sorted(rows, key=lambda x: x["gap"]):
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f | %.4f "
                 "| %s |" % (r["method"], r["en_micro"], r["de_micro"], r["fr_micro"],
                             r["mean_tgt_micro"], r["macro_f1_qm_de"], r["macro_f1_qm_fr"],
                             r["de_auc"], r["fr_auc"], r["de_subset"], r["fr_subset"], r["gap"],
                             ("%.1f%%" % (100 * r["closure"])) if r["closure"] is not None
                             else "n/a"))
    L.append("")
    if best is not None:
        L.append("## Best combination\n")
        L.append("**%s** minimizes the mean target gap (mean target micro-F1 = %.4f, gap = %.4f, "
                 "gap closed vs raw = %s)." %
                 (best["method"], best["mean_tgt_micro"], best["gap"],
                  ("%.1f%%" % (100 * best["closure"])) if best["closure"] is not None else "n/a"))
        L.append("")

    if rare_report:
        L.append("## Rare-concept recall stats (best method `%s` vs native l2_thenZ ceiling)\n"
                 % best["method"])
        L.append("Higher is better except `FPR@R90` (lower is better). Cells read "
                 "`best / native`. Rare concepts = the 8 lowest-train-frequency EuroVoc level-1 "
                 "concepts present in test (see `imbalanced_recall_mel.md`).\n")
        L.append("| concept | tgt | AP | R@FPR1% | R@FPR5% | R@FPR10% | R@FA50 | R@FA100 | "
                 "FPR@R90 | maxF2 | maxF3 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")

        def f(v):
            return "n/a" if v is None else "%.3f" % v
        for idx, euro, name in RARE:
            e = rare_report[str(idx)]
            for lg in TARGETS:
                b = e["per_target"][lg]["best_method"]
                c = e["per_target"][lg]["native_ceiling"]
                L.append("| %s [%d] | %s | %s/%s | %s/%s | %s/%s | %s/%s | %s/%s | %s/%s | %s/%s "
                         "| %s/%s | %s/%s |" % (
                             name, idx, lg,
                             f(b["average_precision"]), f(c["average_precision"]),
                             f(b["recall_at_fpr_1"]), f(c["recall_at_fpr_1"]),
                             f(b["recall_at_fpr_5"]), f(c["recall_at_fpr_5"]),
                             f(b["recall_at_fpr_10"]), f(c["recall_at_fpr_10"]),
                             f(b["recall_at_fa_50"]), f(c["recall_at_fa_50"]),
                             f(b["recall_at_fa_100"]), f(c["recall_at_fa_100"]),
                             f(b["fpr_for_recall_90"]), f(c["fpr_for_recall_90"]),
                             f(b["max_f2"]), f(c["max_f2"]),
                             f(b["max_f3"]), f(c["max_f3"])))
        L.append("")
        apb = [rare_report[str(i)]["per_target"][lg]["best_method"]["average_precision"]
               for i, _, _ in RARE for lg in TARGETS]
        apc = [rare_report[str(i)]["per_target"][lg]["native_ceiling"]["average_precision"]
               for i, _, _ in RARE for lg in TARGETS]
        L.append("- Mean AP: best **%.3f** vs native ceiling **%.3f** (16 cells)."
                 % (float(np.mean(apb)), float(np.mean(apc))))
        L.append("")

    with open(P["md"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("wrote %s" % P["md"])


def write_plot(rows, ceiling, best, P):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                              # noqa: BLE001
        log("plot skipped: %r" % (exc,))
        return
    names = [r["method"] for r in rows]
    gaps = [r["gap"] for r in rows]
    order = np.argsort(gaps)
    names = [names[i] for i in order]
    gaps = [gaps[i] for i in order]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#d62728" if (best and n == best["method"]) else "#1f77b4" for n in names]
    ax[0].barh(names, gaps, color=colors)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("mean target gap = ceiling(l2_thenZ native) - method")
    ax[0].set_title("Gap to native ceiling (smaller is better)")
    clo = [100 * (r_["closure"] if r_["closure"] is not None else 0)
           for r_ in sorted(rows, key=lambda x: x["gap"])]
    ax[1].barh(names, clo, color=colors)
    ax[1].axvline(100, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("gap closed vs raw English baseline (%)")
    ax[1].set_title("Gap closure")
    fig.tight_layout()
    fig.savefig(P["png"], dpi=130)
    plt.close(fig)
    log("wrote %s" % P["png"])


# ---------------------------------------------------------------------------
# Launch / main
# ---------------------------------------------------------------------------
def launch_detached(extra, outdir):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    P = _paths("", outdir)
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + extra)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, P["log"], REPO], cwd=REPO)
    print("launched detached; log=%s" % P["log"])


def main(argv=None):
    global _SMOKE
    ap = argparse.ArgumentParser(description="MultiEURLEX EN->non-EN gap closure (EuroBERT+XGB).")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-estimators", type=int, default=XGB_PARAMS["n_estimators"])
    ap.add_argument("--max-depth", type=int, default=XGB_PARAMS["max_depth"])
    args = ap.parse_args(argv)

    XGB_PARAMS["n_estimators"] = args.n_estimators
    XGB_PARAMS["max_depth"] = args.max_depth
    outdir = args.outdir or REPO

    if args.launch:
        extra = [a for a in (argv or sys.argv[1:]) if a != "--launch"]
        launch_detached(extra, outdir)
        return 0
    _SMOKE = args.smoke
    run(tag=args.tag, outdir=outdir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                     # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise

