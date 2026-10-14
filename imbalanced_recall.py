"""imbalanced_recall.py -- English-trained vs native-trained detector at HIGH RECALL.

Question.  On HIGHLY UNBALANCED detection tasks (rare positives) where MISSING A POSITIVE IS
UNACCEPTABLE, how much worse is an English-trained model than a model trained natively in the
target language -- when we explicitly TOLERATE many false alarms (recall-oriented evaluation)?

Recipe (identical for every cell; mean pooling only, no chunking):
  * representation : EuroBERT-210m mean pool, already restored as ``emb_<lang>.npy``.
  * transform      : ``l2_thenZ [both]`` -- L2-normalize then per-language TRAIN z-score, applied
                     to English AND target using each language's own train split stats
                     (chunk_pool_mel.fit_l2_thenZ / apply_l2_thenZ).
  * model          : one-vs-rest XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu,
                     n_jobs=2, seed=0, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400).
  * English-trained: fit on the ENGLISH train split (positives = class c), score the TARGET test.
  * native-trained : fit on the TARGET-language train split (positives = class c), score the
                     TARGET test.

Metrics (recall-oriented):
  * Average Precision (PR-AUC).
  * recall at FPR in {1%, 5%, 10%}.
  * recall at #false-alarms in {5, 20, 50, 100} (best recall with at most K false positives).
  * the FPR needed to reach recall = 0.90.
  * max-F2 and max-F3.

Datasets
--------
* sent : FLORES 12-class single-label; rare classes = business and economy, health, geography,
         safety.  Embeddings ``embeddings/mean/emb_<lang>.npy`` + ``train_flores12.prepare_sentence``.
* agg  : FLORES 11-class URL-aggregated (business and economy dropped); rare classes = health,
         geography, safety, education and society.
         ``embeddings_agg/mean/emb_<lang>.npy`` + ``train_flores12.prepare_agg``.
* mel  : MultiEURLEX 21-concept; rare concepts = the 4 lowest-frequency.  The mean embeddings were
         lost, so ``ensure_mel_embeddings`` rebuilds ``multieurlex/emb_mean_<lang>_<split>.npy``
         (EuroBERT-210m mean, 512-token truncation, batch 64) create-only, before training.

Outputs (repo root): ``imbalanced_recall.json``, ``imbalanced_recall.md``,
``imbalanced_pr_<dataset>.png`` (PR curves EN vs native per rare class) plus
``imbalanced_recall.log`` / ``.pid`` / ``_status.json``.

Usage
-----
    python imbalanced_recall.py                     # send + agg (CPU, fast)
    python imbalanced_recall.py --datasets sent,agg,mel
    python imbalanced_recall.py --launch            # detached hidden (for long/mel runs)

SAFETY: this script only CREATES files; it never deletes or moves existing files.
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

from sklearn.metrics import average_precision_score, precision_recall_curve
import xgboost as xgb

# ---------------------------------------------------------------------------- config
REPO = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "it", "de", "fr"]
TARGETS = ["it", "de", "fr"]
EMB_SENT_DIR = os.path.join(REPO, "embeddings")
EMB_AGG_DIR = os.path.join(REPO, "embeddings_agg")
MEL_DIR = os.path.join(REPO, "multieurlex")
MEL_MODEL_ID = "EuroBERT/EuroBERT-210m"
MEL_MAX_LEN, MEL_BATCH = 512, 64

OUT_JSON = os.path.join(REPO, "imbalanced_recall.json")
OUT_MD = os.path.join(REPO, "imbalanced_recall.md")
LOG_PATH = os.path.join(REPO, "imbalanced_recall.log")
PID_PATH = os.path.join(REPO, "imbalanced_recall.pid")
STATUS_PATH = os.path.join(REPO, "imbalanced_recall_status.json")

# Exactly the task-specified detector.
XGB_PARAMS = dict(
    objective="binary:logistic", tree_method="hist", max_bin=64, device="cpu",
    n_jobs=2, seed=0, max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400,
)

FPR_GRID = [0.01, 0.05, 0.10]
FA_GRID = [5, 20, 50, 100]
RECALL_TARGET = 0.90

RARE = {
    "sent": ["business and economy", "health", "geography", "safety"],
    "agg": ["health", "geography", "safety", "education and society"],
    "mel": None,  # auto: 4 lowest-frequency concepts
}

_T0 = time.time()


def log(msg):
    line = "[%8.1fs] %s" % (time.time() - _T0, msg)
    print(line, flush=True)


def write_status(phase, extra=None):
    doc = {"phase": phase, "elapsed_s": round(time.time() - _T0, 1),
           "pid": os.getpid(), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        doc.update(extra)
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
    except Exception as e:                                     # never fatal
        log("WARN: status write failed: %r" % (e,))


# ---------------------------------------------------------------------- transform
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_l2_thenZ(X_train):
    Xn = l2norm(X_train)
    mu = Xn.mean(0)
    sd = Xn.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def apply_l2_thenZ(X, mu, sd):
    return (l2norm(X) - mu) / sd


def l2_thenZ_both(X_by_lang, train_idx):
    """L2-normalize then per-language TRAIN z-score; each language uses its own stats."""
    stats = {lg: fit_l2_thenZ(X_by_lang[lg][train_idx]) for lg in LANGS}
    return {lg: apply_l2_thenZ(X_by_lang[lg], *stats[lg]) for lg in LANGS}


# ------------------------------------------------------------------- recall metrics
def _cum_curves(y, s):
    """Return (n_pos, cum_tp, cum_fp) over items sorted by descending score."""
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")
    ys = np.asarray(y, dtype=np.int64)[order]
    cum_tp = np.cumsum(ys)
    cum_fp = np.cumsum(1 - ys)
    return int(ys.sum()), cum_tp, cum_fp


def recall_at_fp_budget(y, s, budget):
    """Best recall achievable while using at most `budget` false positives."""
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    if n_pos == 0:
        return None
    j = int(np.searchsorted(cum_fp, budget, side="right")) - 1
    if j < 0:
        return 0.0
    return float(cum_tp[j] / n_pos)


def recall_at_fpr(y, s, fpr):
    """Best recall while capping the false-positive RATE at `fpr`."""
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    n_neg = len(y) - n_pos
    if n_neg == 0:
        return None
    return recall_at_fp_budget(y, s, int(np.floor(fpr * n_neg)))


def fpr_for_recall(y, s, target=RECALL_TARGET):
    """FPR needed to reach `target` recall (min false-positive rate attaining it)."""
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    need = int(np.ceil(target * n_pos - 1e-9))
    j = int(np.searchsorted(cum_tp, need, side="left"))
    if j >= len(cum_tp):
        return None
    return float(cum_fp[j] / n_neg)


def max_fbeta(y, s, beta):
    """Maximum F-beta over all score thresholds."""
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    if n_pos == 0:
        return None
    denom = cum_tp + cum_fp
    prec = np.where(denom > 0, cum_tp / np.maximum(denom, 1), 0.0)
    rec = cum_tp / n_pos
    b2 = beta * beta
    d = b2 * prec + rec
    f = np.where(d > 0, (1 + b2) * prec * rec / np.maximum(d, 1e-12), 0.0)
    return float(np.nanmax(f))


def eval_scores(y, s):
    """All recall-oriented metrics for one (scores, labels) cell."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    out = {
        "n_pos_test": n_pos,
        "n_neg_test": n_neg,
        "prevalence_test": (n_pos / len(y)) if len(y) else None,
        "average_precision": float(average_precision_score(y, s)) if n_pos and n_neg else None,
    }
    for p in FPR_GRID:
        out["recall_at_fpr_%d" % int(round(p * 100))] = recall_at_fpr(y, s, p)
    for k in FA_GRID:
        out["recall_at_fa_%d" % k] = recall_at_fp_budget(y, s, k)
    out["fpr_for_recall_90"] = fpr_for_recall(y, s, RECALL_TARGET)
    out["max_f2"] = max_fbeta(y, s, 2)
    out["max_f3"] = max_fbeta(y, s, 3)
    return out


# ------------------------------------------------------------------------- data
def load_sent(emb_mode="mean"):
    import train_flores12 as T
    prep = T.prepare_sentence()
    X = T.load_sent_embeddings(emb_mode, prep["kept"], n_source=2009)
    return {
        "name": "sent", "classes": prep["classes"], "y": prep["y"],
        "train_idx": prep["train_idx"], "test_idx": prep["test_idx"],
        "X": X,
    }


def load_agg(emb_mode="mean"):
    import train_flores12 as T
    sent = T.prepare_sentence()
    prep = T.prepare_agg(sent["cur"], sent["classes"])
    X = T.load_agg_embeddings(emb_mode, n_rows=562)
    return {
        "name": "agg", "classes": prep["classes"], "y": prep["y"],
        "train_idx": prep["train_idx"], "test_idx": prep["test_idx"],
        "X": X,
    }


# ------------------------------------------------------------------------- mel
def _read_jsonl(path):
    texts, Y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            texts.append(r["text"])
            Y.append(r["multi_hot"])
    return texts, np.asarray(Y, dtype=np.int8)


def _mel_split(lang, split):
    """Return (texts, Y[n,21]) from the cached jsonl, else download+cache (create-only)."""
    path = os.path.join(MEL_DIR, "%s_%s.jsonl" % (lang, split))
    if os.path.isfile(path):
        return _read_jsonl(path)
    from datasets import load_dataset
    uri = ("hf://datasets/nlpaueb/multi_eurlex@refs/convert/parquet/"
           "%s/%s/*.parquet" % (lang, split))
    ds = load_dataset("parquet", data_files=uri, split="train")
    texts = list(ds["text"])
    Y = np.zeros((len(texts), 21), dtype=np.int8)
    for i, lab in enumerate(ds["labels"]):
        for c in lab:
            Y[i, int(c)] = 1
    os.makedirs(MEL_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:               # create-only
        for t, y in zip(texts, Y):
            f.write(json.dumps({"text": t, "multi_hot": y.tolist()}) + "\n")
    return texts, Y


def ensure_mel_embeddings(lang, split, device):
    """Create-only mean-pooled embedding cache for one (lang, split)."""
    path = os.path.join(MEL_DIR, "emb_mean_%s_%s.npy" % (lang, split))
    if os.path.isfile(path):
        return path
    import torch
    from transformers import AutoModel, AutoTokenizer
    texts, _ = _mel_split(lang, split)
    tok = AutoTokenizer.from_pretrained(MEL_MODEL_ID)
    model = AutoModel.from_pretrained(MEL_MODEL_ID).to(device).eval()
    hidden = model.config.hidden_size
    pad_id = tok.pad_token_id or 0
    emb = np.empty((len(texts), hidden), dtype=np.float32)
    torch.set_num_threads(2)
    with torch.inference_mode():
        for i in range(0, len(texts), MEL_BATCH):
            batch = texts[i:i + MEL_BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=MEL_MAX_LEN,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb[i:i + len(batch)] = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
            if (i // MEL_BATCH) % 200 == 0:
                log("    mel %s/%s embed %d/%d" % (lang, split, i, len(texts)))
    np.save(path, emb)                                          # create-only
    log("mel %s/%s: wrote %s %s" % (lang, split, os.path.basename(path), emb.shape))
    return path


def load_mel(device):
    """Assemble {lang: {split: emb}} and PER-LANGUAGE concept labels; rebuild caches if missing.

    MultiEURLEX labels are NOT language-independent in the TRAIN split: the training corpora
    differ per language (an EN train document is not the DE/FR one), so the EN train labels do
    NOT describe the DE/FR train documents.  Only the TEST split is parallel (the same 5000
    documents translated, labels row-identical across en/de/fr -- verified).  Labels are
    therefore loaded per ``(lang, split)`` from that language's own jsonl; each language's
    native model fits on its OWN train labels, and the shared test labels score every model.
    """
    X_by_lang = {}
    for lg in LANGS:
        X_by_lang[lg] = {}
        for split in ["train", "test"]:
            p = ensure_mel_embeddings(lg, split, device)
            X_by_lang[lg][split] = np.load(p)
    # Keep rows aligned across languages (all 11000/5000 in practice; defensive min).
    n_tr = min(X_by_lang[lg]["train"].shape[0] for lg in LANGS)
    n_te = min(X_by_lang[lg]["test"].shape[0] for lg in LANGS)
    Y_train_by_lang, Y_test_by_lang = {}, {}
    for lg in LANGS:
        _, Ytr = _mel_split(lg, "train")            # this language's OWN train labels
        _, Yte = _mel_split(lg, "test")             # this language's own (parallel) test labels
        Y_train_by_lang[lg] = Ytr[:n_tr]
        Y_test_by_lang[lg] = Yte[:n_te]
        X_by_lang[lg]["train"] = X_by_lang[lg]["train"][:n_tr]
        X_by_lang[lg]["test"] = X_by_lang[lg]["test"][:n_te]
    # ``Y_train``/``Y_test`` stay the EN labels for backward compatibility (rare-concept
    # selection reference); drivers train natively from the per-language dicts.
    return {
        "name": "mel", "n_concepts": 21,
        "Y_train": Y_train_by_lang["en"], "Y_test": Y_test_by_lang["en"],
        "Y_train_by_lang": Y_train_by_lang, "Y_test_by_lang": Y_test_by_lang,
        "X": X_by_lang,
    }


# ------------------------------------------------------------------------- train/eval
def run_cell(X_train, X_score, train_idx, test_idx, y_pos):
    """Fit OVR XGB on (X_train, y_pos) train rows; return scores on X_score test rows."""
    Xtr = X_train[train_idx]
    ytr = y_pos[train_idx]
    if ytr.sum() == 0:
        return None, y_pos[test_idx]
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(Xtr, ytr)
    scores = clf.predict_proba(X_score[test_idx])[:, 1]
    return scores, y_pos[test_idx]


def evaluate_dataset(ds_name, X_raw, y_class_getter, train_idx, test_idx, classes, rare_classes):
    """Return (results_dict, curve_data).  X_raw: {lang: array}.

    ``y_class_getter(cidx)`` returns the label source for concept ``cidx``: either a single
    one-vs-rest-label array shared by every language (FLORES ``sent``/``agg``, whose rows are
    aligned across languages) or a dict ``{lang: array}`` (MultiEURLEX, whose TRAIN splits are
    NOT parallel across languages, so each language's native model must use its own train
    labels).  In both cases the (shared) test rows carry the same labels.
    """
    X_feat = l2_thenZ_both(X_raw, train_idx)
    results = {}
    curves = {}

    def _ypos(yg, lang):
        arr = yg[lang] if isinstance(yg, dict) else yg
        return (np.asarray(arr) == 1).astype(np.int64)

    for cname in rare_classes:
        cidx = classes.index(cname)
        yg = y_class_getter(cidx)
        y_ref = _ypos(yg, "en")                    # reference labels for the header counts
        n_tr = int(y_ref[train_idx].sum())
        n_te = int(y_ref[test_idx].sum())
        log("%s / %-26s train_pos=%d test_pos=%d" % (ds_name, cname, n_tr, n_te))
        results[cname] = {"class_index": cidx, "n_pos_train": n_tr, "n_pos_test": n_te,
                          "per_target": {}}
        for L in TARGETS:
            # English-trained: fit on EN train (EN labels), score TARGET test (cross-lingual).
            s_en, yte = run_cell(X_feat["en"], X_feat[L], train_idx, test_idx, _ypos(yg, "en"))
            # Native-trained: fit on TARGET train with the TARGET's OWN labels, score TARGET test.
            s_nat, _ = run_cell(X_feat[L], X_feat[L], train_idx, test_idx, _ypos(yg, L))
            m_en = eval_scores(yte, s_en) if s_en is not None else None
            m_nat = eval_scores(yte, s_nat) if s_nat is not None else None
            deg_en = bool(s_en is None or np.std(s_en) == 0.0)
            deg_nat = bool(s_nat is None or np.std(s_nat) == 0.0)
            results[cname]["per_target"][L] = {"en": m_en, "native": m_nat,
                                               "degenerate_en": deg_en,
                                               "degenerate_native": deg_nat}
            curves[(cname, L)] = {
                "y": yte.tolist(),
                "en": s_en.tolist() if s_en is not None else None,
                "native": s_nat.tolist() if s_nat is not None else None,
                "ap_en": None if m_en is None else m_en["average_precision"],
                "ap_native": None if m_nat is None else m_nat["average_precision"],
            }
    return results, curves


# ------------------------------------------------------------------------- plots
def plot_pr(ds_name, results, curves, rare_classes, out_png):
    nrow, ncol = len(rare_classes), len(TARGETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for i, cname in enumerate(rare_classes):
        for j, L in enumerate(TARGETS):
            ax = axes[i][j]
            cd = curves[(cname, L)]
            y = np.asarray(cd["y"])
            for key, color, lbl in (("en", "#1f77b4", "English-trained"),
                                    ("native", "#ff7f0e", "native")):
                s = cd[key]
                ap = cd["ap_en"] if key == "en" else cd["ap_native"]
                if s is None or y.sum() == 0 or y.sum() == len(y):
                    continue
                prec, rec, _ = precision_recall_curve(y, np.asarray(s))
                ax.plot(rec, prec, color=color, lw=1.6,
                        label="%s (AP=%.2f)" % (lbl, ap if ap is not None else float("nan")))
            base = y.mean() if len(y) else 0.0
            ax.axhline(base, color="grey", ls=":", lw=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("%s  |  %s" % (cname, L), fontsize=9)
            ax.set_xlabel("recall", fontsize=8)
            ax.set_ylabel("precision", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("imbalanced_recall: PR curves, English-trained vs native-trained (%s)" % ds_name,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------------- report
METRIC_ORDER = [
    ("average_precision", "AveragePrecision"),
    ("recall_at_fpr_1", "R@FPR1%"),
    ("recall_at_fpr_5", "R@FPR5%"),
    ("recall_at_fpr_10", "R@FPR10%"),
    ("recall_at_fa_5", "R@FA5"),
    ("recall_at_fa_20", "R@FA20"),
    ("recall_at_fa_50", "R@FA50"),
    ("recall_at_fa_100", "R@FA100"),
    ("fpr_for_recall_90", "FPR@R90"),
    ("max_f2", "maxF2"),
    ("max_f3", "maxF3"),
]


def _fmt(v, key):
    if v is None:
        return "n/a"
    if key == "average_precision" or key.startswith("recall") or key.startswith("max_f"):
        return "%.3f" % v
    if key == "fpr_for_recall_90":
        return "%.3f" % v
    return "%.3f" % v


def build_md(payload, margin):
    lines = []
    lines.append("# imbalanced_recall -- English-trained vs native-trained at high recall")
    lines.append("")
    lines.append("Recipe: EuroBERT-210m **mean** pool -> `l2_thenZ [both]` -> OVR "
                 "`XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2, seed=0, "
                 "max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400)`. "
                 "**English-trained** = fit on EN train, score target test; "
                 "**native** = fit on the target-language train, score target test. "
                 "Comparable tolerance (headline): EN within %.3f of native." % margin)
    lines.append("")
    lines.append("Lower is better only for `FPR@R90`; for every other column **higher is better**.")
    lines.append("")

    n_cells = 0
    n_comp_ap = 0
    n_comp_recall = 0
    n_cells_hi = 0
    n_comp_hi = 0
    fa100_en, fa100_nat = [], []
    fpr90_en, fpr90_nat = [], []
    # High-recall regime = the points the task cares about (many false alarms tolerated).
    HL_KEYS = ["recall_at_fpr_10", "recall_at_fa_50", "recall_at_fa_100"]
    for ds_name in payload["datasets"]:
        d = payload["datasets"][ds_name]
        res = d["results"]
        lines.append("## Dataset: `%s`" % ds_name)
        lines.append("")
        if ds_name == "sent":
            lines.append("Rare classes (total counts): business and economy 25, health 68, "
                         "geography 87, safety 97.")
        elif ds_name == "agg":
            lines.append("Rare classes (11-class, business dropped): health, geography, safety, "
                         "education and society.")
        lines.append("")
        for cname in res:
            c = res[cname]
            lines.append("### `%s`  (train_pos=%d, test_pos=%d)" % (cname, c["n_pos_train"],
                                                                    c["n_pos_test"]))
            lines.append("")
            hdr = "| metric | " + " | ".join("%s EN / nat" % L for L in TARGETS) + " |"
            sep = "|" + "---|" * (len(TARGETS) + 1)
            lines.append(hdr)
            lines.append(sep)
            for key, label in METRIC_ORDER:
                row = [label]
                for L in TARGETS:
                    cell = c["per_target"][L]
                    en = cell["en"][key] if cell["en"] else None
                    nat = cell["native"][key] if cell["native"] else None
                    row.append("%s / %s" % (_fmt(en, key), _fmt(nat, key)))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            # per-cell comparability judgement (strict: all points)
            for L in TARGETS:
                cell = c["per_target"][L]
                if not cell["en"] or not cell["native"]:
                    continue
                n_cells += 1
                e_ap, n_ap = cell["en"]["average_precision"], cell["native"]["average_precision"]
                if e_ap is not None and n_ap is not None and e_ap >= n_ap - margin:
                    n_comp_ap += 1
                rec_keys = [k for k, _ in METRIC_ORDER if k.startswith("recall_at")]
                ok = 0
                tot = 0
                for rk in rec_keys:
                    ev, nv = cell["en"][rk], cell["native"][rk]
                    if ev is None or nv is None:
                        continue
                    tot += 1
                    if ev >= nv - margin:
                        ok += 1
                if tot and ok == tot:
                    n_comp_recall += 1
                # high-recall regime
                h_ok = 0
                h_tot = 0
                for rk in HL_KEYS:
                    ev, nv = cell["en"][rk], cell["native"][rk]
                    if ev is None or nv is None:
                        continue
                    h_tot += 1
                    if ev >= nv - margin:
                        h_ok += 1
                if h_tot and h_ok == h_tot:
                    n_comp_hi += 1
                if h_tot:
                    n_cells_hi += 1
                if cell["en"].get("recall_at_fa_100") is not None and \
                        cell["native"].get("recall_at_fa_100") is not None:
                    fa100_en.append(cell["en"]["recall_at_fa_100"])
                    fa100_nat.append(cell["native"]["recall_at_fa_100"])
                if cell["en"].get("fpr_for_recall_90") is not None and \
                        cell["native"].get("fpr_for_recall_90") is not None:
                    fpr90_en.append(cell["en"]["fpr_for_recall_90"])
                    fpr90_nat.append(cell["native"]["fpr_for_recall_90"])
            # high-recall comparison table for this class
            lines.append("High-recall comparison (tolerate many false alarms):")
            lines.append("")
            lines.append("| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |")
            lines.append("|---|---|---|---|")
            for L in TARGETS:
                cell = c["per_target"][L]
                if not cell["en"] or not cell["native"]:
                    continue
                lines.append("| %s | %s / %s | %s / %s | %s / %s |" % (
                    L, _fmt(cell["en"]["recall_at_fa_50"], "recall_at_fa_50"),
                    _fmt(cell["native"]["recall_at_fa_50"], "recall_at_fa_50"),
                    _fmt(cell["en"]["recall_at_fa_100"], "recall_at_fa_100"),
                    _fmt(cell["native"]["recall_at_fa_100"], "recall_at_fa_100"),
                    _fmt(cell["en"]["fpr_for_recall_90"], "fpr_for_recall_90"),
                    _fmt(cell["native"]["fpr_for_recall_90"], "fpr_for_recall_90")))
            lines.append("")

    # headline summary
    lines.append("## Headline")
    lines.append("")
    frac_ap = (n_comp_ap / n_cells) if n_cells else 0.0
    frac_rec = (n_comp_recall / n_cells) if n_cells else 0.0
    frac_hi = (n_comp_hi / n_cells_hi) if n_cells_hi else 0.0
    lines.append("- Cells evaluated (rare-class x target-language): **%d**." % n_cells)
    lines.append("- English-trained Average Precision within %.2f of native in **%d/%d** cells "
                 "(%.0f%%)." % (margin, n_comp_ap, n_cells, 100 * frac_ap))
    lines.append("- **High-recall regime** (R@FPR10pct + R@FA50 + R@FA100 all within %.2f of "
                 "native): **%d/%d** cells (%.0f%%).  This is the regime the task cares about -- "
                 "many false alarms tolerated."
                 % (margin, n_comp_hi, n_cells_hi, 100 * frac_hi))
    lines.append("- English-trained recall (ALL operating points) within %.2f of native in "
                 "**%d/%d** cells (%.0f%%) -- strict, and dominated by the low-FPR points where a "
                 "handful of positives makes %.2f unattainable." %
                 (margin, n_comp_recall, n_cells, 100 * frac_rec, margin))
    lines.append("")
    lines.append("### False-alarm cost at high recall")
    lines.append("")
    if fa100_en:
        lines.append("- Mean recall at 100 false alarms: EN **%.3f** vs native **%.3f** "
                     "(across %d cells)." % (float(np.mean(fa100_en)), float(np.mean(fa100_nat)),
                                             len(fa100_en)))
    if fpr90_en:
        lines.append("- Mean FPR needed to reach recall=0.90: EN **%.3f** vs native **%.3f** "
                     "(across %d cells).  A smaller number is better; the difference is the extra "
                     "false-alarm rate the English model spends for the same 90%% recall."
                     % (float(np.mean(fpr90_en)), float(np.mean(fpr90_nat)), len(fpr90_en)))
    lines.append("")
    lines.append("Read the `R@FA*` and `FPR@R90` columns: they state how many false alarms the "
                 "detector spends to reach high recall. Under the 'missing a positive is "
                 "unacceptable' rule, English-trained and native-tracked curves are compared at "
                 "the SAME recall, so a small EN penalty in `FPR@R90` is the extra false-alarm "
                 "cost of using the English model on target text.")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main
def run(datasets, emb_mode="mean", margin=0.05, mel_device="cpu"):
    payload = {"config": {
        "langs": LANGS, "targets": TARGETS, "xgb_params": XGB_PARAMS,
        "fpr_grid": FPR_GRID, "fa_grid": FA_GRID, "recall_target": RECALL_TARGET,
        "transform": "l2_thenZ [both]", "emb_mode": emb_mode, "margin": margin,
        "rare_classes": RARE,
    }, "datasets": {}}

    if "sent" in datasets:
        write_status("load sent")
        d = load_sent(emb_mode)
        log("sent loaded X=%s train=%d test=%d" % (d["X"]["en"].shape, len(d["train_idx"]),
                                                    len(d["test_idx"])))
        write_status("eval sent")
        res, curves = evaluate_dataset("sent", d["X"], lambda c: null_class_y(d, c),
                                       d["train_idx"], d["test_idx"], d["classes"], RARE["sent"])
        payload["datasets"]["sent"] = {"results": res, "classes": d["classes"]}
        plot_pr("sent", res, curves, RARE["sent"], os.path.join(REPO, "imbalanced_pr_sent.png"))

    if "agg" in datasets:
        write_status("load agg")
        d = load_agg(emb_mode)
        log("agg loaded X=%s train=%d test=%d" % (d["X"]["en"].shape, len(d["train_idx"]),
                                                   len(d["test_idx"])))
        write_status("eval agg")
        res, curves = evaluate_dataset("agg", d["X"], lambda c: null_class_y(d, c),
                                       d["train_idx"], d["test_idx"], d["classes"], RARE["agg"])
        payload["datasets"]["agg"] = {"results": res, "classes": d["classes"]}
        plot_pr("agg", res, curves, RARE["agg"], os.path.join(REPO, "imbalanced_pr_agg.png"))

    if "mel" in datasets:
        write_status("load mel")
        m = load_mel(mel_device)
        # rare concepts = 4 lowest TRAIN frequency among concepts present in test
        freq = m["Y_train"].sum(0)
        test_present = np.where(m["Y_test"].sum(0) > 0)[0]
        order = test_present[np.argsort(freq[test_present])]
        rare_idx = list(order[:4])
        log("mel rare concepts = %s (train freq %s)" % (rare_idx, [int(freq[i]) for i in rare_idx]))
        X_all = {lg: np.concatenate([m["X"][lg]["train"], m["X"][lg]["test"]], axis=0) for lg in LANGS}
        ntr = m["Y_train"].shape[0]
        train_idx = np.arange(ntr)
        test_idx = np.arange(ntr, ntr + m["Y_test"].shape[0])
        # Per-language labels: native models use each language's OWN train labels (mel train
        # splits are not parallel); test labels are shared/parallel.
        Y_all = {lg: np.concatenate([m["Y_train_by_lang"][lg], m["Y_test_by_lang"][lg]], axis=0)
                 for lg in LANGS}
        write_status("eval mel")
        res, curves = evaluate_dataset("mel", X_all,
                                       lambda c: {lg: Y_all[lg][:, c] for lg in LANGS},
                                       train_idx, test_idx,
                                       [str(i) for i in range(21)], [str(i) for i in rare_idx])
        payload["datasets"]["mel"] = {"results": res, "n_concepts": 21, "rare_indices": rare_idx}
        plot_pr("mel", res, curves, [str(i) for i in rare_idx],
                os.path.join(REPO, "imbalanced_pr_mel.png"))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log("wrote %s" % OUT_JSON)

    # MD (drop curve arrays from the md builder's view via payload only)
    md = build_md(payload, margin)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    log("wrote %s" % OUT_MD)
    write_status("done", {"datasets": list(payload["datasets"].keys())})
    return payload


def null_class_y(d, cidx):
    """One-vs-rest indicator for class index cidx."""
    y = d["y"]
    return (y == cidx).astype(np.int64)


def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd], cwd=REPO)


def main():
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="sent,agg",
                    help="comma list of {sent,agg,mel} (default sent,agg)")
    ap.add_argument("--emb-mode", default="mean", help="embedding mode subdir (default mean)")
    ap.add_argument("--margin", type=float, default=0.05, help="comparable tolerance (default 0.05)")
    ap.add_argument("--mel-device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--launch", action="store_true", help="relaunch detached via launch_hidden.vbs")
    args = ap.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    if args.launch:
        passthrough = ["--datasets", args.datasets, "--emb-mode", args.emb_mode,
                       "--margin", str(args.margin), "--mel-device", args.mel_device]
        logpath = os.path.join(os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp"),
                               "opencode", "imbalanced_recall.log")
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        vbs = os.path.join(REPO, "launch_hidden.vbs")
        cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
        subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, logpath, REPO], cwd=REPO)
        print("launched detached; log=%s" % logpath)
        return

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    dev = args.mel_device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("start datasets=%s emb_mode=%s mel_device=%s pid=%d" % (datasets, args.emb_mode, dev,
                                                                os.getpid()))
    write_status("start")
    run(datasets, emb_mode=args.emb_mode, margin=args.margin, mel_device=dev)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        write_status("error")
        raise
