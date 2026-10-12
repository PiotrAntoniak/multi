"""toxi_pr.py -- precision/recall 2x2 matrix (EN/IT model x EN/IT test) on the toxicity data.

Reports, for each cell of the 2x2 (averaged over seeds 0/1/2 when the model exists):
    precision, recall, F1, accuracy, AUC, and confusion counts TP/FP/FN/TN,
at
    (a) the model's OWN val-tuned threshold (from models_toxi/manifest.json) -- primary,
    (b) fixed thresholds 0.3 / 0.5 / 0.7,
    (c) the operating point of recall ~= 0.90 (its precision/F1 + threshold).

Pipeline consistency (EXACTLY the toxi_proof_clean.py convention):
    * EuroBERT-210m masked-MEAN embeddings (fp32) -- reused from the committed cache
      %TEMP%\\opencode\\toxi_clean_cache\\emb_<lang>_<split>.npy (re-embedded on GPU only if missing).
    * transform = perlang_z = L2-normalize then per-language z-score, where the stats are fit on
      the TEST language's OWN TRAIN rows.  So for cell (model, test-lang L) the transform is
      fit_perlang_z(emb_<L>_train) applied to emb_<L>_test -- the transform language is the TEST
      language, independent of which model is evaluated (this is what toxi_proof_clean.py does:
      feats(L, split) = apply_perlang_z(emb[(L, split)], *zstats[L]), zstats[L] = fit on L train).
    * detector = the persisted XGBoost model models_toxi/<lang>_seed<s>.ubj, scored out-of-the-box.
    * val-tuned threshold = manifest per-model val_threshold (F1-optimal on the model's own-lang
      validation split).

CREATE-only: writes toxi_pr.json / toxi_pr.md.  Nothing existing is deleted or moved.

Usage:  python toxi_pr.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
OPCODE = os.path.join(TEMP, "opencode")
CACHE = os.path.join(OPCODE, "toxi_clean_cache")
DATA_DIR = os.path.join(REPO, "data", "toxi_clean")
MODELS_DIR = os.path.join(REPO, "models_toxi")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")

OUT_JSON = os.path.join(REPO, "toxi_pr.json")
OUT_MD = os.path.join(REPO, "toxi_pr.md")
OUT_PNG = os.path.join(REPO, "toxi_pr.png")

THR_SWEEP = np.round(np.arange(0.0, 1.0000001, 0.01), 2)   # 0.00 .. 1.00 step 0.01
# plot colours for the 4 (model, test) cells
CELL_COLORS = {
    "en_model_on_en": "#1f77b4",   # blue   -- EN model on EN test (native)
    "en_model_on_it": "#d62728",   # red    -- EN model on IT test (cross-lingual)
    "it_model_on_en": "#ff7f0e",   # orange -- IT model on EN test (cross-lingual)
    "it_model_on_it": "#2ca02c",   # green  -- IT model on IT test (native)
}
CELL_LABELS = {
    "en_model_on_en": "EN model / EN test",
    "en_model_on_it": "EN model / IT test",
    "it_model_on_en": "IT model / EN test",
    "it_model_on_it": "IT model / IT test",
}

MODEL_ID = "EuroBERT/EuroBERT-210m"
PAIR = ["en", "it"]                 # the 2 languages in the 2x2
SEEDS = [0, 1, 2]
MAX_LEN, BATCH = 512, 64
THR_GRID = [0.3, 0.5, 0.7]
RECALL_TARGET = 0.90

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ------------------------------------------------------------------ transforms (copied verbatim)
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


# ------------------------------------------------------------------ embeddings (reuse cache)
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
    """Masked-mean pool (fp32, length-sorted batching) -- identical to toxi_proof_clean.py."""
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
    log("embedding %s (%d texts) on %s ..." % (tag, len(texts), device))
    emb = embed_mean(texts, device)
    np.save(path, emb)
    log("  %s -> %s" % (tag, emb.shape))
    return emb


# ------------------------------------------------------------------ metrics
def metrics_at(y, s, thr):
    """precision/recall/F1/accuracy + confusion (TP/FP/FN/TN) at a fixed threshold."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    pred = (s >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {"threshold": float(thr), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "accuracy": float(acc),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def auc_at(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, s))


def at_recall(y, s, target=RECALL_TARGET):
    """Operating point with recall as close to `target` as possible from above (>= target).

    Sorts scores descending; finds the smallest number of predicted positives whose cumulative
    TP already reaches ceil(target*n_pos); the threshold is the score at that rank.  The reported
    confusion is recomputed from `s >= thr`.
    """
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    cum = np.cumsum(ys)
    need = int(np.ceil(target * n_pos - 1e-9))
    idx = int(np.searchsorted(cum, need, side="left"))
    thr = float(s[order][idx]) if idx < len(order) else float(np.min(s))
    return metrics_at(y, s, thr)


# ------------------------------------------------------------------ per-cell evaluation
def eval_seed_cell(clf, X, y, val_thr):
    """All operating points for one (model, test) cell at one seed."""
    s = clf.predict_proba(X)[:, 1].astype(np.float64)
    cell = {
        "s": s,
        "val_thr": metrics_at(y, s, val_thr),
        "auc": auc_at(y, s),
        "grid": {("%.1f" % t): metrics_at(y, s, t) for t in THR_GRID},
        "recall90": at_recall(y, s, RECALL_TARGET),
    }
    return cell


def _agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mean": m, "sd": sd, "n": len(v)}


def agg_metric_dict(per_seed_cells, keys):
    """Aggregate a list of metric dicts (one per seed) -> {key: {mean, sd, n}}."""
    out = {}
    for k in keys:
        out[k] = _agg([c.get(k) for c in per_seed_cells])
    return out


PR_KEYS = ["precision", "recall", "f1", "accuracy", "threshold", "TP", "FP", "FN", "TN"]


def agg_cell(per_seed_cell):
    """Aggregate one seed-independent cell -> mean/sd for the val-thr metrics, auc, grid, recall90."""
    return {
        "val_thr": agg_metric_dict([c["val_thr"] for c in per_seed_cell], PR_KEYS),
        "auc": _agg([c["auc"] for c in per_seed_cell]),
        "grid": {t: agg_metric_dict([c["grid"][t] for c in per_seed_cell], PR_KEYS)
                 for t in ("%.1f" % x for x in THR_GRID)},
        "recall90": agg_metric_dict([c["recall90"] for c in per_seed_cell], PR_KEYS),
    }


# ------------------------------------------------------------------ run
def load_models(device):
    import xgboost as xgb
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    by_name = {m["name"]: m for m in manifest["models"]}
    models = {}
    for lg in PAIR:
        models[lg] = []
        for s in SEEDS:
            name = "%s_seed%d" % (lg, s)
            path = os.path.join(MODELS_DIR, "%s_seed%d.ubj" % (lg, s))
            if not os.path.exists(path):
                continue
            clf = xgb.XGBClassifier()
            clf.load_model(path)
            models[lg].append({"seed": s, "clf": clf,
                               "val_thr": float(by_name[name]["val_threshold"])})
    return models, manifest


def run(device):
    import pandas as pd

    models, manifest = load_models(device)

    # data + embeddings + transforms (transform language = TEST language, per toxi_proof_clean.py)
    y = {}
    X = {}
    emb_meta = {}
    for lg in PAIR:
        tr = pd.read_parquet(os.path.join(DATA_DIR, "%s_train.parquet" % lg))
        te = pd.read_parquet(os.path.join(DATA_DIR, "%s_test.parquet" % lg))
        y[lg] = te["is_toxic"].astype(int).to_numpy()
        e_tr = ensure_emb("%s_train" % lg, tr["text"].astype(str).tolist(), device)
        e_te = ensure_emb("%s_test" % lg, te["text"].astype(str).tolist(), device)
        assert e_tr.shape[0] == len(tr) and e_te.shape[0] == len(te), "cache/parquet row mismatch"
        mu, sd = fit_perlang_z(e_tr)
        X[lg] = apply_perlang_z(e_te, mu, sd)
        emb_meta[lg] = {"n_train": len(tr), "n_test": len(te), "n_pos_test": int(y[lg].sum()),
                        "prevalence_test": float(y[lg].mean()), "dim": int(e_te.shape[1])}
        log("%s: n_train=%d n_test=%d n_pos=%d transform=fit_perlang_z(%s_train)"
            % (lg, len(tr), len(te), int(y[lg].sum()), lg))

    cells = {}
    curves = {}
    for model_lg in PAIR:
        for test_lg in PAIR:
            name = "%s_model_on_%s" % (model_lg, test_lg)
            per_seed = []
            for m in models[model_lg]:
                per_seed.append(eval_seed_cell(m["clf"], X[test_lg], y[test_lg], m["val_thr"]))
            cells[name] = agg_cell(per_seed)
            s_mean = np.mean([c["s"] for c in per_seed], axis=0)
            val_thr_mean = float(np.mean([m["val_thr"] for m in models[model_lg]]))
            curves[name] = {"s_mean": s_mean, "y": y[test_lg], "val_thr": val_thr_mean,
                            "model_lg": model_lg, "test_lg": test_lg}
            log("cell %-18s seeds=%d  P=%s R=%s F1=%s AUC=%s"
                % (name, len(per_seed),
                   _f(cells[name]["val_thr"]["precision"]["mean"]),
                   _f(cells[name]["val_thr"]["recall"]["mean"]),
                   _f(cells[name]["val_thr"]["f1"]["mean"]),
                   _f(cells[name]["auc"]["mean"])))

    payload = {
        "config": {
            "data_dir": "data/toxi_clean", "models_dir": "models_toxi",
            "manifest": "models_toxi/manifest.json",
            "emb_model": MODEL_ID, "pooling": "mean", "max_len": MAX_LEN, "batch": BATCH,
            "transform": ("perlang_z (L2-normalize then per-language z-score, stats fit on the "
                          "TEST language's OWN TRAIN rows) -- transform language = test language, "
                          "matching toxi_proof_clean.py feats(L,split)=apply_perlang_z(emb[L,split], "
                          "*zstats[L])"),
            "embedding_cache": os.path.relpath(CACHE, REPO) if CACHE.startswith(REPO)
                               else CACHE,
            "encoding_convention": ("cell (model M, test lang L): X = perlang_z(emb_L_test; "
                                    "fit_perlang_z(emb_L_train)); transform independent of M"),
            "seeds": SEEDS, "threshold_grid": THR_GRID, "recall_target": RECALL_TARGET,
            "primary_threshold": "per-model val_threshold from models_toxi/manifest.json "
                                 "(F1-optimal on the model's own-language validation split)",
            "note": "scores come from the persisted XGBoost models, loaded out-of-the-box.",
        },
        "splits": emb_meta,
        "cells": cells,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("wrote %s" % OUT_JSON)

    md = build_md(payload)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    log("wrote %s" % OUT_MD)

    plot_curves(curves, payload, OUT_PNG)
    return payload


# ------------------------------------------------------------------ curve sweep + plot
def _sweep(y, s, grid):
    """Precision / recall / F1 as a function of threshold t over `grid` (nan precision when no
    positive is predicted at t)."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    prec, rec, f1 = [], [], []
    for t in grid:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = n_pos - tp
        p = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        r = tp / n_pos if n_pos > 0 else np.nan
        f = (2.0 * p * r / (p + r)) if (not np.isnan(p)) and (p + r) > 0 else 0.0
        prec.append(p); rec.append(r); f1.append(f)
    return np.array(prec), np.array(rec), np.array(f1)


def plot_curves(curves, payload, out_png):
    """2x2 figure: rows = Precision / Recall vs threshold; cols = EN model (left) / IT-pair (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["en_model_on_en", "en_model_on_it", "it_model_on_en", "it_model_on_it"]
    swept = {name: _sweep(curves[name]["y"], curves[name]["s_mean"], THR_SWEEP) for name in order}
    arr_idx = {"precision": 0, "recall": 1, "f1": 2}

    # (row, col) -> (metric key, plottable cell names, title)
    layout = {
        (0, 0): ("precision", ["en_model_on_en"],
                 "Precision vs threshold\ntop-left: EN model / EN test"),
        (0, 1): ("precision", ["en_model_on_it", "it_model_on_it"],
                 "Precision vs threshold\ntop-right: EN model / IT test + IT model / IT test"),
        (1, 0): ("recall", ["en_model_on_en"],
                 "Recall vs threshold\nbottom-left: EN model / EN test"),
        (1, 1): ("recall", ["en_model_on_it", "it_model_on_it"],
                 "Recall vs threshold\nbottom-right: EN model / IT test + IT model / IT test"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey="row")

    for (r, c), (key, names, title) in layout.items():
        ax = axes[r, c]
        idx = arr_idx[key]
        for name in names:
            arr = swept[name][idx]
            ax.plot(THR_SWEEP, arr, color=CELL_COLORS[name], lw=2.0, label=CELL_LABELS[name])
            vt = curves[name]["val_thr"]
            j = int(np.argmin(np.abs(THR_SWEEP - vt)))
            yv = arr[j]
            if not np.isnan(yv):
                ax.plot([vt], [yv], marker="o", ms=8, mfc="white", mec=CELL_COLORS[name],
                        mew=1.8, zorder=6)
            ax.axvline(vt, color=CELL_COLORS[name], ls=":", lw=1.0, alpha=0.6)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.legend(fontsize=9.5, loc="upper right" if key == "recall" else "lower right",
                  framealpha=0.92)

    for c in (0, 1):
        axes[1, c].set_xlabel("decision threshold  t")
    for r in (0, 1):
        axes[r, 0].set_ylabel("precision" if r == 0 else "recall")

    fig.suptitle("toxi_pr: Precision (top) / Recall (bottom) vs decision threshold  --  "
                 "mean probabilities over seeds 0/1/2 (EuroBERT-210m mean + perlang_z)\n"
                 "left = EN model on EN test;  right = EN model on IT test + IT model on IT test;  "
                 "circles = model's val-tuned threshold (dotted vlines)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------ report
def _f(v, nd=3):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def _ms(d, nd=3):
    if not d or d.get("mean") is None:
        return "n/a"
    if d.get("n", 1) > 1:
        return "%s+/-%s" % (_f(d["mean"], nd), _f(d["sd"], nd))
    return _f(d["mean"], nd)


def _cnt(d):
    return "n/a" if not d or d.get("mean") is None else "%d" % int(round(d["mean"]))


CELL_ORDER = [("en", "en"), ("en", "it"), ("it", "en"), ("it", "it")]


def _cell(p, model_lg, test_lg):
    return p["cells"]["%s_model_on_%s" % (model_lg, test_lg)]


def build_md(p):
    cfg = p["config"]
    L = []
    L.append("# toxi_pr -- precision / recall 2x2 (EN/IT model x EN/IT test)")
    L.append("")
    L.append("Toxicity precision/recall for the English model and the Italian model on the English "
             "and Italian test splits (`data/toxi_clean/`), using the persisted "
             "`models_toxi/{en,it}_seed{0,1,2}.ubj` XGBoost detectors on **EuroBERT-210m masked-mean** "
             "embeddings.  Primary operating point = each model's **own val-tuned threshold** "
             "(`models_toxi/manifest.json`, F1-optimal on the model's own-language validation split). "
             "Metrics are **mean+/-sd over seeds %s**." % cfg["seeds"])
    L.append("")
    L.append("**Encoding convention (matches `toxi_proof_clean.py`).**  For cell (model M, test "
             "language L) the transform is `perlang_z` fit on **L's own train rows** and applied to "
             "**L's test rows** -- i.e. the transform language is the *test* language and is "
             "independent of which model is scored.  Concretely: `EN model x EN test` and "
             "`IT model x EN test` both use `fit_perlang_z(emb_en_train)` applied to `emb_en_test`; "
             "`EN model x IT test` and `IT model x IT test` both use `fit_perlang_z(emb_it_train)` "
             "applied to `emb_it_test`.  `perlang_z` = L2-normalize then per-language z-score.  "
             "Embeddings reused from the `opencode/toxi_clean_cache` cache dir under the system TEMP "
             "(re-embedded on GPU only if missing).")
    L.append("")
    L.append("**Test-set imbalance (positives).** " + "; ".join(
        "%s test n=%d, n_pos=%d (%.1f%%)" % (lg, p["splits"][lg]["n_test"],
                                             p["splits"][lg]["n_pos_test"],
                                             100.0 * p["splits"][lg]["prevalence_test"])
        for lg in PAIR) + ".")
    L.append("")

    # ---- 2x2 tables at the val-tuned threshold
    L.append("## 2x2 @ val-tuned threshold (mean over seeds)")
    L.append("")
    L.append("Rows = model, columns = test language.  Cells read `metric (mean+/-sd)`.")
    L.append("")

    def two_by_two(title, key):
        L.append("### %s" % title)
        L.append("")
        L.append("| model \\ test | EN test | IT test |")
        L.append("|---|---|---|")
        for m in PAIR:
            if key == "auc":
                c_en = _cell(p, m, "en")["auc"]
                c_it = _cell(p, m, "it")["auc"]
            else:
                c_en = _cell(p, m, "en")["val_thr"][key]
                c_it = _cell(p, m, "it")["val_thr"][key]
            L.append("| %s model | %s | %s |" % (m.upper(), _ms(c_en), _ms(c_it)))
        L.append("")

    two_by_two("Precision", "precision")
    two_by_two("Recall", "recall")
    two_by_two("F1", "f1")
    two_by_two("Accuracy", "accuracy")
    two_by_two("AUC (threshold-independent)", "auc")

    L.append("### Confusion counts @ val-tuned threshold (TP/FP/FN/TN, mean over seeds)")
    L.append("")
    L.append("| cell | TP | FP | FN | TN |")
    L.append("|---|---|---|---|---|")
    for ml, tl in CELL_ORDER:
        c = _cell(p, ml, tl)["val_thr"]
        L.append("| %s model on %s test | %s | %s | %s | %s |"
                 % (ml.upper(), tl.upper(), _cnt(c["TP"]), _cnt(c["FP"]),
                    _cnt(c["FN"]), _cnt(c["TN"])))
    L.append("")
    L.append("### Val-tuned thresholds used (per seed)")
    L.append("")
    L.append("| cell | threshold (mean over seeds) |")
    L.append("|---|---|")
    for ml, tl in CELL_ORDER:
        L.append("| %s model on %s test | %s |"
                 % (ml.upper(), tl.upper(), _ms(_cell(p, ml, tl)["val_thr"]["threshold"])))
    L.append("")

    # ---- fixed-threshold table
    L.append("## Precision / Recall at fixed thresholds 0.3 / 0.5 / 0.7 (mean over seeds)")
    L.append("")
    L.append("Cells read `precision / recall`.")
    L.append("")
    L.append("| cell | thr=0.3 | thr=0.5 | thr=0.7 |")
    L.append("|---|---|---|---|")
    for ml, tl in CELL_ORDER:
        c = _cell(p, ml, tl)["grid"]
        row = ["%s model on %s test" % (ml.upper(), tl.upper())]
        for t in ("0.3", "0.5", "0.7"):
            row.append("%s / %s" % (_ms(c[t]["precision"]), _ms(c[t]["recall"])))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("### F1 at fixed thresholds 0.3 / 0.5 / 0.7 (mean over seeds)")
    L.append("")
    L.append("| cell | thr=0.3 | thr=0.5 | thr=0.7 |")
    L.append("|---|---|---|---|")
    for ml, tl in CELL_ORDER:
        c = _cell(p, ml, tl)["grid"]
        L.append("| %s model on %s test | %s | %s | %s |"
                 % (ml.upper(), tl.upper(), _ms(c["0.3"]["f1"]), _ms(c["0.5"]["f1"]),
                    _ms(c["0.7"]["f1"])))
    L.append("")

    # ---- recall ~= 0.90 operating points
    L.append("## Operating point at recall ~= %.2f (recall-first; mean over seeds)" % RECALL_TARGET)
    L.append("")
    L.append("Smallest-threshold operating point whose recall first reaches the target; precision "
             "at that point.")
    L.append("")
    L.append("| cell | recall | precision | F1 | threshold | TP/FP/FN/TN |")
    L.append("|---|---|---|---|---|---|")
    for ml, tl in CELL_ORDER:
        c = _cell(p, ml, tl)["recall90"]
        L.append("| %s model on %s test | %s | %s | %s | %s | %s/%s/%s/%s |"
                 % (ml.upper(), tl.upper(), _ms(c["recall"]), _ms(c["precision"]), _ms(c["f1"]),
                    _ms(c["threshold"]), _cnt(c["TP"]), _cnt(c["FP"]), _cnt(c["FN"]), _cnt(c["TN"])))
    L.append("")

    # ---- interpretation
    en_en = _cell(p, "en", "en")["recall90"]["precision"]["mean"]
    en_it = _cell(p, "en", "it")["recall90"]["precision"]["mean"]
    it_it = _cell(p, "it", "it")["recall90"]["precision"]["mean"]
    it_en = _cell(p, "it", "en")["recall90"]["precision"]["mean"]
    L.append("## Interpretation")
    L.append("")
    if it_it is not None and en_it is not None:
        if en_it < it_it:
            L.append("- On **Italian** at recall~=0.90, the **EN model** has precision **%s** vs the "
                     "**IT model** **%s** -- the EN model gives up **%.3f precision** for the same "
                     "recall (cross-lingual cost)." % (_f(en_it), _f(it_it), it_it - en_it))
        else:
            L.append("- On **Italian** at recall~=0.90, the **EN model** has precision **%s** vs the "
                     "**IT model** **%s** -- the EN model is at or above native." % (_f(en_it), _f(it_it)))
    if en_en is not None and it_en is not None:
        L.append("- On **English** at recall~=0.90, the **IT model** has precision **%s** vs the "
                 "**EN model** **%s** (reverse-direction cross-lingual cost %s)."
                 % (_f(it_en), _f(en_en),
                    ("%.3f" % (en_en - it_en)) if (en_en is not None and it_en is not None) else "n/a"))
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    dev = args.device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("start device=%s pid=%d" % (dev, os.getpid()))
    run(dev)


if __name__ == "__main__":
    main()
