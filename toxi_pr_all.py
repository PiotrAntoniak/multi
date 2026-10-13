"""toxi_pr_all.py -- combined precision/recall 2x2 for BOTH backbones.

Regenerates `toxi_pr.png` so that it shows the EuroBERT-210m curves (SOLID) and the
EuroBERT-610m curves (DOTTED) on the same 2x2 grid:

    rows : top = Precision vs threshold ; bottom = Recall vs threshold
    cols : left  = EN model on EN test
           right = EN model on IT test  AND  IT model on IT test

Pipeline (deliberately identical for both backbones, so the comparison is purely about the
embedding backbone):

    * data  = data/toxi_clean/{en,it}_{train,validation,test}.parquet   (label is_toxic)
    * emb   = emb_bf16/toxic/{210m,610m}/emb_<lang>_<split>.npy         (mean pooled, bf16)
    * transform = perlang_z = L2-normalize then per-language z-score, stats fit on the TEST
      language's OWN TRAIN rows (transform language == test language, independent of model).
    * detector  = XGBClassifier(objective=binary:logistic, tree_method=hist, max_bin=64,
      device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, learning_rate=0.1,
      n_estimators=400) -- trained FRESH for both backbones (210m no longer reuses the
      models_toxi/*.ubj files, which were fitted on a different embedding cache).
    * val-tuned threshold = F1-optimal on the model's OWN-language validation split
      (grid 0.00..1.00 step 0.01), same rule as models_toxi/manifest.json.

Outputs (create-only except toxi_pr.png which is overwritten by design):
    toxi_pr_all.json -- all the numbers (per-backbone per-cell metrics + full threshold sweeps)
    toxi_pr.png      -- the combined figure

Constraints: mean pooling only; CPU XGBoost; no GPU needed (embeddings already cached);
no file deleted or moved; no console window.

Usage:  python toxi_pr_all.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO, "data", "toxi_clean")
EMB_ROOT = os.path.join(REPO, "emb_bf16", "toxic")
BB_DIRS = {"210m": os.path.join(EMB_ROOT, "210m"), "610m": os.path.join(EMB_ROOT, "610m")}
BACKBONES = ["210m", "610m"]                    # 210m = solid, 610m = dotted
PAIR = ["en", "it"]
SEEDS = [0]

OUT_JSON = os.path.join(REPO, "toxi_pr_all.json")
OUT_PNG = os.path.join(REPO, "toxi_pr.png")

XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_bin=64,
                  device="cpu", n_jobs=2, seed=0, max_depth=8, min_child_weight=5,
                  learning_rate=0.1, n_estimators=400)

THR_SWEEP = np.round(np.arange(0.0, 1.0000001, 0.01), 2)   # 0.00 .. 1.00 step 0.01
THR_GRID = [0.3, 0.5, 0.7]
RECALL_TARGET = 0.90

# model (colour) x backbone (solid/dotted)
CELL_COLORS = {"en_model_on_en": "#1f77b4", "en_model_on_it": "#d62728",
               "it_model_on_it": "#2ca02c"}
CELL_LABELS = {"en_model_on_en": "EN model / EN test", "en_model_on_it": "EN model / IT test",
               "it_model_on_it": "IT model / IT test"}
BB_STYLE = {"210m": {"ls": "-", "label": "EuroBERT-210m (solid)"},
            "610m": {"ls": ":", "label": "EuroBERT-610m (dotted)"}}

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ------------------------------------------------------------------ transforms
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_perlang_z(X_train):
    Z = l2norm(X_train)
    mu = Z.mean(0)
    sd = Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def apply_perlang_z(X, mu, sd):
    return (l2norm(X) - mu) / sd


# ------------------------------------------------------------------ metrics
def metrics_at(y, s, thr):
    y = np.asarray(y, dtype=np.int64)
    pred = (np.asarray(s, dtype=np.float64) >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"threshold": float(thr), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def auc_at(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, s))


def f1_opt_threshold(y, s):
    from sklearn.metrics import f1_score
    best_t, best_f = 0.5, -1.0
    for t in THR_SWEEP:
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t


def at_recall(y, s, target=RECALL_TARGET):
    """Smallest-threshold operating point whose recall first reaches >= target."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(y[order])
    need = int(np.ceil(target * n_pos - 1e-9))
    idx = int(np.searchsorted(cum, need, side="left"))
    thr = float(s[order][idx]) if idx < len(order) else float(np.min(s))
    return metrics_at(y, s, thr)


def sweep(y, s, grid):
    """precision and recall as a function of threshold t over grid (nan precision if no pred pos)."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    prec = np.empty(len(grid))
    rec = np.empty(len(grid))
    for i, t in enumerate(grid):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        prec[i] = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        rec[i] = tp / n_pos if n_pos > 0 else np.nan
    return prec, rec


# ------------------------------------------------------------------ data + models
def load_raw(emb_dir):
    import pandas as pd
    raw = {}
    for lg in PAIR:
        tr = pd.read_parquet(os.path.join(DATA_DIR, "%s_train.parquet" % lg))
        va = pd.read_parquet(os.path.join(DATA_DIR, "%s_validation.parquet" % lg))
        te = pd.read_parquet(os.path.join(DATA_DIR, "%s_test.parquet" % lg))
        raw[lg] = {}
        for split, df in (("train", tr), ("validation", va), ("test", te)):
            e = np.load(os.path.join(emb_dir, "emb_%s_%s.npy" % (lg, split)))
            assert e.shape[0] == len(df), "%s/%s row mismatch %d vs %d" % (
                emb_dir, split, e.shape[0], len(df))
            raw[lg][split] = (e, df["is_toxic"].astype(int).to_numpy())
    return raw


def train_backbone(emb_dir, name):
    """Train FRESH EN and IT XGBoost detectors on this backbone's embeddings; return cells."""
    import xgboost as xgb
    raw = load_raw(emb_dir)
    zstats = {lg: fit_perlang_z(raw[lg]["train"][0]) for lg in PAIR}

    def feats(lg, split):
        mu, sd = zstats[lg]
        return apply_perlang_z(raw[lg][split][0], mu, sd).astype(np.float32)

    clfs, vthr, meta = {}, {}, {}
    for lg in PAIR:
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(feats(lg, "train"), raw[lg]["train"][1])
        clfs[lg] = clf
        sv = clf.predict_proba(feats(lg, "validation"))[:, 1]
        vthr[lg] = f1_opt_threshold(raw[lg]["validation"][1], sv)
        meta[lg] = {"n_train": int(len(raw[lg]["train"][1])),
                    "n_val": int(len(raw[lg]["validation"][1])),
                    "n_test": int(len(raw[lg]["test"][1])),
                    "n_pos_test": int(raw[lg]["test"][1].sum()),
                    "val_threshold": float(vthr[lg]),
                    "dim": int(raw[lg]["test"][0].shape[1])}
        log("  %s/%s trained (dim=%d) val_f1-opt thr=%.3f"
            % (name, lg, meta[lg]["dim"], vthr[lg]))

    cells = {}
    for ml in PAIR:
        for tl in PAIR:
            key = "%s_model_on_%s" % (ml, tl)
            s = clfs[ml].predict_proba(feats(tl, "test"))[:, 1].astype(np.float64)
            y = raw[tl]["test"][1]
            prec, rec = sweep(y, s, THR_SWEEP)
            cells[key] = {
                "model_lang": ml, "test_lang": tl,
                "n_test": int(len(y)), "n_pos_test": int(y.sum()),
                "auc": auc_at(y, s),
                "val_thr": metrics_at(y, s, vthr[ml]),
                "recall90": at_recall(y, s, RECALL_TARGET),
                "grid": {("%.1f" % t): metrics_at(y, s, t) for t in THR_GRID},
                "curve": {
                    "threshold": [float(t) for t in THR_SWEEP],
                    "precision": [None if np.isnan(v) else round(float(v), 5) for v in prec],
                    "recall": [None if np.isnan(v) else round(float(v), 5) for v in rec],
                },
            }
            log("  %s %-18s auc=%.3f P@val=%.3f R@val=%.3f  P@R90=%.3f"
                % (name, key, cells[key]["auc"], cells[key]["val_thr"]["precision"],
                   cells[key]["val_thr"]["recall"], cells[key]["recall90"]["precision"]))
    return cells, meta


# ------------------------------------------------------------------ plot
def plot(all_cells, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # (row, col) -> (metric, cell names plotted there)
    layout = {
        (0, 0): ("precision", ["en_model_on_en"]),
        (0, 1): ("precision", ["en_model_on_it", "it_model_on_it"]),
        (1, 0): ("recall", ["en_model_on_en"]),
        (1, 1): ("recall", ["en_model_on_it", "it_model_on_it"]),
    }
    col_titles = {
        0: "EN model / EN test",
        1: "EN model / IT test  +  IT model / IT test",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey="row")

    for (r, c), (metric, names) in layout.items():
        ax = axes[r, c]
        for name in names:
            color = CELL_COLORS[name]
            for bb in BACKBONES:
                cell = all_cells[bb][name]
                x = np.array(cell["curve"]["threshold"])
                arr = np.array([np.nan if v is None else v
                                for v in cell["curve"][metric]], dtype=np.float64)
                ax.plot(x, arr, color=color, lw=2.2, alpha=0.95,
                        ls=BB_STYLE[bb]["ls"],
                        marker=("o" if bb == "210m" else "s"), markevery=[], ms=0)
                # val-tuned threshold marker + vertical guide
                vt = cell["val_thr"]["threshold"]
                j = int(np.argmin(np.abs(x - vt)))
                if np.isfinite(arr[j]):
                    ax.plot([vt], [arr[j]], marker=("o" if bb == "210m" else "s"), ms=8,
                            mfc="white", mec=color, mew=1.8, zorder=6)
                ax.axvline(vt, color=color, ls=(":" if bb == "610m" else "-"),
                           lw=1.0, alpha=0.28, zorder=1)
        ax.set_title("%s vs threshold\n%s" % (metric.capitalize(), col_titles[c]), fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)

        handles = [Line2D([], [], color=CELL_COLORS[n], lw=2.2, label=CELL_LABELS[n])
                   for n in names]
        handles += [Line2D([], [], color="black", lw=2.2, ls=BB_STYLE[bb]["ls"],
                           label=BB_STYLE[bb]["label"]) for bb in BACKBONES]
        handles += [Line2D([], [], color="grey", marker="o", ls="none", mfc="white", mec="grey",
                           mew=1.8, ms=8, label="val-tuned threshold")]
        ax.legend(handles=handles, fontsize=9, loc="upper right" if metric == "recall"
                  else "lower right", framealpha=0.92)

    for c in (0, 1):
        axes[1, c].set_xlabel("decision threshold  t")
    for r in (0, 1):
        axes[r, 0].set_ylabel("precision" if r == 0 else "recall")

    fig.suptitle("toxi_pr -- Precision (top) / Recall (bottom) vs decision threshold  --  "
                 "EuroBERT-210m (solid) vs EuroBERT-610m (dotted), mean pooling + perlang_z\n"
                 "left = EN model / EN test;  right = EN model / IT test + IT model / IT test;  "
                 "circles/squares = each model's own val-tuned F1-optimal threshold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------ main
def main():
    log("start pid=%d" % os.getpid())
    all_cells, all_meta = {}, {}
    for bb in BACKBONES:
        log("training backbone %s ..." % bb)
        all_cells[bb], all_meta[bb] = train_backbone(BB_DIRS[bb], bb)

    def num(cell, group, key):
        return cell[group][key]

    payload = {
        "config": {
            "data_dir": "data/toxi_clean",
            "emb_root": "emb_bf16/toxic",
            "backbones": {"210m": "emb_bf16/toxic/210m (solid)", "610m": "emb_bf16/toxic/610m (dotted)"},
            "pooling": "mean", "transform": "perlang_z (L2-normalize then per-language z-score, "
            "stats fit on the TEST language's OWN TRAIN rows)",
            "xgb_params": XGB_PARAMS, "seeds": SEEDS, "threshold_grid": THR_GRID,
            "recall_target": RECALL_TARGET,
            "val_threshold_rule": "F1-optimal on the model's own-language validation split",
            "note": "Both backbones trained FRESH with the identical recipe; 210m curves are solid, "
                    "610m curves are dotted in toxi_pr.png.",
        },
        "splits": all_meta,
        "backbones": {},
        "recall90_summary": {},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # strip bulky per-cell curve into json too (keep -- these are the numbers behind the figure)
    for bb in BACKBONES:
        payload["backbones"][bb] = all_cells[bb]

    # compact recall90 comparison for the cells the task asks about
    for name in ("en_model_on_en", "en_model_on_it", "it_model_on_it"):
        payload["recall90_summary"][name] = {
            bb: {"precision": num(all_cells[bb][name], "recall90", "precision"),
                 "recall": num(all_cells[bb][name], "recall90", "recall"),
                 "threshold": num(all_cells[bb][name], "recall90", "threshold")}
            for bb in BACKBONES}

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("wrote %s" % OUT_JSON)

    plot(all_cells, OUT_PNG)

    log("=" * 70)
    for name in ("en_model_on_en", "en_model_on_it", "it_model_on_it"):
        line = "recall=0.90 precision  %-18s " % name
        line += "  ".join("210m=%.3f" % all_cells["210m"][name]["recall90"]["precision"]
                          if bb == "210m" else "610m=%.3f" % all_cells["610m"][name]["recall90"]["precision"]
                          for bb in BACKBONES)
        # explicit, ordered:
        line = ("recall=0.90 precision  %-18s 210m=%.3f  610m=%.3f"
                % (name, all_cells["210m"][name]["recall90"]["precision"],
                   all_cells["610m"][name]["recall90"]["precision"]))
        log(line)
    log("done")


if __name__ == "__main__":
    main()
