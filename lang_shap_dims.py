"""Which embedding dimensions carry language, and does the English task model use them?

EuroBERT-210m **mean** embeddings (768-d, float32) only.  Three datasets:

* ``sent`` : FLORES sentence, 12-class topic (``train_flores12.prepare_sentence``).
* ``agg``  : FLORES URL-aggregated, 11-class topic (``train_flores12.prepare_agg``).
* ``mel``  : MultiEURLEX en/de/fr, 21-concept multi-label
             (``multieurlex/emb_mean_{en,de,fr}_{train,test}.npy`` + ``<lang>_{train,test}.jsonl``).

Part A -- WHICH DIMS CARRY LANGUAGE
-----------------------------------
Per dataset, per-dimension one-way ANOVA eta^2 across the languages (en/it/de/fr for FLORES,
en/de/fr for mel), via ``lang_neutralize.eta2_per_dim``.  Report the total eta^2 mass, the max
per-dim eta^2, the number of dims with eta^2 > 0.5 / > 0.25, the k for cumulative
25/50/80/90/95% of the language mass, the top-20 language dims, and a **language linear probe**
(4- or 3-way logistic regression) on the full dims and after zeroing the top-k eta^2 dims
(k = the 50%-mass and 95%-mass counts) -- does the probe collapse?

Part B -- SHAP CONTRIBUTION OF THOSE DIMS
-----------------------------------------
Train the **English** model on the English train split with the fixed spec params
(``tree_method=hist, max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5,
learning_rate=0.1, n_estimators=400``); FLORES = ``multi:softprob`` (12/11-class), mel = 21
one-vs-rest ``binary:logistic``.  Compute ``shap.TreeExplainer`` mean |SHAP| per feature on a
test-row sample.  Report the Spearman correlation between per-dim eta^2 and per-dim mean|SHAP|,
the top-50 overlap, and the fraction of total SHAP mass landing on the top-k language dims
(k = 50%-mass count) vs the random baseline k/d -- do the language dims get MORE or LESS SHAP
than random?

Outputs (create-only): ``lang_shap_dims.json``, ``lang_shap_dims.md`` and one 3-panel plot per
dataset ``lang_shap_dims_{sent,agg,mel}.png`` (per-dim eta^2 sorted; per-dim mean|SHAP| sorted;
eta^2-vs-SHAP scatter with Spearman).  Use ``--launch`` for a hidden detached run (PID + log in
%TEMP%\\opencode) and ``--smoke`` for a tiny plumbing check.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import shap
import xgboost as xgb

from lang_neutralize import eta2_per_dim, top_dims

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T12  # noqa: E402

EMB_DIR_MEL = os.path.join(REPO, "multieurlex")
OUT_JSON = os.path.join(REPO, "lang_shap_dims.json")
OUT_MD = os.path.join(REPO, "lang_shap_dims.md")
TMP = os.path.join(os.environ.get("TEMP", REPO), "opencode")
os.makedirs(TMP, exist_ok=True)

FLORES_LANGS = ["en", "it", "de", "fr"]
MEL_LANGS = ["en", "de", "fr"]
MEL_N_CLASSES = 21
SEED = 0
FRACS = [0.25, 0.50, 0.80, 0.90, 0.95]

# Fixed XGBoost spec params (task Part B).
TREE_COMMON = dict(tree_method="hist", max_bin=64, device="cpu", n_jobs=2, seed=SEED,
                   max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)
FLORES_PARAMS = dict(objective="multi:softprob", **TREE_COMMON)
MEL_PARAMS = dict(objective="binary:logistic", **TREE_COMMON)

SAMPLE_MEL = 1000        # test rows explained for mel
PROBE_CAP_MEL = 3000     # rows per language used for the mel language probe
PROBE_CAP_FLORES = 2000  # rows per language used for the FLORES language probes

_SMOKE = False
_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _f(x):
    return float(x)


# ---------------------------------------------------------------------------
# Part A -- language-carrying dims
# ---------------------------------------------------------------------------
def language_probe(X_by_lang, zero_dims=None, cap=None, seed=SEED):
    """k-way language linear probe (logistic regression) on stacked language embeddings.

    ``zero_dims`` dims are set to 0 before scaling/probing.  ``cap`` subsamples each language
    to at most that many rows (seed) for speed.  Returns held-out (30%) accuracy.
    """
    langs = list(X_by_lang.keys())
    mats, ys = [], []
    for i, l in enumerate(langs):
        A = np.asarray(X_by_lang[l], dtype=np.float64)
        if cap is not None and len(A) > cap:
            idx = np.random.default_rng(seed).choice(len(A), cap, replace=False)
            A = A[np.sort(idx)]
        mats.append(A)
        ys.append(np.full(len(A), i, dtype=int))
    X = np.vstack(mats)
    y = np.concatenate(ys)
    if zero_dims is not None and len(zero_dims):
        X = np.array(X, copy=True)
        X[:, np.asarray(zero_dims, dtype=int)] = 0.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_te = int(round(0.3 * len(y)))
    te, tr = perm[:n_te], perm[n_te:]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=300, C=1.0))
    clf.fit(X[tr], y[tr])
    return float(clf.score(X[te], y[te]))


def part_a(X_by_lang, langs, probe_cap):
    """Per-dim eta^2 table + probe collapse for one dataset (stores ``_e2``/``_order``/``_k``)."""
    mats = {l: np.asarray(X_by_lang[l], dtype=np.float64) for l in langs}
    e2 = eta2_per_dim(mats)
    order, k = {}, {}
    for fr in FRACS:
        o, kk = top_dims(e2, fr)
        order[fr], k[fr] = o, kk
    top20 = [[int(d), _f(e2[d])] for d in order[0.50][:20]]
    out = {
        "langs": list(langs),
        "d": int(e2.shape[0]),
        "total_eta2": _f(e2.sum()),
        "max_eta2": _f(e2.max()),
        "n_eta2_gt_0.5": int((e2 > 0.5).sum()),
        "n_eta2_gt_0.25": int((e2 > 0.25).sum()),
        "k_cum": {("%g" % (100 * fr)): int(k[fr]) for fr in FRACS},
        "top20_dims": top20,
        "_e2": e2, "_order": order, "_k": k,
    }
    k50, k95 = int(k[0.50]), int(k[0.95])
    chance = 1.0 / len(langs)
    acc_full = language_probe(mats, None, probe_cap)
    acc_k50 = language_probe(mats, order[0.50][:k50], probe_cap)
    acc_k95 = language_probe(mats, order[0.95][:k95], probe_cap)
    out["probe"] = {
        "chance": _f(chance),
        "full": _f(acc_full),
        "zero_top_k50": _f(acc_k50), "k50": k50,
        "zero_top_k95": _f(acc_k95), "k95": k95,
        "collapsed_k50": bool(acc_k50 <= chance + 0.05),
        "collapsed_k95": bool(acc_k95 <= chance + 0.05),
    }
    return out


# ---------------------------------------------------------------------------
# Part B -- SHAP
# ---------------------------------------------------------------------------
def mean_abs_shap_multiclass(model, Xs):
    """Per-feature mean |SHAP| for a multi:softprob XGBClassifier, averaged over rows+classes."""
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xs, check_additivity=False))
    if sv.ndim == 3:            # (n, d, n_classes)
        return np.abs(sv).mean(axis=(0, 2))
    return np.abs(sv).mean(axis=0)  # (n, d)


def mean_abs_shap_binary_ovr(models, Xs):
    """Per-feature mean |SHAP| averaged over the one-vs-rest binary concept models."""
    imp = np.zeros(Xs.shape[1], dtype=np.float64)
    for m in models:
        sv = np.asarray(shap.TreeExplainer(m).shap_values(Xs, check_additivity=False))
        if sv.ndim == 3:        # defensive: (n, d, 2)
            sv = sv[:, :, 1]
        imp += np.abs(sv).mean(axis=0)
    return imp / max(1, len(models))


def part_b(shap_imp, e2, order, k, d):
    """SHAP-vs-eta^2 comparison for one dataset."""
    rho, pval = spearmanr(shap_imp, e2)
    top_shap = np.argsort(-shap_imp)
    top50_e2 = list(order[0.50][:50])
    top50_shap = list(top_shap[:50])
    overlap = len(set(int(x) for x in top50_e2) & set(int(x) for x in top50_shap))
    total = float(shap_imp.sum())
    mass = {}
    for fr in (0.25, 0.50, 0.95):
        kf = int(k[fr])
        mass["mass_k%d" % int(round(100 * fr))] = {
            "k": kf,
            "shap_mass": _f(shap_imp[order[fr][:kf]].sum() / total),
            "random": _f(kf / d),
        }
    mk = mass["mass_k50"]
    out = {
        "spearman": _f(rho), "spearman_p": _f(pval),
        "top50_overlap": int(overlap),
        "mean_shap_total": total,
        "top10_shap_dims": [[int(x), _f(shap_imp[x])] for x in top_shap[:10]],
        **mass,
        "k50_lang_dims_more_shap": bool(mk["shap_mass"] > mk["random"]),
    }
    return out, top_shap


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _read_mel_labels(path, n_expect):
    Y = np.zeros((n_expect, MEL_N_CLASSES), dtype=np.int8)
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
    Xtr, Xte = {}, {}
    for lg in MEL_LANGS:
        Xtr[lg] = np.load(os.path.join(EMB_DIR_MEL, "emb_mean_%s_train.npy" % lg)).astype(np.float64)
        Xte[lg] = np.load(os.path.join(EMB_DIR_MEL, "emb_mean_%s_test.npy" % lg)).astype(np.float64)
    ytr = _read_mel_labels(os.path.join(EMB_DIR_MEL, "en_train.jsonl"), len(Xtr["en"]))
    yte = _read_mel_labels(os.path.join(EMB_DIR_MEL, "en_test.jsonl"), len(Xte["en"]))
    return Xtr, Xte, ytr, yte


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_dataset(name, A, imp, B):
    e2 = A["_e2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    e2_sorted = np.sort(e2)[::-1]
    axes[0].plot(np.arange(1, len(e2_sorted) + 1), e2_sorted, lw=1.2)
    axes[0].set_xlabel("dim rank (by eta^2)")
    axes[0].set_ylabel("per-dim eta^2")
    axes[0].set_title("%s: per-dim language eta^2 (sorted)" % name, fontsize=9)
    imp_sorted = np.sort(imp)[::-1]
    axes[1].plot(np.arange(1, len(imp_sorted) + 1), imp_sorted, lw=1.2, color="C1")
    axes[1].set_xlabel("dim rank (by mean |SHAP|)")
    axes[1].set_ylabel("per-dim mean |SHAP|")
    axes[1].set_title("%s: per-dim mean |SHAP| (sorted)" % name, fontsize=9)
    axes[2].scatter(e2, imp, s=6, alpha=0.4)
    axes[2].set_xlabel("per-dim eta^2 (language)")
    axes[2].set_ylabel("per-dim mean |SHAP|")
    axes[2].set_title("%s: eta^2 vs SHAP (Spearman %.3f)" % (name, B["spearman"]), fontsize=9)
    fig.tight_layout()
    path = os.path.join(REPO, "lang_shap_dims_%s.png" % name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log("wrote %s" % path)


# ---------------------------------------------------------------------------
# Per-dataset drivers
# ---------------------------------------------------------------------------
def run_sent(sent, emb_sent):
    X_by_lang = {l: emb_sent[l] for l in FLORES_LANGS}
    A = part_a(X_by_lang, FLORES_LANGS, PROBE_CAP_FLORES)
    log("sent Part A: total eta2=%.1f max=%.3f k50=%d k95=%d n>0.5=%d probe full=%.3f "
        "zero-k50=%.3f zero-k95=%.3f"
        % (A["total_eta2"], A["max_eta2"], A["_k"][0.50], A["_k"][0.95], A["n_eta2_gt_0.5"],
           A["probe"]["full"], A["probe"]["zero_top_k50"], A["probe"]["zero_top_k95"]))
    tr, te = sent["train_idx"], sent["test_idx"]
    if _SMOKE:                       # keep the full train split (all 12 classes must be present)
        te = te[:80]
    params = dict(FLORES_PARAMS); params["num_class"] = 12
    if _SMOKE:
        params["n_estimators"] = 40
    model = xgb.XGBClassifier(**params)
    model.fit(emb_sent["en"][tr].astype(np.float32), sent["y"][tr])
    imp = mean_abs_shap_multiclass(model, emb_sent["en"][te].astype(np.float32))
    B, _ = part_b(imp, A["_e2"], A["_order"], A["_k"], emb_sent["en"].shape[1])
    B.update({"model": "xgb_softprob_12class", "n_train": int(len(tr)), "n_explain": int(len(te))})
    log("sent Part B: spearman=%.3f top50_overlap=%d SHAP-mass k50=%.3f (rand %.3f) %s"
        % (B["spearman"], B["top50_overlap"], B["mass_k50"]["shap_mass"],
           B["mass_k50"]["random"], "MORE" if B["k50_lang_dims_more_shap"] else "LESS"))
    plot_dataset("sent", A, imp, B)
    return A, B


def run_agg(sent, agg, emb_agg):
    X_by_lang = {l: emb_agg[l] for l in FLORES_LANGS}
    A = part_a(X_by_lang, FLORES_LANGS, PROBE_CAP_FLORES)
    log("agg Part A: total eta2=%.1f max=%.3f k50=%d k95=%d n>0.5=%d probe full=%.3f "
        "zero-k50=%.3f zero-k95=%.3f"
        % (A["total_eta2"], A["max_eta2"], A["_k"][0.50], A["_k"][0.95], A["n_eta2_gt_0.5"],
           A["probe"]["full"], A["probe"]["zero_top_k50"], A["probe"]["zero_top_k95"]))
    tr, te = agg["train_idx"], agg["test_idx"]
    if _SMOKE:                       # keep the full train split (all 11 classes must be present)
        te = te[:80]
    params = dict(FLORES_PARAMS); params["num_class"] = agg["n_classes"]
    if _SMOKE:
        params["n_estimators"] = 40
    model = xgb.XGBClassifier(**params)
    model.fit(emb_agg["en"][tr].astype(np.float32), agg["y"][tr])
    imp = mean_abs_shap_multiclass(model, emb_agg["en"][te].astype(np.float32))
    B, _ = part_b(imp, A["_e2"], A["_order"], A["_k"], emb_agg["en"].shape[1])
    B.update({"model": "xgb_softprob_%dclass" % agg["n_classes"],
              "n_train": int(len(tr)), "n_explain": int(len(te))})
    log("agg Part B: spearman=%.3f top50_overlap=%d SHAP-mass k50=%.3f (rand %.3f) %s"
        % (B["spearman"], B["top50_overlap"], B["mass_k50"]["shap_mass"],
           B["mass_k50"]["random"], "MORE" if B["k50_lang_dims_more_shap"] else "LESS"))
    plot_dataset("agg", A, imp, B)
    return A, B


def run_mel():
    Xtr, Xte, ytr_mel, yte_mel = load_mel()
    Xall = {lg: np.vstack([Xtr[lg], Xte[lg]]) for lg in MEL_LANGS}
    A = part_a(Xall, MEL_LANGS, PROBE_CAP_MEL)
    log("mel Part A: total eta2=%.1f max=%.3f k50=%d k95=%d n>0.5=%d probe full=%.3f "
        "zero-k50=%.3f zero-k95=%.3f"
        % (A["total_eta2"], A["max_eta2"], A["_k"][0.50], A["_k"][0.95], A["n_eta2_gt_0.5"],
           A["probe"]["full"], A["probe"]["zero_top_k50"], A["probe"]["zero_top_k95"]))
    Xtr_en = Xtr["en"].astype(np.float32)
    Xte_en = Xte["en"].astype(np.float32)
    ytr = ytr_mel
    if _SMOKE:                       # keep the full train split (rare concepts need positives)
        Xte_en = Xte_en[:120]
    rng = np.random.default_rng(SEED)
    n_s = min(SAMPLE_MEL, len(Xte_en))
    sidx = np.sort(rng.choice(len(Xte_en), n_s, replace=False))
    Xs = Xte_en[sidx]
    mel_params = dict(MEL_PARAMS)
    if _SMOKE:
        mel_params["n_estimators"] = 40
    models = []
    for c in range(MEL_N_CLASSES):
        m = xgb.XGBClassifier(**mel_params)
        m.fit(Xtr_en, ytr[:len(Xtr_en), c].astype(int))
        models.append(m)
    imp = mean_abs_shap_binary_ovr(models, Xs)
    B, _ = part_b(imp, A["_e2"], A["_order"], A["_k"], Xtr["en"].shape[1])
    B.update({"model": "xgb_21x_ovr_binary", "n_train": int(len(Xtr_en)), "n_explain": int(n_s),
              "note": "mean |SHAP| averaged over the 21 concept models"})
    log("mel Part B: spearman=%.3f top50_overlap=%d SHAP-mass k50=%.3f (rand %.3f) %s"
        % (B["spearman"], B["top50_overlap"], B["mass_k50"]["shap_mass"],
           B["mass_k50"]["random"], "MORE" if B["k50_lang_dims_more_shap"] else "LESS"))
    plot_dataset("mel", A, imp, B)
    return A, B


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def strip_private(A):
    return {k: v for k, v in A.items() if not k.startswith("_")}


def write_report(results, elapsed):
    lines = ["# Which dims carry language, and does the English model use them?\n"]
    lines.append("EuroBERT-210m **mean** embeddings (768-d). Part A: per-dim one-way ANOVA eta^2 "
                 "across languages (en/it/de/fr FLORES; en/de/fr MultiEURLEX) + a language linear "
                 "probe before / after zeroing the top eta^2 dims. Part B: fixed English XGBoost "
                 "(`hist`, `max_bin=64`, `cpu`, `n_jobs=2`, `seed=0`, `max_depth=8`, "
                 "`min_child_weight=5`, `lr=0.1`, `n_estimators=400`); per-dim mean |SHAP| "
                 "(`shap.TreeExplainer`) explained on a test-row sample. SHAP-mass k50 = fraction of "
                 "total SHAP landing on the dims that carry 50% of the eta^2 mass; random = k/d.\n")
    lines.append("| dataset | model | d | langs | total eta^2 | max eta^2 | #>0.5 | #>0.25 | "
                 "k25/k50/k80/k90/k95 | probe full | probe -k50 | probe -k95 | Spearman | top-50 "
                 "overlap | SHAP-mass k50 | random | lang dims get |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name in ("sent", "agg", "mel"):
        if name not in results:
            continue
        A = results[name]["part_a"]
        B = results[name]["part_b"]
        kc = A["k_cum"]
        pr = A.get("probe", {})
        lines.append(
            "| %s | %s | %d | %s | %.1f | %.3f | %d | %d | %s | %.3f | %.3f | %.3f | %.3f | "
            "%d/50 | %.3f | %.3f | %s |" % (
                name, B["model"], A["d"], "/".join(A["langs"]), A["total_eta2"], A["max_eta2"],
                A["n_eta2_gt_0.5"], A["n_eta2_gt_0.25"],
                "/".join(str(kc["%g" % (100 * f)]) for f in FRACS),
                pr.get("full", float("nan")), pr.get("zero_top_k50", float("nan")),
                pr.get("zero_top_k95", float("nan")),
                B["spearman"], B["top50_overlap"], B["mass_k50"]["shap_mass"],
                B["mass_k50"]["random"], "MORE" if B["k50_lang_dims_more_shap"] else "LESS"))
    lines.append("")
    lines.append("(probe chance = 1/#langs; `probe -k50/-k95` zero the top-eta^2 dims covering that "
                 "fraction of the language mass.)\n")

    for name in ("sent", "agg", "mel"):
        if name not in results:
            continue
        A = results[name]["part_a"]
        B = results[name]["part_b"]
        lines.append("## %s\n" % name)
        lines.append("Part A -- top-20 language dims by eta^2:\n")
        lines.append("| rank | dim | eta^2 |  | rank | dim | eta^2 |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|")
        t20 = A["top20_dims"]
        for i in range(10):
            a, b = t20[i], t20[i + 10]
            lines.append("| %d | %d | %.4f |  | %d | %d | %.4f |"
                         % (i + 1, a[0], a[1], i + 11, b[0], b[1]))
        lines.append("")
        lines.append("Part B -- top-10 SHAP dims:\n")
        lines.append("| rank | dim | mean |SHAP| |  | rank | dim | mean |SHAP| |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|")
        t10 = B["top10_shap_dims"]
        for i in range(5):
            a, b = t10[i], t10[i + 5]
            lines.append("| %d | %d | %.5f |  | %d | %d | %.5f |"
                         % (i + 1, a[0], a[1], i + 6, b[0], b[1]))
        lines.append("")
        sm = B["mass_k50"]
        lines.append("SHAP mass on the %d language dims (50%%-mass): **%.3f** vs random %.3f -> "
                     "language dims get **%s** SHAP than random.\n"
                     % (sm["k"], sm["shap_mass"], sm["random"],
                        "MORE" if B["k50_lang_dims_more_shap"] else "LESS"))

    lines.append("## Conclusion\n")
    for name in ("sent", "agg", "mel"):
        if name not in results:
            continue
        A = results[name]["part_a"]
        B = results[name]["part_b"]
        conc = "small set" if A["k_cum"]["50"] <= max(30, A["d"] // 20) else "distributed/redundant"
        used = "used by the classifier (SHAP mass >= random)" if B["k50_lang_dims_more_shap"] \
            else "ignored / under-used (SHAP mass <= random)"
        lines.append("- **%s**: language lives in a **%s** of dims (k50=%d, k95=%d of %d; probe "
                     "%.3f -> %.3f after zeroing k50). The task model **%s** (Spearman %.3f, "
                     "top-50 overlap %d/50)."
                     % (name, conc, A["k_cum"]["50"], A["k_cum"]["95"], A["d"],
                        A.get("probe", {}).get("full", float("nan")),
                        A.get("probe", {}).get("zero_top_k50", float("nan")),
                        used, B["spearman"], B["top50_overlap"]))
    lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("wrote %s" % OUT_MD)


# ---------------------------------------------------------------------------
# Launch / main
# ---------------------------------------------------------------------------
def launch_detached(extra):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    logf = os.path.join(TMP, "lang_shap_dims.log")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + extra)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, logf, REPO], cwd=REPO)
    print("launched detached; log=%s" % logf)


def main(argv=None):
    global _SMOKE
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Language dims + SHAP on EuroBERT-210m mean.")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", default="sent,agg,mel")
    args = ap.parse_args(argv)
    if args.launch:
        extra = [a for a in (argv or sys.argv[1:]) if a != "--launch"]
        launch_detached(extra)
        return 0
    _SMOKE = args.smoke
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]

    with open(os.path.join(REPO, "lang_shap_dims.pid"), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    sent = agg = emb_sent = emb_agg = None
    if "sent" in names or "agg" in names:
        sent = T12.prepare_sentence()
        agg = T12.prepare_agg(sent["cur"], sent["classes"])
        emb_sent = T12.load_sent_embeddings("mean", sent["kept"], len(sent["git"]))
        emb_agg = T12.load_agg_embeddings("mean", len(agg["agg"]))

    results = {}
    if "sent" in names:
        A, B = run_sent(sent, emb_sent)
        results["sent"] = {"part_a": A, "part_b": B}
    if "agg" in names:
        A, B = run_agg(sent, agg, emb_agg)
        results["agg"] = {"part_a": A, "part_b": B}
    if "mel" in names:
        A, B = run_mel()
        results["mel"] = {"part_a": A, "part_b": B}

    elapsed = round(time.time() - _T0, 1)
    clean = {n: {"part_a": strip_private(results[n]["part_a"]),
                 "part_b": results[n]["part_b"]} for n in results}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"config": {
            "emb_model": "EuroBERT-210m", "pooling": "mean", "d": 768,
            "flores_langs": FLORES_LANGS, "mel_langs": MEL_LANGS,
            "tree_params": TREE_COMMON, "sample_mel": SAMPLE_MEL,
            "smoke": _SMOKE}, "datasets": clean, "elapsed_s": elapsed}, f, indent=2)
    log("wrote %s" % OUT_JSON)
    write_report(clean, elapsed)
    log("ALL DONE in %.1fs" % elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
