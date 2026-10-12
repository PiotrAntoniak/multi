"""more_datasets.py -- extend the out-of-the-box cross-lingual proof to more datasets.

Claim under test (identical to proof_mean_alllangs.py, restricted to the ONE headline config)
----------------------------------------------------------------------------------------------
ONE English-trained XGBoost per dataset, fed **EuroBERT-210m mean-pooled** embeddings with a
**model-free** ``perlang_z`` transform (L2-normalize, then per-language train mean/std z-score,
applied to BOTH train and test of that language), works **comparably** across languages:
target-language performance >= 0.9 x English self performance under that single model.  No
machine translation, no translate-train, no target labels for the shared model.  The mild target
sample is *unlabeled*: each target language contributes only its own TRAIN TEXTS through the SAME
encoder to compute that language's feature stats.

For every dataset/target we ALSO train a **native target-trained model** (same encoder/config,
labelled target-train) and report it as a within-language reference (NOT part of the "one English
XGBoost" claim).

Datasets (detected programmatically; nothing is ever written back to them)
--------------------------------------------------------------------------
* ``xnli``  : facebook/xnli, per-language parquet dirs. 3-class (entailment/neutral/contradiction).
              text = premise + " </s> " + hypothesis. en train -> de/fr/es (+en self).
* ``pawsx`` : google-research-datasets/paws-x, dirs en,de,es,fr,ja,ko,zh. BINARY paraphrase.
              text = sentence1 + " </s> " + sentence2. en train -> de/fr/es.
* ``textdetox`` : textdetox/multilingual_toxicity_dataset, single ``default`` config whose SPLIT
              NAMES are the languages (en,ru,uk,de,es,am,zh,ar,hi,it,fr,he,hin,tt,ja); columns
              text/toxic (int64). No predefined split -> stratified 80/20 train/test per language.
              en train -> it/de/fr/es.
* ``sib200`` : Davlan/sib200, per-language TSV dirs (data/<code>/{train,dev,test}.tsv) with columns
              index_id/category/text. Single-label topic. en coordinate ``eng_Latn`` -> deu/fra/spa.
              (Optional dataset, but straightforward via pandas.)
* ``mlsnt``  : EXCLUDED -- ComplexDataLab/MLSNT (``ML_SNS_with_game.parquet``) is Chinese-only
              (column ``full_text``; no English config/column), so an English-trained proof does
              not apply. Recorded as skipped.

Feature config (mean embeddings only)
-------------------------------------
* ``perlang_z`` (HEADLINE) : per-language (each language's OWN train texts) L2 + z-score. Mild
  unlabeled target sample. This is the config named in the task.
* ``global_z``  (diagnostic): English-train L2 + z-score stats, NO target info at all.

Model: XGBClassifier(objective='multi:softprob', num_class=C) for multiclass; 'binary:logistic'
       for binary. Params (fixed, same as the repo proof): tree_method=hist, max_bin=64,
       device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, learning_rate=0.1,
       n_estimators=400. English is the sole shared model language.

Metrics
-------
* multiclass (xnli, sib200): accuracy (argmax), macro-F1 (argmax), OVR-macro AUC.
* binary (pawsx, textdetox): accuracy@0.5, macro-F1@0.5, AUC, recall@FPR{5%,10%},
  FPR@recall=0.90. Class 1 = paraphrase / toxic is the positive class.
Ratio = target / English self, both under the SAME English-trained model (the transfer ratio);
comparability holds when every target ratio >= 0.9. The native model is reported separately, and
native/EN is given too.

Outputs (create-only: these four files only)
--------------------------------------------
``more_datasets.py`` / ``.json`` / ``.md`` / ``more_datasets.png``.
Operational pid/log/status + embedding cache live under %TEMP%\\opencode\\more_datasets.

Usage
-----
    python more_datasets.py --dataset xnli,pawsx,textdetox,sib200   # foreground (may be long)
    python more_datasets.py --dataset all --launch                  # heavy: detached hidden run
    python more_datasets.py --render                               # (re)build md/png from the JSON
    python more_datasets.py --smoke                                # tiny plumbing check
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:  # cp1252 consoles: never let a unicode print crash the run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT_JSON = os.path.join(REPO, "more_datasets.json")
OUT_MD = os.path.join(REPO, "more_datasets.md")
OUT_PNG = os.path.join(REPO, "more_datasets.png")

TMP = os.path.join(os.environ.get("TEMP", REPO), "opencode", "more_datasets")
CACHE = os.path.join(TMP, "emb_cache")
os.makedirs(CACHE, exist_ok=True)
LOG_PATH = os.path.join(TMP, "more_datasets.log")
PID_PATH = os.path.join(TMP, "more_datasets.pid")
STATUS_PATH = os.path.join(TMP, "more_datasets_status.json")

MODEL_ID = "EuroBERT/EuroBERT-210m"
SEED = 0
MAX_TRAIN = 20000     # cap on EN train (and, for fairness, native target train)
MAX_TEST = 5000       # cap on every test set
SMOKE_TRAIN = 300
SMOKE_TEST = 200
TEXTDETOX_TEST_FRAC = 0.20
EMB_BATCH = 64
MAX_LEN = 512

CONFIGS = ["perlang_z", "global_z"]
HEADLINE = {"xnli": "macro_f1", "sib200": "macro_f1", "pawsx": "auc", "textdetox": "auc"}
METRIC_KEYS = {
    "mc": ["accuracy", "macro_f1", "auc"],
    "binary": ["accuracy", "macro_f1", "auc", "recall_at_fpr5", "recall_at_fpr10",
               "fpr_at_recall90"],
}
XGB_COMMON = dict(tree_method="hist", max_bin=64, device="cpu", n_jobs=2, seed=SEED,
                  max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)

DATASET_ORDER = ["xnli", "pawsx", "textdetox", "sib200"]

_T0 = time.time()
_SMOKE = False


def log(msg):
    print("[%8.1fs] %s" % (time.time() - _T0, msg), flush=True)


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    try:
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % (type(o),))


# ---------------------------------------------------------------------------
# Model-free feature transforms (operate on the SAME mean embeddings)
# ---------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_z(Xtrain):
    Z = l2norm(np.asarray(Xtrain, dtype=np.float64))
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd < 1e-6, 1e-6, sd)
    return mu, sd


def apply_z(X, mu, sd):
    return np.ascontiguousarray((l2norm(np.asarray(X, dtype=np.float64)) - mu) / sd,
                                dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding (EuroBERT-210m, plain tokenization, masked mean) -- fp32 GPU
# ---------------------------------------------------------------------------
@np.errstate(all="ignore")
def _embed(texts, tok, model, device):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), EMB_BATCH):
            b = tok(texts[i:i + EMB_BATCH], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
            ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    if not out:
        return np.zeros((0, 768), np.float32)
    return np.concatenate(out, 0).astype(np.float32)


def get_emb(ds_name, lg, split, texts, tok, model, device):
    path = os.path.join(CACHE, "emb_%s_%s_%s_%d.npy" % (ds_name, lg, split, len(texts)))
    if os.path.isfile(path):
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            log("  cache hit %s" % os.path.basename(path))
            return arr
    t0 = time.time()
    arr = _embed(texts, tok, model, device)
    np.save(path, arr)
    log("  embedded %-8s %-5s n=%-6d -> %s (%.1fs)"
        % (ds_name, lg + "/" + split, len(texts), arr.shape, time.time() - t0))
    return arr


# ---------------------------------------------------------------------------
# Dataset loaders (each returns a unified dict; nothing written back)
# ---------------------------------------------------------------------------
def _rng():
    return np.random.RandomState(SEED)


def _sub(n, k, rng):
    return np.sort(rng.permutation(n)[:k])


def _load_xnli(smoke):
    from datasets import load_dataset
    langs = ["en", "de", "fr", "es"]
    tr_cap = SMOKE_TRAIN if smoke else MAX_TRAIN
    te_cap = SMOKE_TEST if smoke else MAX_TEST
    texts, ys = {}, {}
    for lg in langs:
        rng = _rng()
        dtr = load_dataset("facebook/xnli", lg, split="train")
        dte = load_dataset("facebook/xnli", lg, split="test")
        itr = _sub(len(dtr), min(tr_cap, len(dtr)), rng)
        ite = _sub(len(dte), min(te_cap, len(dte)), rng)
        s_tr, s_te = dtr.select(itr), dte.select(ite)
        texts[lg] = {"train": [p + " </s> " + h for p, h in zip(s_tr["premise"], s_tr["hypothesis"])],
                     "test": [p + " </s> " + h for p, h in zip(s_te["premise"], s_te["hypothesis"])]}
        ys[lg] = {"train": np.asarray(s_tr["label"], int), "test": np.asarray(s_te["label"], int)}
    return {"name": "xnli", "kind": "mc", "n_classes": 3,
            "classes": ["entailment", "neutral", "contradiction"],
            "langs": langs, "targets": ["de", "fr", "es"], "texts": texts, "y": ys,
            "spec": "facebook/xnli per-language parquet; text = premise + ' </s> ' + hypothesis; "
                    "label = {0:entailment,1:neutral,2:contradiction}; en train -> de/fr/es."}


def _load_pawsx(smoke):
    from datasets import load_dataset
    langs = ["en", "de", "fr", "es"]
    tr_cap = SMOKE_TRAIN if smoke else MAX_TRAIN
    te_cap = SMOKE_TEST if smoke else MAX_TEST
    texts, ys = {}, {}
    for lg in langs:
        rng = _rng()
        dtr = load_dataset("google-research-datasets/paws-x", lg, split="train")
        dte = load_dataset("google-research-datasets/paws-x", lg, split="test")
        itr = _sub(len(dtr), min(tr_cap, len(dtr)), rng)
        ite = _sub(len(dte), min(te_cap, len(dte)), rng)
        s_tr, s_te = dtr.select(itr), dte.select(ite)
        texts[lg] = {"train": [a + " </s> " + b for a, b in zip(s_tr["sentence1"], s_tr["sentence2"])],
                     "test": [a + " </s> " + b for a, b in zip(s_te["sentence1"], s_te["sentence2"])]}
        ys[lg] = {"train": np.asarray(s_tr["label"], int), "test": np.asarray(s_te["label"], int)}
    return {"name": "pawsx", "kind": "binary", "n_classes": 2, "classes": ["not_paraphrase", "paraphrase"],
            "langs": langs, "targets": ["de", "fr", "es"], "texts": texts, "y": ys,
            "spec": "google-research-datasets/paws-x dirs en,de,es,fr; "
                    "text = sentence1 + ' </s> ' + sentence2; label {0,1}; en train -> de/fr/es."}


def _load_textdetox(smoke):
    from datasets import load_dataset
    langs = ["en", "it", "de", "fr", "es"]
    d = load_dataset("textdetox/multilingual_toxicity_dataset", "default")
    texts, ys = {}, {}
    for lg in langs:
        s = d[lg]                                   # split name == language
        y = np.asarray(s["toxic"], int)
        tx = list(s["text"])
        rng = _rng()
        tr_idx, te_idx = [], []
        for c in np.unique(y):                       # stratified per language
            ci = np.where(y == c)[0]
            rng.shuffle(ci)
            n_te = max(1, int(round(len(ci) * TEXTDETOX_TEST_FRAC)))
            te_idx += list(ci[:n_te]); tr_idx += list(ci[n_te:])
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        rng.shuffle(tr_idx); rng.shuffle(te_idx)
        if smoke:
            tr_idx, te_idx = tr_idx[:SMOKE_TRAIN], te_idx[:SMOKE_TEST]
        else:
            tr_idx, te_idx = tr_idx[:MAX_TRAIN], te_idx[:MAX_TEST]
        texts[lg] = {"train": [tx[i] for i in tr_idx], "test": [tx[i] for i in te_idx]}
        ys[lg] = {"train": y[tr_idx], "test": y[te_idx]}
    return {"name": "textdetox", "kind": "binary", "n_classes": 2, "classes": ["non_toxic", "toxic"],
            "langs": langs, "targets": ["it", "de", "fr", "es"], "texts": texts, "y": ys,
            "spec": "textdetox/multilingual_toxicity_dataset default config; SPLIT NAMES are langs "
                    "(columns text/toxic); stratified 80/20 train/test per language; en -> it/de/fr/es."}


def _load_sib200(smoke):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    codes = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn"}
    frames, cats = {}, set()
    for lg, code in codes.items():
        tr = pd.read_csv(hf_hub_download("Davlan/sib200", "data/%s/train.tsv" % code,
                                         repo_type="dataset"), sep="\t")
        te = pd.read_csv(hf_hub_download("Davlan/sib200", "data/%s/test.tsv" % code,
                                         repo_type="dataset"), sep="\t")
        frames[lg] = (tr, te)
        cats |= set(tr["category"]) | set(te["category"])
    cats = sorted(cats); c2i = {c: i for i, c in enumerate(cats)}
    texts, ys = {}, {}
    for lg, (tr, te) in frames.items():
        texts[lg] = {"train": list(tr["text"].astype(str)), "test": list(te["text"].astype(str))}
        ys[lg] = {"train": np.array([c2i[c] for c in tr["category"]]),
                  "test": np.array([c2i[c] for c in te["category"]])}
    return {"name": "sib200", "kind": "mc", "n_classes": len(cats), "classes": cats,
            "langs": ["en", "de", "fr", "es"], "targets": ["de", "fr", "es"],
            "texts": texts, "y": ys,
            "spec": "Davlan/sib200 per-language TSV (cols index_id/category/text); single-label "
                    "topic; en eng_Latn -> deu/fra/spa Latn; full train/test (701/204)."}


LOADERS = {"xnli": _load_xnli, "pawsx": _load_pawsx, "textdetox": _load_textdetox,
           "sib200": _load_sib200}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _binary_metrics(y, s):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
    y = np.asarray(y, int); s = np.asarray(s, float)
    out = {"accuracy": float(accuracy_score(y, (s >= 0.5).astype(int))),
           "macro_f1": float(f1_score(y, (s >= 0.5).astype(int), average="macro", zero_division=0))}
    try:
        out["auc"] = float(roc_auc_score(y, s))
    except Exception:
        out["auc"] = float("nan")
    try:
        fpr, tpr, _ = roc_curve(y, s)
        out["recall_at_fpr5"] = float(np.interp(0.05, fpr, tpr))
        out["recall_at_fpr10"] = float(np.interp(0.10, fpr, tpr))
        idx = np.where(tpr >= 0.90)[0]
        out["fpr_at_recall90"] = float(fpr[idx[0]]) if len(idx) else float("nan")
    except Exception:
        out.update({"recall_at_fpr5": float("nan"), "recall_at_fpr10": float("nan"),
                    "fpr_at_recall90": float("nan")})
    return out


def _mc_metrics(y, P, C):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y, int); pred = np.asarray(P).argmax(1)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(C)),
                                      zero_division=0))}
    try:
        out["auc"] = float(roc_auc_score(y, P, multi_class="ovr", average="macro",
                                         labels=list(range(C))))
    except Exception:
        out["auc"] = float("nan")
    return out


def _train(ds_kind, X, y, n_classes):
    import xgboost as xgb
    p = dict(XGB_COMMON)
    if _SMOKE:
        p["n_estimators"] = 60
    if ds_kind == "mc":
        p["objective"] = "multi:softprob"; p["num_class"] = n_classes
    else:
        p["objective"] = "binary:logistic"
    clf = xgb.XGBClassifier(**p)
    clf.fit(X, np.asarray(y, int))
    return clf


def _score(ds_kind, clf, X, y, n_classes):
    if ds_kind == "mc":
        return _mc_metrics(y, clf.predict_proba(X), n_classes)
    return _binary_metrics(y, clf.predict_proba(X)[:, 1])


# ---------------------------------------------------------------------------
# One dataset evaluation
# ---------------------------------------------------------------------------
def evaluate(data, tok, model, device):
    name, kind, C = data["name"], data["kind"], data["n_classes"]
    langs, targets = data["langs"], data["targets"]
    # embed everything once (raw mean)
    E = {}
    for lg in langs:
        E[lg] = {}
        for split in ("train", "test"):
            E[lg][split] = get_emb(name, lg, split, data["texts"][lg][split], tok, model, device)
    result = {"name": name, "kind": kind, "n_classes": C, "classes": data["classes"],
              "langs": langs, "targets": targets, "spec": data["spec"], "configs": {},
              "n": {lg: {"train": int(len(data["y"][lg]["train"])),
                         "test": int(len(data["y"][lg]["test"]))} for lg in langs}}
    for config in CONFIGS:
        t0 = time.time()
        if config == "perlang_z":
            stats = {lg: fit_z(E[lg]["train"]) for lg in langs}
        else:
            mu, sd = fit_z(E["en"]["train"]); stats = {lg: (mu, sd) for lg in langs}

        def T(lg, split):
            return apply_z(E[lg][split], *stats[lg])

        model_en = _train(kind, T("en", "train"), data["y"]["en"]["train"], C)
        en_self = _score(kind, model_en, T("en", "test"), data["y"]["en"]["test"], C)
        per_target = {}
        for tg in targets:
            en_trained = _score(kind, model_en, T(tg, "test"), data["y"][tg]["test"], C)
            native = None
            if config == "perlang_z":          # native model uses its own labelled train
                nm = _train(kind, T(tg, "train"), data["y"][tg]["train"], C)
                native = _score(kind, nm, T(tg, "test"), data["y"][tg]["test"], C)
            per_target[tg] = {"en_trained": en_trained, "native": native}
        result["configs"][config] = {"en_self": en_self, "targets": per_target}
        log("%s / %-9s done in %.1fs" % (name, config, time.time() - t0))
        write_status({"phase": "computed", "dataset": name, "config": config})
    return result


# ---------------------------------------------------------------------------
# Summary (target / EN ratios)
# ---------------------------------------------------------------------------
def summarize(result):
    hl = HEADLINE[result["name"]]
    kind = result["kind"]; keys = METRIC_KEYS[kind]
    summary = {"headline": hl, "metrics": {}, "per_target": {}, "comparable": {}}
    for config, blob in result["configs"].items():
        en_self = blob["en_self"]
        m = {}
        for key in keys:
            en = en_self.get(key)
            row = {"en_self": en, "targets": {}}
            for tg, t in blob["targets"].items():
                et = t["en_trained"].get(key)
                nat = (t["native"] or {}).get(key)
                row["targets"][tg] = {
                    "en_trained": et, "native": nat,
                    "ratio": (et / en) if (en and et is not None and en != 0) else None,
                    "native_ratio": (nat / en) if (en and nat is not None and en != 0) else None}
            m[key] = row
        summary["metrics"][config] = m
        # comparability on the headline metric
        ratios = [summary["metrics"][config][hl]["targets"][tg]["ratio"]
                  for tg in result["targets"]]
        finite = [r for r in ratios if r is not None and np.isfinite(r)]
        summary["comparable"][config] = bool(finite) and all(r >= 0.9 for r in finite)
        summary["per_target"][config] = {
            tg: {"en_trained": summary["metrics"][config][hl]["targets"][tg]["en_trained"],
                 "native": summary["metrics"][config][hl]["targets"][tg]["native"],
                 "ratio": summary["metrics"][config][hl]["targets"][tg]["ratio"],
                 "native_ratio": summary["metrics"][config][hl]["targets"][tg]["native_ratio"]}
            for tg in result["targets"]}
    return summary


# ---------------------------------------------------------------------------
# JSON / MD / PNG
# ---------------------------------------------------------------------------
def load_json():
    if os.path.isfile(OUT_JSON):
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"datasets": {}, "meta": {}}


def save_json(doc):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=_json_default, ensure_ascii=False)
    os.replace(tmp, OUT_JSON)


def _fmt(v):
    return "%.3f" % v if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else "n/a"


def render():
    doc = load_json()
    datasets = doc.get("datasets", {})
    order = [d for d in DATASET_ORDER if d in datasets]
    if not order:
        log("render: nothing to render")
        return
    lines = []
    lines.append("# Out-of-the-box cross-lingual proof extended -- EuroBERT-210m mean pooling + "
                 "ONE English XGBoost\n")
    lines.append("Features are **mean-pooled** EuroBERT-210m with a **model-free** `perlang_z` "
                 "transform (L2 + per-language train z-score). English is the sole shared model "
                 "language; every target test set is scored by that one English model. A **native "
                 "target-trained model** is reported alongside as a within-language reference. "
                 "**ratio = target / English self, both under the SAME English model** (the "
                 "transfer ratio); **comparable when ratio >= 0.9**. `global_z` (English-only stats, "
                 "no target sample) is a diagnostic.\n")
    if doc.get("meta", {}).get("skipped"):
        lines.append("Skipped: " + "; ".join(doc["meta"]["skipped"]) + "\n")

    # ---- combined table (dataset x target -> EN vs native + ratio) ----
    lines.append("## Combined table (headline metric per dataset)\n")
    lines.append("| dataset | metric | target | EN on EN | EN on target | native on target | "
                 "ratio (target/EN) | native/EN | comparable |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in order:
        res = datasets[d]; summ = res["summary"]; hl = summ["headline"]
        row = summ["per_target"].get("perlang_z", {})
        for tg in res["targets"]:
            t = row.get(tg, {})
            lines.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s |"
                         % (d, hl, tg,
                            _fmt(summ["metrics"]["perlang_z"][hl]["en_self"]) if "perlang_z" in summ["metrics"] else "n/a",
                            _fmt(t.get("en_trained")), _fmt(t.get("native")),
                            _fmt(t.get("ratio")), _fmt(t.get("native_ratio")),
                            "YES" if (t.get("ratio") or 0) >= 0.9 else "NO"))
    lines.append("")
    lines.append("**Comparability (perlang_z headline >= 0.9 for every target):** " +
                 "; ".join("%s = %s" % (d, "YES" if datasets[d]["summary"]["comparable"].get("perlang_z")
                                        else "NO") for d in order) + "\n")

    # ---- per-dataset detail ----
    for d in order:
        res = datasets[d]; summ = res["summary"]; hl = summ["headline"]; kind = res["kind"]
        lines.append("## %s (%s, %d classes) -- headline `%s`\n"
                     % (d, "binary" if kind == "binary" else "single-label multiclass",
                        res["n_classes"], hl))
        lines.append("spec: %s\n" % res["spec"])
        lines.append("sizes: " + " | ".join("%s tr=%d te=%d" % (lg, res["n"][lg]["train"],
                                                                res["n"][lg]["test"])
                                             for lg in res["langs"]) + "\n")
        for config in CONFIGS:
            if config not in summ["metrics"]:
                continue
            lines.append("**config `%s`**\n" % config)
            hdr = "| metric | EN self | " + " | ".join("%s (EN-model)" % tg for tg in res["targets"]) \
                  + " | " + " | ".join("%s (native)" % tg for tg in res["targets"]) \
                  + " | " + " | ".join("%s/EN" % tg for tg in res["targets"]) + " |"
            lines.append(hdr)
            lines.append("|" + "---|" * (2 + 3 * len(res["targets"])))
            for key in METRIC_KEYS[kind]:
                m = summ["metrics"][config][key]
                cells = [_fmt(m["en_self"])]
                cells += [_fmt(m["targets"][tg]["en_trained"]) for tg in res["targets"]]
                cells += [_fmt(m["targets"][tg]["native"]) for tg in res["targets"]]
                cells += [_fmt(m["targets"][tg]["ratio"]) for tg in res["targets"]]
                lines.append("| `%s` | " % key + " | ".join(cells) + " |")
            lines.append("")
        lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("wrote %s" % OUT_MD)

    # ---- figure: target/EN ratio bars + 0.9 comparability line ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = {"xnli": "#1f77b4", "pawsx": "#d62728", "textdetox": "#2ca02c", "sib200": "#9467bd"}
    labels, transfer, native, colors = [], [], [], []
    for d in order:
        summ = datasets[d]["summary"]
        for tg in datasets[d]["targets"]:
            t = summ["per_target"]["perlang_z"][tg]
            labels.append("%s:%s" % (d, tg))
            transfer.append(t["ratio"])
            native.append(t["native_ratio"])
            colors.append(cmap.get(d, "gray"))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(7, 1.15 * len(labels) + 3), 5.0), squeeze=False)
    for ax, vals, title in [(axes[0][0], transfer, "Transfer ratio: EN model on target / EN self"),
                            (axes[0][1], native, "Native ratio: native model on target / EN self")]:
        b = ax.bar(x, [v if v is not None and np.isfinite(v) else 0.0 for v in vals],
                   color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0.9, color="red", linestyle="--", linewidth=1.2)
        ax.text(-0.45, 0.905, "comparability 0.9", color="red", fontsize=8,
                ha="left", va="bottom")
        for xi, v in zip(x, vals):
            if v is not None and np.isfinite(v):
                ax.text(xi, v + 0.015, "%.2f" % v, ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, max(1.15, (max([v for v in vals if v is not None and np.isfinite(v)],
                                      default=1.0) + 0.15)))
        ax.set_ylabel("ratio vs EN self"); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[d]) for d in order]
    fig.suptitle("EuroBERT-210m mean + one English XGBoost: target/EN ratio per dataset/target "
                 "(`perlang_z`)\n(ratio = target / English self under the SAME English model)",
                 fontsize=12, y=0.995)
    fig.legend(handles, order, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(order), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(OUT_PNG, dpi=130)
    log("wrote %s" % OUT_PNG)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(names):
    doc = {"datasets": {}, "meta": {}} if _SMOKE else load_json()
    doc.setdefault("datasets", {})
    doc.setdefault("meta", {})
    doc["meta"].update({"model": MODEL_ID, "configs": CONFIGS, "headline": HEADLINE,
                        "max_train": MAX_TRAIN, "max_test": MAX_TEST, "seed": SEED,
                        "smoke": _SMOKE, "xgb": XGB_COMMON,
                        "started": doc["meta"].get("started", time.strftime("%Y-%m-%d %H:%M:%S")),
                        "skipped": ["mlsnt: ComplexDataLab/MLSNT is Chinese-only (no English)"]})
    if not _SMOKE:
        with open(PID_PATH, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    # load encoder once
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("loading %s on %s" % (MODEL_ID, device))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).float().eval()

    for name in names:
        if name in doc["datasets"] and not _SMOKE:
            log("%s: already present, skipping" % name)
            continue
        write_status({"phase": "loading", "dataset": name})
        data = LOADERS[name](_SMOKE)
        log("=== %s loaded: %s" % (name, {lg: (len(data["y"][lg]["train"]),
                                               len(data["y"][lg]["test"])) for lg in data["langs"]}))
        result = evaluate(data, tok, model, device)
        result["summary"] = summarize(result)
        doc["datasets"][name] = result
        save_json(doc)                                 # incremental: survive interruption
        render()
        log("%s complete; comparable(perlang_z)=%s"
            % (name, result["summary"]["comparable"]["perlang_z"]))
    doc["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(doc)
    render()
    # one-line verdicts
    for name in names:
        if name in doc["datasets"]:
            s = doc["datasets"][name]["summary"]
            log("VERDICT %-9s perlang_z comparable=%s per_target=%s"
                % (name, s["comparable"]["perlang_z"], {k: round(v["ratio"], 3)
                                                        if v["ratio"] else None
                                                        for k, v in s["per_target"]["perlang_z"].items()}))


def launch(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    script = os.path.join(REPO, os.path.basename(__file__))
    assert " " not in sys.executable and " " not in script, "space in launch path"
    command = "%s %s %s" % (sys.executable, script, " ".join(passthrough))
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, command, LOG_PATH, REPO])
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    global _SMOKE, OUT_JSON, OUT_MD, OUT_PNG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="all | comma list of xnli,pawsx,textdetox,sib200")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args(argv)
    _SMOKE = args.smoke
    if _SMOKE:
        OUT_JSON = os.path.join(TMP, "more_datasets_smoke.json")
        OUT_MD = os.path.join(TMP, "more_datasets_smoke.md")
        OUT_PNG = os.path.join(TMP, "more_datasets_smoke.png")
    if args.render:
        render(); return
    if args.launch:
        passthrough = []
        if args.dataset:
            passthrough += ["--dataset", args.dataset]
        launch(passthrough); return
    sel = [s.strip() for s in args.dataset.split(",") if s.strip()]
    names = DATASET_ORDER if ("all" in sel) else [s for s in DATASET_ORDER if s in sel]
    if not names:
        raise SystemExit("no datasets selected; choose from %r" % (DATASET_ORDER,))
    run(names)


if __name__ == "__main__":
    main()
