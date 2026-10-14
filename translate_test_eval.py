"""translate_test_eval.py -- translate-test on FLORES sent (12-class).

Question
--------
Train ONE model on **English**.  For a target language, machine-translate its
**TEST** sentences into English, embed the translated English, and feed the SAME
English-trained model.  Does translate-test **recover English-level performance**?  How does it
compare to (a) the **untranslated cross-lingual** number (English model fed the original target
embeddings) and (b) the **in-language ceiling** (a native model trained on the target's own
train+val)?

Data & features
---------------
* FLORES ``sent`` (12-class single-label) via ``train_flores12.prepare_sentence()``  ->
  train = dev (997) / val = first 504 devtest / test = last 505 devtest (seed-0 permutation,
  parallel across languages so the split indices apply row-for-row to every language).
* EuroBERT mean sentence embeddings, aligned to the current 2006-row CSV by
  ``train_flores12.load_sent_embeddings`` (backbone 210m from ``embeddings/``; 610m from
  ``embeddings_610/``).
* Feature transform = **``perlang_z`` in ENGLISH space**: L2-normalize each vector, then z-score
  with the **English-train** mean/std.  The SAME English-train stats are applied to English, to the
  translated-English test, and to the original (untranslated) target test -- so the only thing that
  changes across the three evaluations is the *text*, never the feature space.

Protocol (per backbone; primary = 210m, 610m included)
-----------------------------------------------------
1. ``EN model``: one ``XGBClassifier(objective='multi:softprob', num_class=12, tree_method='hist',
   max_bin=64, device='cpu', n_jobs=2, seed=0, max_depth=8, min_child_weight=5, learning_rate=0.1,
   n_estimators=400)`` trained on EN **train+val** (features = EN-space ``perlang_z``).
   Reference = the EN **test** score (in-language reference at this feature/model).
2. For each target in {it, de, fr}: translate the target **test** text -> English with
   ``Helsinki-NLP/opus-mt-<lang>-en`` (fp16 greedy, batched, cached), embed the translated English
   with EuroBERT-<backbone> **mean**, transform with the EN-train stats, and evaluate the SAME
   EN-trained model -> **translated-test** score.
3. Untranslated cross-lingual: the SAME model + SAME EN-space transform on the ORIGINAL target test
   embeddings.
4. In-language ceiling: a native model trained on the target's own **train+val** (features = the
   target's OWN-train ``perlang_z``), evaluated on the target test.

Outputs (create-only): ``translate_test_eval.py`` / ``.json`` / ``.md`` / ``.png``.
Cache (MT outputs + translated-EN embeddings): ``%TEMP%\\opencode\\translate_test_eval\\``.

Usage
-----
    python translate_test_eval.py                 # foreground (GPU MT+emb, CPU XGB)
    python translate_test_eval.py --launch        # detached hidden run via launch_hidden.vbs
    python translate_test_eval.py --smoke         # tiny plumbing check (no outputs written)
    python translate_test_eval.py --backbones 210m --targets it,de,fr
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T                       # noqa: E402  (data prep + alignment)

OUT_JSON = os.path.join(REPO, "translate_test_eval.json")
OUT_MD = os.path.join(REPO, "translate_test_eval.md")
OUT_PNG = os.path.join(REPO, "translate_test_eval.png")

TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
OPCODE = os.path.join(TEMP, "opencode")
CACHE = os.path.join(OPCODE, "translate_test_eval")
os.makedirs(CACHE, exist_ok=True)
LOG_PATH = os.path.join(CACHE, "translate_test_eval.log")
PID_PATH = os.path.join(CACHE, "translate_test_eval.pid")
STATUS_PATH = os.path.join(CACHE, "translate_test_eval_status.json")

N_CLASSES = 12
TRAIN_LANGS = ["en", "it", "de", "fr"]
TARGETS = ["it", "de", "fr"]

# Backbone registry: id -> (EuroBERT model id, sentence-embedding root, hidden size)
BACKBONES = {
    "210m": {"model": "EuroBERT/EuroBERT-210m", "dir": "embeddings", "hidden": 768},
    "610m": {"model": "EuroBERT/EuroBERT-610m", "dir": "embeddings_610", "hidden": 1152},
}
DEFAULT_BACKBONES = ["210m", "610m"]

# Bilingual target -> English MarianMT models (one per target).
MT_MODELS = {
    "it": "Helsinki-NLP/opus-mt-it-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}

MT_BATCH = 64
MT_MAX_LEN = 512
EMB_BATCH = 64
EMB_MAX_LEN = 512            # plain tokenize to MAX_LEN-1=511, matching extract_embeddings.py

XGB_PARAMS = dict(objective="multi:softprob", num_class=N_CLASSES, tree_method="hist",
                  max_bin=64, device="cpu", n_jobs=2, seed=0, max_depth=8,
                  min_child_weight=5, learning_rate=0.1, n_estimators=400)

_T0 = time.time()
_SMOKE = False
_DEVICE = "cpu"


def log(msg):
    # When launched via launch_hidden.vbs the process stdout/stderr is redirected to LOG_PATH by
    # cmd, so logging to stdout is enough (writing the file here too would duplicate lines).
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def write_status(obj):
    obj = dict(obj)
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % (type(o),))


# ------------------------------------------------------------------ transforms
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_en_z(X_train):
    """EN-space perlang_z: L2-normalize, then mean/std over the English TRAIN rows."""
    Z = l2norm(X_train)
    mu = Z.mean(0)
    sd = Z.std(0)
    sd = np.where(sd < 1e-6, 1e-6, sd)
    return mu, sd


def apply_z(X, mu, sd):
    return np.ascontiguousarray((l2norm(X) - mu) / sd, dtype=np.float32)


# ------------------------------------------------------------------ metrics
def mc_metrics(y_true, probs, C=N_CLASSES):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(probs).argmax(1)
    labels = list(range(C))
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                                   average="macro", labels=labels))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out


# ------------------------------------------------------------------ embeddings (EuroBERT mean)
def load_sent_mean(root, kept, n_source, langs):
    """Load + align EuroBERT mean sentence embeddings for `langs` from `root/mean/emb_<lang>.npy`."""
    out = {}
    for lg in langs:
        path = os.path.join(root, "mean", "emb_%s.npy" % lg)
        arr = np.load(path)
        if arr.shape[0] != n_source:
            raise ValueError("expected %d rows in %s, got %s" % (n_source, path, arr.shape))
        out[lg] = np.asarray(arr[kept], dtype=np.float64)
    return out


_EMB_MODEL = {}


def get_emb_model(model_id):
    if model_id not in _EMB_MODEL:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(_DEVICE).to(torch.float32).eval()
        _EMB_MODEL[model_id] = (tok, model)
        log("  loaded embedder %s (hidden=%d) on %s"
            % (model_id, model.config.hidden_size, _DEVICE))
    return _EMB_MODEL[model_id]


def embed_mean(texts, model_id):
    """Masked mean over the last hidden state, fp32, plain tokenization truncated to 511.

    Exactly the ``mean`` pooling of ``extract_embeddings.py`` (so translated-English vectors live
    in the same space as the stored ``embeddings/mean/emb_*.npy``).  Batches are length-sorted for
    speed and restored to input order (each text is embedded independently -> identical vectors).
    """
    import torch
    tok, model = get_emb_model(model_id)
    n = len(texts)
    order = np.argsort(np.array([len(t) for t in texts], dtype=np.int64), kind="stable")
    sorted_texts = [texts[i] for i in order]
    acc = []
    with torch.inference_mode():
        for i in range(0, n, EMB_BATCH):
            batch = tok(sorted_texts[i:i + EMB_BATCH], padding=True, truncation=True,
                        max_length=EMB_MAX_LEN - 1, return_tensors="pt")
            ids, mask = batch["input_ids"], batch["attention_mask"]
            h = model(input_ids=ids.to(_DEVICE),
                      attention_mask=mask.to(_DEVICE)).last_hidden_state
            m = mask.to(h.device).unsqueeze(-1)
            acc.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    e_sorted = np.concatenate(acc, 0).astype(np.float32)
    out = np.empty_like(e_sorted)
    out[order] = e_sorted
    return out


def ensure_translated_emb(texts, model_id, backbone, lang):
    """Return (translated-English mean embeddings, cache_path); compute + cache if missing."""
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    tag = "translemb_%s_%s_%d_%s" % (backbone, lang, len(texts), h)
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        log("  [cache] translated-EN embeddings %s" % tag)
        return np.load(path), path
    log("  embedding translated-EN %s (%d texts)" % (tag, len(texts)))
    emb = embed_mean(texts, model_id)
    np.save(path, emb)
    return emb, path


# ------------------------------------------------------------------ machine translation
def mt_translate(lang, texts):
    """Translate target `texts` -> English with opus-mt-<lang>-en (fp16 greedy); cache on disk."""
    model_name = MT_MODELS[lang]
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    cpath = os.path.join(CACHE, "mt_%s_%d_%s.json" % (lang, len(texts), h))
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            out = json.load(f)
        log("  [cache] MT %s (%d lines)" % (model_name, len(out)))
        return out
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_num_threads(2)
    log("  [MT load] %s ..." % model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if _DEVICE == "cuda":
        model = model.half()
    model = model.to(_DEVICE).eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(texts), MT_BATCH):
            batch = texts[i:i + MT_BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=MT_MAX_LEN).to(_DEVICE)
            gen = model.generate(**enc, num_beams=1, max_length=MT_MAX_LEN, do_sample=False)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
            done = min(i + MT_BATCH, len(texts))
            if (i // MT_BATCH) % 8 == 0 or done == len(texts):
                log("    MT %s %d/%d (%.1fs)" % (lang, done, len(texts), time.time() - _T0))
    del model
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    return preds


# ------------------------------------------------------------------ one backbone
def run_backbone(bk, sent, y, tr, va, te, targets, smoke=False):
    import xgboost as xgb
    cfg = BACKBONES[bk]
    root = os.path.join(REPO, cfg["dir"])
    log("==== BACKBONE %s (%s, %s) ====" % (bk, cfg["model"], cfg["dir"]))
    n_source = len(sent["git"])
    X = load_sent_mean(root, sent["kept"], n_source, TRAIN_LANGS)
    log("  loaded mean embeddings: " + ", ".join("%s%s" % (lg, X[lg].shape) for lg in TRAIN_LANGS))

    trva = np.concatenate([tr, va])
    # EN-space transform (fit on EN TRAIN rows only).
    mu, sd = fit_en_z(X["en"][tr])
    Xen = apply_z(X["en"], mu, sd)

    # sanity: fresh mean-pool of a few EN test sentences should match the stored npy (EN space raw)
    sanity = {}
    if not smoke:
        k = 16
        fresh = embed_mean([sent["cur"].iloc[int(i)]["en"] for i in te[:k]], cfg["model"])
        diff = float(np.abs(fresh.astype(np.float64) - np.load(
            os.path.join(root, "mean", "emb_en.npy"))[sent["kept"][te[:k]]].astype(np.float64)).max())
        sanity["emb_repro_max_abs_diff"] = diff
        log("  sanity: fresh EN-test mean-pool vs stored npy max abs diff = %.3e" % diff)

    p = dict(XGB_PARAMS)
    if smoke:
        p["n_estimators"] = 40
    clf_en = xgb.XGBClassifier(**p)
    clf_en.fit(Xen[trva], y[trva])

    res = {"backbone": bk, "model": cfg["model"], "emb_dir": cfg["dir"],
           "xgb_params": p, "sanity": sanity, "en_test": mc_metrics(y[te], clf_en.predict_proba(Xen[te])),
           "translated": {}, "untranslated": {}, "ceiling": {}}
    log("  EN test: acc=%.4f macro=%.4f auc=%.4f"
        % (res["en_test"]["accuracy"], res["en_test"]["macro_f1"], res["en_test"]["auc_ovr_macro"]))

    for tg in targets:
        # --- translated-test ---
        src_texts = sent["cur"].iloc[te][tg].astype(str).tolist()
        if smoke:
            src_texts = src_texts[:60]
        hyp = mt_translate(tg, src_texts)
        temb, _tcache = ensure_translated_emb(hyp, cfg["model"], bk, tg)
        Zt = apply_z(temb, mu, sd)
        res["translated"][tg] = mc_metrics(y[te][:len(hyp)] if smoke else y[te],
                                           clf_en.predict_proba(Zt))

        # --- untranslated cross-lingual (same EN-space transform, original target embeddings) ---
        Xtg = apply_z(X[tg], mu, sd)
        res["untranslated"][tg] = mc_metrics(y[te], clf_en.predict_proba(Xtg[te]))

        # --- in-language ceiling (own per-lang_z train+val -> own test) ---
        cmu, csd = fit_en_z(X[tg][tr])
        Cxl = apply_z(X[tg], cmu, csd)
        clf_nat = xgb.XGBClassifier(**p)
        clf_nat.fit(Cxl[trva], y[trva])
        res["ceiling"][tg] = mc_metrics(y[te], clf_nat.predict_proba(Cxl[te]))

        log("  %s: translated acc=%.4f macro=%.4f auc=%.4f | untranslated macro=%.4f | ceiling macro=%.4f"
            % (tg, res["translated"][tg]["accuracy"], res["translated"][tg]["macro_f1"],
               res["translated"][tg]["auc_ovr_macro"], res["untranslated"][tg]["macro_f1"],
               res["ceiling"][tg]["macro_f1"]))
    return res


# ------------------------------------------------------------------ report / plot
def verdict_text(bb):
    en_macro = bb["en_test"]["macro_f1"]
    lines = []
    for tg in bb["translated"]:
        t = bb["translated"][tg]; u = bb["untranslated"][tg]; c = bb["ceiling"][tg]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
                     % (tg, en_macro, t["macro_f1"], t["accuracy"], t["auc_ovr_macro"],
                        u["macro_f1"], u["accuracy"], u["auc_ovr_macro"],
                        c["macro_f1"], c["auc_ovr_macro"]))
    return lines


def plot(payload, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bks = [b for b in payload["backbones"]]
    fig, axes = plt.subplots(1, len(bks), figsize=(6.0 * len(bks), 4.8), squeeze=False)
    labels = ["EN test"] + ["MT %s" % t for t in TARGETS] + ["untr %s" % t for t in TARGETS]
    for ax, bk in zip(axes[0], bks):
        bb = payload["backbones"][bk]
        vals = [bb["en_test"]["macro_f1"]]
        vals += [bb["translated"][t]["macro_f1"] for t in TARGETS]
        vals += [bb["untranslated"][t]["macro_f1"] for t in TARGETS]
        colors = ["#333333"] + ["#1f77b4"] * len(TARGETS) + ["#ff7f0e"] * len(TARGETS)
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors)
        ax.axhline(bb["en_test"]["macro_f1"], color="black", ls="--", lw=1,
                   label="EN-test reference")
        for r, v in zip(bars, vals):
            ax.annotate("%.3f" % v, (r.get_x() + r.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, max(0.7, max(vals) * 1.15))
        ax.set_title("EuroBERT-%s mean, EN-trained XGBoost" % bk, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Translate-test on FLORES sent: macro-F1 (EN test / MT-translated target test / "
                 "untranslated target test)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log("wrote %s" % out_png)


def build_md(payload):
    cfg = payload["config"]
    L = []
    L.append("# FLORES sent translate-test: does MT-to-English recover English-level performance?")
    L.append("")
    L.append("One **English-trained** `XGBClassifier` (EuroBERT mean, **feature space = English-space "
             "`perlang_z`**: L2-normalize + EN-train z-score).  The same model + same EN-train stats "
             "are evaluated on: the **original EN test** (reference), the **MT-translated target test** "
             "(`Helsinki-NLP/opus-mt-<lang>-en`, fp16 greedy), and the **untranslated target test**.  "
             "The **in-language ceiling** (native model on the target's own train+val, own-train "
             "`perlang_z`) is shown for context.")
    L.append("")
    L.append("- Data: FLORES `sent` (12-class), train=dev 997 / val=504 / test=505 (seed-0 parallel "
             "split). XGBoost: `multi:softprob`, hist, max_bin 64, cpu, n_jobs 2, seed 0, depth 8, "
             "mcw 5, lr 0.1, n 400; trained on EN **train+val**.")
    L.append("- MT: greedy (`num_beams=1`), fp16, batch %d, max_length %d, device=%s."
             % (cfg["mt_batch"], cfg["mt_max_len"], cfg["device"]))
    L.append("- Backbones: %s." % ", ".join("%s (%s, %d-d)"
             % (b, payload["backbones"][b]["model"], BACKBONES[b]["hidden"])
             for b in payload["backbones"]))
    L.append("")
    for bk in payload["backbones"]:
        bb = payload["backbones"][bk]
        L.append("## Backbone %s" % bk)
        L.append("")
        L.append("EN-test reference: macro-F1 **%.3f**, acc %.3f, OVR-AUC **%.3f**."
                 % (bb["en_test"]["macro_f1"], bb["en_test"]["accuracy"], bb["en_test"]["auc_ovr_macro"]))
        L.append("")
        L.append("| target | EN-test macro | MT macro | MT acc | MT auc | untr macro | untr acc | "
                 "untr auc | ceiling macro | ceiling auc |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        L.extend(verdict_text(bb))
        L.append("")
        # verdict
        rec = [bb["translated"][t]["macro_f1"] / bb["en_test"]["macro_f1"] for t in bb["translated"]
               if bb["en_test"]["macro_f1"]]
        untr = [bb["untranslated"][t]["macro_f1"] for t in bb["untranslated"]]
        mt = [bb["translated"][t]["macro_f1"] for t in bb["translated"]]
        L.append("- Transformed target/EN macro-F1 ratio (MT): %s -- **MT reaches EN-level "
                 "(ratio >= 0.9) in %d/%d targets**."
                 % (", ".join("%s=%.3f" % (t, bb["translated"][t]["macro_f1"] / bb["en_test"]["macro_f1"])
                              for t in bb["translated"]),
                    sum(1 for r in rec if r >= 0.9), len(rec)))
        L.append("- MT vs untranslated (macro-F1): %s."
                 % ", ".join("%s %.3f->%.3f (%+.3f)"
                             % (t, bb["untranslated"][t]["macro_f1"], bb["translated"][t]["macro_f1"],
                                bb["translated"][t]["macro_f1"] - bb["untranslated"][t]["macro_f1"])
                             for t in bb["translated"]))
        L.append("- 610m/210m note only where present; ratios to the in-language ceiling: %s."
                 % ", ".join("%s MT %.3f / ceil %.3f"
                             % (t, bb["translated"][t]["macro_f1"], bb["ceiling"][t]["macro_f1"])
                             for t in bb["translated"]))
        L.append("")
    L.append("![translate-test macro-F1](translate_test_eval.png)")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ launch / main
def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, LOG_PATH, REPO], cwd=REPO)
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    global _SMOKE, _DEVICE
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="FLORES sent translate-test evaluation.")
    ap.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                    help="comma list from {%s} (default all)" % ",".join(BACKBONES))
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args(argv)

    if args.launch:
        passthrough = []
        if args.backbones != ",".join(DEFAULT_BACKBONES):
            passthrough += ["--backbones", args.backbones]
        if args.targets != ",".join(TARGETS):
            passthrough += ["--targets", args.targets]
        if args.device != "auto":
            passthrough += ["--device", args.device]
        launch_detached(passthrough)
        return 0

    _SMOKE = args.smoke
    bks = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for b in bks:
        if b not in BACKBONES:
            raise SystemExit("unknown backbone %r (choose from %r)" % (b, list(BACKBONES)))
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    import torch
    _DEVICE = args.device
    if _DEVICE == "auto":
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    log("start pid=%d device=%s smoke=%s backbones=%s targets=%s"
        % (os.getpid(), _DEVICE, _SMOKE, bks, targets))
    write_status({"phase": "start", "device": _DEVICE, "backbones": bks, "targets": targets})

    sent = T.prepare_sentence()
    y = np.asarray(sent["y"], dtype=int)
    tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
    log("data: classes=%d train=%d val=%d test=%d kept=%d"
        % (len(sent["classes"]), len(tr), len(va), len(te), len(sent["kept"])))

    payload = {
        "config": {
            "dataset": "flores sent", "n_classes": N_CLASSES, "classes": sent["classes"],
            "splits": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "feature": "perlang_z in ENGLISH space (L2-normalize + EN-train z-score, applied to EN, "
                       "translated-EN and original target)",
            "xgb": XGB_PARAMS, "mt_models": MT_MODELS, "mt_batch": MT_BATCH, "mt_max_len": MT_MAX_LEN,
            "device": _DEVICE, "smoke": _SMOKE, "seed": 0,
            "ceiling_def": "native model on target own train+val (own-train perlang_z) -> target test",
            "python": sys.version.split()[0],
        },
        "mt_samples": {},
        "backbones": {},
    }

    for bk in bks:
        write_status({"phase": "backbone %s" % bk, "device": _DEVICE})
        res = run_backbone(bk, sent, y, tr, va, te, targets, smoke=_SMOKE)
        payload["backbones"][bk] = res

    # one MT sample per target for eyeballing (skipped in smoke: would need the full test set)
    if not _SMOKE:
        for tg in targets:
            src_all = sent["cur"].iloc[te][tg].astype(str).tolist()
            hyp = mt_translate(tg, src_all)
            payload["mt_samples"][tg] = [{"src": s, "mt_en": h}
                                         for s, h in zip(src_all[:3], hyp[:3])]

    if not _SMOKE:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_jd, ensure_ascii=False)
        log("wrote %s" % OUT_JSON)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write(build_md(payload))
        log("wrote %s" % OUT_MD)
        plot(payload, OUT_PNG)
    else:
        log("SMOKE: skipped writing outputs")
    write_status({"phase": "done"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        write_status({"phase": "error"})
        raise
