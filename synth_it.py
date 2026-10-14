"""synth_it.py -- train a FLORES-12 topic classifier on LLM-GENERATED Italian,
evaluate on the REAL Italian FLORES test.

Question
--------
The FinePhrase recipe (arXiv:2604.13977 / github huggingface/finephrase) shows a
small ~1B-class instruct generator + a clear rewrite prompt is the sweet spot for
synthetic *training* text.  Here we test the recipe end to end on FLORES-12:
generate Italian training text from the ENGLISH train(+val) sentences with a small
instruct LLM, embed the synthetic Italian with EuroBERT mean, train an XGBoost
topic classifier on it, and evaluate on the **real Italian** FLORES test.

Does training on LLM-generated Italian approach the real-Italian ceiling?

Data & features
---------------
* FLORES ``sent`` (12-class single-label) via ``train_flores12.prepare_sentence()``:
  train = dev (997) / val = first 504 of the seed-0 permuted devtest / test = last 505.
* English source = the ``en`` column of the train+val rows (997+504 = 1501).
* Real Italian reference = the ``it`` column of the SAME rows (FLORES is parallel),
  used only to eyeball generation quality in the report.
* EuroBERT mean sentence embeddings, aligned to the current 2006-row CSV by
  ``train_flores12.load_sent_embeddings`` convention (180m? no: 210m from
  ``embeddings/``; 610m from ``embeddings_610/``).  Synthetic Italian text is
  embedded at runtime with the SAME masked-mean pooling as ``extract_embeddings.py``
  (plain tokenization truncated to 511), so synthetic and real Italian vectors live
  in one space.

Generation
----------
A small instruct LLM (default ``Qwen/Qwen3-0.6B``) is prompted with a fixed,
explicit rewrite prompt::

    system : "You rewrite/translate English text into natural Italian, preserving
              meaning; output only the Italian text."
    user   : <the English sentence>

Greedy decoding (deterministic, cache-stable), bf16, batch 32, max_new_tokens 384,
``enable_thinking=False`` when the chat template supports it.  Outputs are cached on
disk so re-runs are free.

Arms (per backbone; features = RAW EuroBERT mean, matching train_flores12 "full")
-------------------------------------------------------------------------------
1. ``synth_it``      : train on synthetic Italian (EN train+val) -> real IT test.
2. ``real_it`` (ceiling): train on REAL Italian train+val -> real IT test.
3. ``en_transfer``   : train on English train+val -> real IT test (cross-lingual).
4. ``mixed``         : train on EN + synthetic-IT train+val (both rows) -> real IT test.
Metrics: accuracy / macro-F1 / OVR-AUC (macro) on the real Italian test.

XGBoost (CPU, fixed params): ``multi:softprob`` num_class 12, hist, max_bin 64,
cpu, n_jobs 2, seed 0, max_depth 8, min_child_weight 5, lr 0.1, n_estimators 400.

Outputs (create-only): ``synth_it.py`` / ``synth_it.json`` / ``synth_it.md`` / ``synth_it.png``.
Cache (generations + synthetic-IT embeddings + log/pid/status): ``%TEMP%\\opencode\\synth_it\\``.

Usage
-----
    python synth_it.py                 # foreground (GPU generate+embed, CPU XGB)
    python synth_it.py --launch        # detached hidden run via launch_hidden.vbs
    python synth_it.py --smoke         # tiny plumbing check (no outputs written)
    python synth_it.py --generator Qwen/Qwen3.5-0.8B --backbones 210m
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T                       # noqa: E402  (data prep + alignment)

OUT_JSON = os.path.join(REPO, "synth_it.json")
OUT_MD = os.path.join(REPO, "synth_it.md")
OUT_PNG = os.path.join(REPO, "synth_it.png")

TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
OPCODE = os.path.join(TEMP, "opencode")
CACHE = os.path.join(OPCODE, "synth_it")
os.makedirs(CACHE, exist_ok=True)
LOG_PATH = os.path.join(CACHE, "synth_it.log")
PID_PATH = os.path.join(CACHE, "synth_it.pid")
STATUS_PATH = os.path.join(CACHE, "synth_it_status.json")

N_CLASSES = 12
TRAIN_LANGS = ["en", "it"]

# Backbone registry: id -> (EuroBERT model id, sentence-embedding root, hidden size)
BACKBONES = {
    "210m": {"model": "EuroBERT/EuroBERT-210m", "dir": "embeddings", "hidden": 768},
    "610m": {"model": "EuroBERT/EuroBERT-610m", "dir": "embeddings_610", "hidden": 1152},
}
DEFAULT_BACKBONES = ["210m", "610m"]

DEFAULT_GENERATOR = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = ("You rewrite/translate English text into natural Italian, preserving "
                 "meaning; output only the Italian text.")
GEN_BATCH = 32
GEN_MAX_NEW = 384            # ~256 was too short: long FLORES sentences truncated mid-sentence

EMB_BATCH = 64
EMB_MAX_LEN = 512            # plain tokenize to MAX_LEN-1=511, matching extract_embeddings.py

XGB_PARAMS = dict(objective="multi:softprob", num_class=N_CLASSES, tree_method="hist",
                  max_bin=64, device="cpu", n_jobs=2, seed=0, max_depth=8,
                  min_child_weight=5, learning_rate=0.1, n_estimators=400)

ARMS = ["synth_it", "real_it", "en_transfer", "mixed"]

_T0 = time.time()
_SMOKE = False
_DEVICE = "cpu"

_GEN_MODEL = {}
_EMB_MODEL = {}


def log(msg):
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


def _hash(texts):
    return hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ metrics
def mc_metrics(y_true, probs, C=N_CLASSES):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(probs).argmax(1)
    labels = list(range(C))
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels,
                                   zero_division=0)),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                                   average="macro", labels=labels))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out


# ------------------------------------------------------------------ generation
def _build_prompts(tokenizer, texts):
    """Chat-template prompts for the fixed rewrite instruction (one per text)."""
    prompts = []
    for t in texts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(t)}]
        try:
            p = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        except TypeError:
            p = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        prompts.append(p)
    return prompts


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean(s):
    """Light post-processing.

    Strips reasoning blocks (SmolLM3 hybrids can emit ``<think>...</think>`` and/or
    ``<|im_end|>``), a stray leading 'Italian:' label, and collapses whitespace so a
    single-line training sentence remains.
    """
    s = _THINK_RE.sub(" ", s)
    # an unterminated think block means everything was reasoning -> keep the tail
    if "<think>" in s:
        s = s.rsplit("<think>", 1)[-1]
    s = s.replace("</think>", " ").replace("<|im_end|>", " ")
    s = s.replace("|im_end|", " ").strip()
    for pre in ("Italian:", "Italiano:", "Translation:", "Traduzione:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
    # collapse internal newlines to a single space (sentences are single-line)
    s = " ".join(s.split())
    return s


def _get_gen_model(generator):
    if generator not in _GEN_MODEL:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(generator)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            generator, torch_dtype=torch.bfloat16 if _DEVICE == "cuda" else torch.float32)
        model = model.to(_DEVICE).eval()
        _GEN_MODEL[generator] = (tok, model)
        log("  loaded generator %s on %s (bf16=%s)" % (generator, _DEVICE, _DEVICE == "cuda"))
    return _GEN_MODEL[generator]


def generate_italian(texts, generator):
    """Generate Italian for `texts` with `generator`; cache the outputs on disk."""
    cpath = os.path.join(CACHE, "gen_%s_%d_%s.json"
                         % (generator.replace("/", "__"), len(texts), _hash([generator] + list(texts))))
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            out = json.load(f)
        log("  [cache] generations %s (%d lines)" % (os.path.basename(cpath), len(out)))
        return out

    import torch
    tok, model = _get_gen_model(generator)
    prompts = _build_prompts(tok, texts)
    preds = []
    n = len(prompts)
    with torch.inference_mode():
        for i in range(0, n, GEN_BATCH):
            batch = prompts[i:i + GEN_BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(_DEVICE)
            in_len = enc["input_ids"].shape[1]
            gen = model.generate(**enc, do_sample=False, num_beams=1,
                                 max_new_tokens=GEN_MAX_NEW,
                                 pad_token_id=tok.pad_token_id)
            new_tokens = gen[:, in_len:]
            preds.extend(_clean(s) for s in tok.batch_decode(new_tokens, skip_special_tokens=True))
            done = min(i + GEN_BATCH, n)
            if (i // GEN_BATCH) % 8 == 0 or done == n:
                log("    gen %s %d/%d (%.1fs)" % (generator, done, n, time.time() - _T0))
    del model
    _GEN_MODEL.pop(generator, None)
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    log("  [save ] generations %s" % os.path.basename(cpath))
    return preds


# ------------------------------------------------------------------ embeddings
def _get_emb_model(model_id):
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
    """Masked mean over the last hidden state (== extract_embeddings.py 'mean' pooling)."""
    import torch
    tok, model = _get_emb_model(model_id)
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


def ensure_synth_emb(texts, backbone, model_id):
    """Return (synthetic-Italian mean embeddings, cache_path); compute + cache if missing."""
    tag = "synthemb_%s_%d_%s" % (backbone, len(texts), _hash(texts))
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        log("  [cache] synthetic-IT embeddings %s" % tag)
        return np.load(path)
    log("  embedding synthetic IT %s (%d texts)" % (tag, len(texts)))
    emb = embed_mean(texts, model_id)
    np.save(path, emb)
    return emb


def load_sent_mean(root, kept, n_source, langs):
    """Load + align EuroBERT mean sentence embeddings from root/mean/emb_<lang>.npy."""
    out = {}
    for lg in langs:
        path = os.path.join(root, "mean", "emb_%s.npy" % lg)
        arr = np.load(path)
        if arr.shape[0] != n_source:
            raise ValueError("expected %d rows in %s, got %s" % (n_source, path, arr.shape))
        out[lg] = np.asarray(arr[kept], dtype=np.float64)
    return out


# ------------------------------------------------------------------ one backbone
def run_backbone(bk, sent, y, te, rows_used, it_synth, smoke=False):
    import xgboost as xgb
    cfg = BACKBONES[bk]
    root = os.path.join(REPO, cfg["dir"])
    log("==== BACKBONE %s (%s, %s) ====" % (bk, cfg["model"], cfg["dir"]))
    n_source = len(sent["git"])
    X = load_sent_mean(root, sent["kept"], n_source, TRAIN_LANGS)
    log("  loaded mean embeddings: " + ", ".join("%s%s" % (lg, X[lg].shape) for lg in TRAIN_LANGS))

    trva_use = np.asarray(rows_used)
    Esyn = ensure_synth_emb(it_synth, bk, cfg["model"])
    if len(Esyn) != len(trva_use):
        raise ValueError("synthetic rows %d != train+val rows %d" % (len(Esyn), len(trva_use)))

    X_it_test = X["it"][te]
    X_it_tr = X["it"][trva_use]
    X_en_tr = X["en"][trva_use]
    y_tr = y[trva_use]
    y_te = y[te]

    p = dict(XGB_PARAMS)
    if smoke:
        p["n_estimators"] = 40

    arms = {}
    # 1. synthetic Italian
    clf = xgb.XGBClassifier(**p)
    clf.fit(Esyn, y_tr)
    arms["synth_it"] = mc_metrics(y_te, clf.predict_proba(X_it_test))
    # 2. real Italian ceiling
    clf = xgb.XGBClassifier(**p)
    clf.fit(X_it_tr, y_tr)
    arms["real_it"] = mc_metrics(y_te, clf.predict_proba(X_it_test))
    # 3. English transfer
    clf = xgb.XGBClassifier(**p)
    clf.fit(X_en_tr, y_tr)
    arms["en_transfer"] = mc_metrics(y_te, clf.predict_proba(X_it_test))
    # 4. EN + synthetic-IT mixed
    Xmix = np.concatenate([X_en_tr, Esyn], 0)
    ymix = np.concatenate([y_tr, y_tr], 0)
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xmix, ymix)
    arms["mixed"] = mc_metrics(y_te, clf.predict_proba(X_it_test))

    for a in ARMS:
        m = arms[a]
        log("  [%s] %-11s acc=%.4f macro=%.4f auc=%.4f"
            % (bk, a, m["accuracy"], m["macro_f1"], m["auc_ovr_macro"]))

    return {"backbone": bk, "model": cfg["model"], "emb_dir": cfg["dir"],
            "n_train_rows": int(len(trva_use)), "xgb_params": p, "arms": arms}


# ------------------------------------------------------------------ report / plot
def build_md(payload):
    cfg = payload["config"]
    L = []
    L.append("# Training on LLM-generated Italian, testing on REAL Italian FLORES")
    L.append("")
    L.append("Small instruct generator **`%s`** with a fixed rewrite prompt produces Italian "
             "training text from the **English** FLORES train+val sentences (997+504=1501). "
             "EuroBERT **mean** embeddings feed a fixed XGBoost topic classifier, evaluated on "
             "the **real Italian** FLORES test (505)." % cfg["generator"])
    L.append("")
    L.append("- Prompt (system): _\"%s\"_." % SYSTEM_PROMPT)
    L.append("- Decoding: greedy (`do_sample=False`), bf16, batch %d, max_new_tokens %d, "
             "`enable_thinking=False` when supported." % (cfg["gen_batch"], cfg["gen_max_new"]))
    L.append("- Data: FLORES `sent` (12-class), train=dev 997 / val=504 / test=505 (seed-0). "
             "Features = **RAW** EuroBERT mean (matching `train_flores12` \"full\").")
    L.append("- XGBoost (CPU, fixed): `multi:softprob`, num_class 12, hist, max_bin 64, n_jobs 2, "
             "seed 0, depth 8, mcw 5, lr 0.1, n 400. Trained on the arm's own train+val.")
    L.append("- Arms: **synth_it** (synthetic IT), **real_it** (ceiling), **en_transfer** (EN), "
             "**mixed** (EN + synthetic IT).")
    L.append("")
    L.append("## Example generations (source EN -> generated IT, and the FLORES IT reference)")
    L.append("")
    for e in payload["gen_samples"]:
        L.append("- **EN:** %s" % e["en"])
        L.append("  - gen IT: _%s_" % e["gen_it"])
        L.append("  - flor IT ref: %s" % e["ref_it"])
    L.append("")
    for bk in payload["backbones"]:
        bb = payload["backbones"][bk]
        L.append("## Backbone %s (%s, %d-d)" % (bk, bb["model"], BACKBONES[bk]["hidden"]))
        L.append("")
        L.append("| arm | accuracy | macro-F1 | OVR-AUC |")
        L.append("|---|---|---|---|")
        for a in ARMS:
            m = bb["arms"][a]
            L.append("| %s | %.4f | %.4f | %.4f |" % (a, m["accuracy"], m["macro_f1"],
                                                      m["auc_ovr_macro"]))
        s = bb["arms"]["synth_it"]["macro_f1"]
        c = bb["arms"]["real_it"]["macro_f1"]
        r = s / c if c else float("nan")
        L.append("")
        e = bb["arms"]["en_transfer"]["macro_f1"]
        m = bb["arms"]["mixed"]["macro_f1"]
        L.append("- synthetic / real-IT ceiling (macro-F1) = **%.3f / %.3f = %.2f**." % (s, c, r))
        L.append("- vs EN-transfer (%+.3f); mixed %.3f (%+.3f vs ceiling, %+.3f vs EN-transfer)."
                 % (s - e, m, m - c, m - e))
        L.append("")
    L.append("![synth_it](synth_it.png)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(payload["verdict"])
    L.append("")
    return "\n".join(L)


def plot(payload, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bks = list(payload["backbones"])
    fig, axes = plt.subplots(1, len(bks), figsize=(6.4 * len(bks), 5.0), squeeze=False)
    metrics = [("accuracy", "accuracy", "#4c72b0"),
               ("macro_f1", "macro-F1", "#dd8452"),
               ("auc_ovr_macro", "OVR-AUC", "#55a868")]
    w = 0.25
    for ax, bk in zip(axes[0], bks):
        bb = payload["backbones"][bk]
        x = np.arange(len(ARMS))
        for j, (mk, label, col) in enumerate(metrics):
            vals = [bb["arms"][a][mk] for a in ARMS]
            bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=col)
            for r in bars:
                ax.annotate("%.3f" % r.get_height(),
                            (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title("EuroBERT-%s mean, XGBoost on real IT test" % bk, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Synthetic-Italian training vs real-Italian ceiling (real Italian FLORES test)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log("wrote %s" % out_png)


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
    global _SMOKE, _DEVICE, GEN_BATCH, GEN_MAX_NEW
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Synthetic-Italian FLORES-12 classifier study.")
    ap.add_argument("--generator", default=DEFAULT_GENERATOR,
                    help="small instruct generator HF id (default %s)" % DEFAULT_GENERATOR)
    ap.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                    help="comma list from {%s} (default all)" % ",".join(BACKBONES))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gen-limit", type=int, default=0,
                    help="limit number of sentences to generate (0 = all)")
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH,
                    help="generation batch size (default %d; lower for big models on small GPUs)"
                         % GEN_BATCH)
    ap.add_argument("--max-new", type=int, default=GEN_MAX_NEW,
                    help="max_new_tokens per generation (default %d)" % GEN_MAX_NEW)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="rebuild synth_it.md/.png from synth_it.json (no generation/training)")
    args = ap.parse_args(argv)

    if args.from_json:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write(build_md(payload))
        plot(payload, OUT_PNG)
        print("rebuilt %s and %s from %s" % (OUT_MD, OUT_PNG, OUT_JSON))
        return 0

    if args.launch:
        passthrough = []
        if args.generator != DEFAULT_GENERATOR:
            passthrough += ["--generator", args.generator]
        if args.backbones != ",".join(DEFAULT_BACKBONES):
            passthrough += ["--backbones", args.backbones]
        if args.device != "auto":
            passthrough += ["--device", args.device]
        if args.gen_limit:
            passthrough += ["--gen-limit", str(args.gen_limit)]
        launch_detached(passthrough)
        return 0

    _SMOKE = args.smoke
    GEN_BATCH = args.gen_batch
    GEN_MAX_NEW = args.max_new
    bks = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for b in bks:
        if b not in BACKBONES:
            raise SystemExit("unknown backbone %r (choose from %r)" % (b, list(BACKBONES)))

    import torch
    _DEVICE = args.device
    if _DEVICE == "auto":
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    log("start pid=%d device=%s smoke=%s generator=%s backbones=%s"
        % (os.getpid(), _DEVICE, _SMOKE, args.generator, bks))
    write_status({"phase": "start", "device": _DEVICE, "generator": args.generator,
                  "backbones": bks})

    sent = T.prepare_sentence()
    y = np.asarray(sent["y"], dtype=int)
    tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
    trva = np.concatenate([tr, va])
    cur = sent["cur"]
    en_src = cur.iloc[trva]["en"].astype(str).tolist()
    log("data: classes=%d train=%d val=%d test=%d -> gen %d EN sentences"
        % (len(sent["classes"]), len(tr), len(va), len(te), len(en_src)))

    if _SMOKE:
        # class-balanced subset so every arm still sees all 12 classes (XGB label check)
        rows_used, seen = [], {}
        for r in trva:
            c = int(y[r])
            if seen.get(c, 0) < 4:
                rows_used.append(int(r))
                seen[c] = seen.get(c, 0) + 1
        rows_used = np.asarray(rows_used, dtype=int)
    else:
        rows_used = trva
        if args.gen_limit:
            rows_used = trva[:args.gen_limit]
    en_use = cur.iloc[rows_used]["en"].astype(str).tolist()
    n_gen = len(en_use)

    write_status({"phase": "generate", "n": n_gen, "generator": args.generator,
                  "device": _DEVICE})
    it_synth = generate_italian(en_use, args.generator)
    if len(it_synth) != n_gen:
        raise ValueError("generation length %d != requested %d" % (len(it_synth), n_gen))
    n_empty = sum(1 for s in it_synth if not s.strip())
    log("generated %d Italian sentences (%d empty)" % (len(it_synth), n_empty))

    # eyeball samples: EN source, generated IT, FLORES IT reference (same rows)
    it_ref = cur.iloc[rows_used]["it"].astype(str).tolist()
    idxs = [0, n_gen // 4, n_gen // 2, (3 * n_gen) // 4, n_gen - 1] if n_gen >= 5 else list(range(n_gen))
    gen_samples = [{"en": en_use[i], "gen_it": it_synth[i], "ref_it": it_ref[i]} for i in idxs]

    payload = {
        "config": {
            "dataset": "flores sent", "n_classes": N_CLASSES, "classes": sent["classes"],
            "generator": args.generator, "system_prompt": SYSTEM_PROMPT,
            "gen_batch": GEN_BATCH, "gen_max_new": GEN_MAX_NEW, "decode": "greedy",
            "feature": "RAW EuroBERT mean (train_flores12 'full'), aligned to current CSV",
            "splits": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "n_synthetic": int(n_gen), "n_empty": int(n_empty),
            "xgb": XGB_PARAMS, "device": _DEVICE, "smoke": _SMOKE, "seed": 0,
            "python": sys.version.split()[0],
        },
        "gen_samples": gen_samples,
        "backbones": {},
    }

    for bk in bks:
        write_status({"phase": "backbone %s" % bk, "device": _DEVICE})
        res = run_backbone(bk, sent, y, te, rows_used, it_synth, smoke=_SMOKE)
        payload["backbones"][bk] = res

    # verdict from the primary (first) backbone
    primary = bks[0]
    arms = payload["backbones"][primary]["arms"]
    s, c = arms["synth_it"]["macro_f1"], arms["real_it"]["macro_f1"]
    e, m = arms["en_transfer"]["macro_f1"], arms["mixed"]["macro_f1"]
    ratio = s / c if c else float("nan")
    if ratio >= 0.9:
        verdict = ("On EuroBERT-%s, synthetic-Italian training reaches **%.0f%%** of the "
                   "real-Italian ceiling (macro-F1 %.3f vs %.3f) -- i.e. it **approaches** the "
                   "ceiling." % (primary, ratio * 100, s, c))
    elif ratio >= 0.7:
        verdict = ("On EuroBERT-%s, synthetic-Italian training reaches **%.0f%%** of the "
                   "real-Italian ceiling (macro-F1 %.3f vs %.3f) -- close, but a clear gap "
                   "remains." % (primary, ratio * 100, s, c))
    else:
        verdict = ("On EuroBERT-%s, synthetic-Italian training reaches only **%.0f%%** of the "
                   "real-Italian ceiling (macro-F1 %.3f vs %.3f) -- a large gap remains."
                   % (primary, ratio * 100, s, c))
    verdict += (" It clearly beats EN-transfer (%.3f) by %+.3f macro-F1; mixing EN + synthetic IT "
                "gives %.3f (%+.3f vs EN-transfer)." % (e, s - e, m, m - e))
    payload["verdict"] = verdict

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
    write_status({"phase": "done", "verdict": payload.get("verdict")})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        write_status({"phase": "error"})
        raise
