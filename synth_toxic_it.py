"""synth_toxic_it.py -- train a binary TOXICITY classifier on LLM-GENERATED Italian,
evaluate on the REAL Italian toxic test.

Question
--------
Same recipe as ``synth_it.py`` (FLORES), but for the recall-critical, unbalanced
toxicity task: take the **English** toxic training text, use a small instruct LLM
(``HuggingFaceTB/SmolLM3-3B``) with the cross-lingual rewrite prompt from
``paraphrase_prompts.py`` / ``PARAPHRASE_TEMPLATE.md`` to generate **Italian**
training text, embed the synthetic Italian with EuroBERT **mean**, train an XGBoost
toxicity classifier on it, and evaluate on the **REAL Italian** toxic test.

Does training on LLM-generated Italian approach the real-Italian ceiling?

Data & features
---------------
* ``data/toxi_clean/{en,it}_{train,validation,test}.parquet`` (columns text,is_toxic,lang).
  English train = 24,000 rows; the generation is capped to a subsample of
  ``--en-cap`` rows (default **8,000**, ``seed 0``) to bound runtime.  Real Italian
  test = ``it_test.parquet`` (**2,310** rows).
* EuroBERT **mean** sentence embeddings, read from the bf16-consistent cache
  ``emb_bf16/toxic/<bb>/emb_<lang>_{train,validation,test}.npy`` (masked mean over the
  plain forward, max_length 512, bf16 forward -> fp32 arrays, matching
  ``backbone_grid._embed_mean`` / ``extract_embeddings.py``).  Synthetic Italian is
  embedded at runtime with the SAME pooling so synthetic and real vectors share a space.

Generation
----------
Cross-lingual rewrite (EN -> IT), greedy, bf16, batch ``--gen-batch`` (default 32),
``max_new_tokens`` ~256, ``enable_thinking=False`` when the chat template supports it.
Outputs are cached on disk so re-runs are free.

Arms (evaluated on the REAL Italian toxic test)
-----------------------------------------------
1. ``synth_it``   : train on synthetic Italian (EN cap) -> real IT test.
2. ``real_it``    : train on the REAL Italian train (ceiling) -> real IT test.
3. ``en_transfer``: train on the SAME English cap -> real IT test (cross-lingual).
4. ``mixed``      : train on EN cap + synthetic-IT (both rows) -> real IT test.

Metrics: accuracy / F1 (positive class, @0.5) / AUC, plus the recall-first view
(recall@FPR{5,10%}, FPR@recall=0.90) for synth_it / real_it / en_transfer (threshold-free).

XGBoost (CPU, fixed): ``binary:logistic``, hist, max_bin 64, cpu, n_jobs 2, seed 0,
max_depth 8, min_child_weight 5, lr 0.1, n_estimators 400.

Outputs (create-only): ``synth_toxic_it.py`` / ``.json`` / ``.md`` / ``.png``.
Cache (generations + synthetic-IT embeddings + log/pid/status): ``%TEMP%\\opencode\\synth_toxic_it\\``.

Usage
-----
    python synth_toxic_it.py                 # foreground (GPU generate+embed, CPU XGB)
    python synth_toxic_it.py --launch        # detached hidden run via launch_hidden.vbs
    python synth_toxic_it.py --smoke         # tiny plumbing check (no outputs written)
    python synth_toxic_it.py --generator HuggingFaceTB/SmolLM3-3B --en-cap 8000
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

try:
    from paraphrase_prompts import build_messages as _build_pp_messages
except Exception:                                              # pragma: no cover
    _build_pp_messages = None

OUT_JSON = os.path.join(REPO, "synth_toxic_it.json")
OUT_MD = os.path.join(REPO, "synth_toxic_it.md")
OUT_PNG = os.path.join(REPO, "synth_toxic_it.png")

TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
OPCODE = os.path.join(TEMP, "opencode")
CACHE = os.path.join(OPCODE, "synth_toxic_it")
os.makedirs(CACHE, exist_ok=True)
LOG_PATH = os.path.join(CACHE, "synth_toxic_it.log")
PID_PATH = os.path.join(CACHE, "synth_toxic_it.pid")
STATUS_PATH = os.path.join(CACHE, "synth_toxic_it_status.json")

DATA_DIR = os.path.join(REPO, "data", "toxi_clean")

# Backbone registry: id -> (EuroBERT model id, bf16 embedding root, hidden size)
BACKBONES = {
    "210m": {"model": "EuroBERT/EuroBERT-210m", "dir": "emb_bf16/toxic/210m", "hidden": 768},
    "610m": {"model": "EuroBERT/EuroBERT-610m", "dir": "emb_bf16/toxic/610m", "hidden": 1152},
}
DEFAULT_BACKBONES = ["210m", "610m"]

DEFAULT_GENERATOR = "HuggingFaceTB/SmolLM3-3B"
GEN_BATCH = 32
GEN_MAX_NEW = 256
GEN_CHUNK = 128            # per-chunk generation cache -> resumable across kills/contention
DEFAULT_EN_CAP = 8000
DEFAULT_SEED = 0

EMB_BATCH = 64
EMB_MAX_LEN = 512

XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_bin=64, device="cpu",
                  n_jobs=2, seed=0, max_depth=8, min_child_weight=5, learning_rate=0.1,
                  n_estimators=400)

ARMS = ["synth_it", "real_it", "en_transfer", "mixed"]
RECALL_ARMS = ["synth_it", "real_it", "en_transfer"]     # recall-first view
FPR_GRID = [0.05, 0.10]
RECALL_TARGET = 0.90

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
def _roc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _f1_at(y, s, thr):
    from sklearn.metrics import f1_score, accuracy_score
    pred = (np.asarray(s) >= thr).astype(int)
    return float(f1_score(y, pred, pos_label=1, zero_division=0)), float(accuracy_score(y, pred))


def binary_metrics(y_true, probs, thr=0.5):
    """accuracy / F1(positive) @thr / AUC."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(probs, dtype=float)
    f1, acc = _f1_at(y, s, thr)
    return {"accuracy": acc, "f1_pos": f1, "auc": _roc(y, s), "threshold": float(thr)}


def _cum_curves(y, s):
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")
    ys = np.asarray(y, dtype=np.int64)[order]
    return int(ys.sum()), np.cumsum(ys), np.cumsum(1 - ys)


def recall_at_fpr(y, s, fpr):
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    budget = int(np.floor(fpr * n_neg))
    j = int(np.searchsorted(cum_fp, budget, side="right")) - 1
    if j < 0:
        return 0.0
    return float(cum_tp[j] / n_pos)


def fpr_for_recall(y, s, target=RECALL_TARGET):
    n_pos, cum_tp, cum_fp = _cum_curves(y, s)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    need = int(np.ceil(target * n_pos - 1e-9))
    j = int(np.searchsorted(cum_tp, need, side="left"))
    if j >= len(cum_tp):
        return None
    return float(cum_fp[j] / n_neg)


def f1_optimal_threshold(y, s):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2.0 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    if k >= len(thr):
        k = len(thr) - 1
    return float(thr[k])


def recall_first(y_true, probs):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(probs, dtype=float)
    out = {}
    for p in FPR_GRID:
        out["recall_at_fpr_%d" % int(round(p * 100))] = recall_at_fpr(y, s, p)
    out["fpr_for_recall_90"] = fpr_for_recall(y, s, RECALL_TARGET)
    return out


# ------------------------------------------------------------------ generation
def _build_prompts(tokenizer, texts):
    """Chat-template prompts for the cross-lingual rewrite instruction (EN -> IT)."""
    prompts = []
    for t in texts:
        if _build_pp_messages is not None:
            messages = _build_pp_messages("crosslingual", text=str(t),
                                          src="English", tgt="Italian")
        else:
            messages = [
                {"role": "system",
                 "content": ("You are a careful multilingual data generator. You rewrite the "
                             "user's text into Italian while preserving its meaning exactly. "
                             "You output only the Italian text, with no quotes, labels, or "
                             "commentary.")},
                {"role": "user",
                 "content": ("Rewrite the following English text into natural, fluent Italian.\n"
                             "- Preserve ALL factual content and entities.\n"
                             "- Do not add or remove information.\n"
                             "- Output ONLY the Italian text. No preamble, no quotes.\n\n"
                             "English text:\n\"\"\" %s \"\"\"" % str(t))},
            ]
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
    """Light post-processing: strip reasoning blocks / stray labels, collapse whitespace."""
    s = _THINK_RE.sub(" ", s)
    if "<think>" in s:
        s = s.rsplit("<think>", 1)[-1]
    s = s.replace("</think>", " ").replace("<|im_end|>", " ")
    s = s.replace("|im_end|", " ").strip()
    for pre in ("Italian:", "Italiano:", "Translation:", "Traduzione:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
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
    """Generate Italian for `texts`; cache the results on disk (whole-list fast path + per-chunk
    resume so a killed/contended run keeps completed work)."""
    n = len(texts)
    cpath = os.path.join(CACHE, "gen_%s_%d_%s.json"
                         % (generator.replace("/", "__"), n, _hash([generator] + list(texts))))
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            out = json.load(f)
        log("  [cache] generations %s (%d lines)" % (os.path.basename(cpath), len(out)))
        return out

    def _chunk_path(c0, sub):
        return os.path.join(CACHE, "genchunk_%s_%d_%d_%s.json"
                            % (generator.replace("/", "__"), c0, n, _hash([generator] + list(sub))))

    # resume: gather already-cached chunks without loading the model
    out, missing = [], []
    for c0 in range(0, n, GEN_CHUNK):
        sub = texts[c0:c0 + GEN_CHUNK]
        cp = _chunk_path(c0, sub)
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                out.extend(json.load(f))
        else:
            missing.append((c0, sub, cp))
    if missing:
        log("  generation: %d/%d chunks cached, %d to do"
            % ((n + GEN_CHUNK - 1) // GEN_CHUNK - len(missing), (n + GEN_CHUNK - 1) // GEN_CHUNK,
               len(missing)))
        import torch
        tok, model = _get_gen_model(generator)
        for c0, sub, cp in missing:
            preds = []
            prompts = _build_prompts(tok, sub)
            with torch.inference_mode():
                for i in range(0, len(prompts), GEN_BATCH):
                    batch = prompts[i:i + GEN_BATCH]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                              max_length=1024).to(_DEVICE)
                    in_len = enc["input_ids"].shape[1]
                    gen = model.generate(**enc, do_sample=False, num_beams=1,
                                         max_new_tokens=GEN_MAX_NEW,
                                         pad_token_id=tok.pad_token_id)
                    new_tokens = gen[:, in_len:]
                    preds.extend(_clean(s) for s in
                                 tok.batch_decode(new_tokens, skip_special_tokens=True))
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False)
            out.extend(preds)
            log("    gen %s chunk@%d done (%d/%d rows cached)"
                % (generator, c0, len(out), n))
        del model
        _GEN_MODEL.pop(generator, None)
        if _DEVICE == "cuda":
            torch.cuda.empty_cache()
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    log("  [save ] generations %s (%d lines)" % (os.path.basename(cpath), len(out)))
    return out


# ------------------------------------------------------------------ embeddings
def _get_emb_model(model_id):
    if model_id not in _EMB_MODEL:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(torch.bfloat16).to(_DEVICE).eval()
        _EMB_MODEL[model_id] = (tok, model)
        log("  loaded embedder %s (hidden=%d) on %s (bf16)"
            % (model_id, model.config.hidden_size, _DEVICE))
    return _EMB_MODEL[model_id]


def embed_mean(texts, model_id):
    """Masked-mean over the last hidden state (== backbone_grid._embed_mean: bf16 forward,
    max_length 512, batch 64, length-sorted then restored)."""
    import torch
    tok, model = _get_emb_model(model_id)
    n = len(texts)
    if n == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    order = np.argsort(np.array([len(t) for t in texts], dtype=np.int64), kind="stable")
    sorted_texts = [texts[i] for i in order]
    acc = []
    with torch.inference_mode():
        for i in range(0, n, EMB_BATCH):
            batch = tok(sorted_texts[i:i + EMB_BATCH], padding=True, truncation=True,
                        max_length=EMB_MAX_LEN, return_tensors="pt")
            ids, mask = batch["input_ids"].to(_DEVICE), batch["attention_mask"].to(_DEVICE)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).to(h.dtype)
            acc.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float32))
    e_sorted = np.concatenate(acc, 0).astype(np.float32)
    out = np.empty_like(e_sorted)
    out[order] = e_sorted
    return out


def ensure_synth_emb(texts, backbone, model_id):
    """Return synthetic-Italian mean embeddings (cache path), compute + cache if missing."""
    tag = "synthemb_%s_%d_%s" % (backbone, len(texts), _hash(texts))
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        log("  [cache] synthetic-IT embeddings %s" % tag)
        return np.load(path)
    log("  embedding synthetic IT %s (%d texts)" % (tag, len(texts)))
    emb = embed_mean(texts, model_id)
    np.save(path, emb)
    return emb


def load_stored(bb, lang, split):
    """Load the bf16-consistent mean embeddings emb_bf16/toxic/<bb>/emb_<lang>_<split>.npy."""
    path = os.path.join(REPO, BACKBONES[bb]["dir"], "emb_%s_%s.npy" % (lang, split))
    return np.asarray(np.load(path), dtype=np.float32)


# ------------------------------------------------------------------ one backbone
def run_backbone(bb, en_tr_emb, y_en_tr, it_tr_emb, y_it_tr,
                 it_val_emb, y_it_val, it_te_emb, y_it_te, it_synth, smoke=False):
    import xgboost as xgb
    cfg = BACKBONES[bb]
    log("==== BACKBONE %s (%s, %s) ====" % (bb, cfg["model"], cfg["dir"]))

    Esyn = ensure_synth_emb(it_synth, bb, cfg["model"])
    if len(Esyn) != len(y_en_tr):
        raise ValueError("synthetic rows %d != EN train rows %d" % (len(Esyn), len(y_en_tr)))

    p = dict(XGB_PARAMS)
    if smoke:
        p["n_estimators"] = 40

    arms = {}
    models = {}

    def fit(X, y):
        clf = xgb.XGBClassifier(**p)
        clf.fit(X, y)
        return clf

    # 1. synthetic Italian
    models["synth_it"] = fit(Esyn, y_en_tr)
    # 2. real Italian ceiling
    models["real_it"] = fit(it_tr_emb, y_it_tr)
    # 3. English transfer (same EN cap)
    models["en_transfer"] = fit(en_tr_emb, y_en_tr)
    # 4. EN + synthetic-IT mixed
    models["mixed"] = fit(np.concatenate([en_tr_emb, Esyn], 0),
                          np.concatenate([y_en_tr, y_en_tr], 0))

    for a in ARMS:
        clf = models[a]
        s_te = clf.predict_proba(it_te_emb)[:, 1]
        m = binary_metrics(y_it_te, s_te, 0.5)
        m.update(recall_first(y_it_te, s_te))
        # F1 tuned on the REAL Italian validation split (calibration only; no training rows)
        s_va = clf.predict_proba(it_val_emb)[:, 1]
        thr_v = f1_optimal_threshold(y_it_val, s_va)
        m["val_threshold"] = float(thr_v)
        m["f1_pos_val_thr"], m["accuracy_val_thr"] = _f1_at(y_it_te, s_te, thr_v)
        arms[a] = m

    for a in ARMS:
        m = arms[a]
        log("  [%s] %-11s acc=%.4f f1=%.4f auc=%.4f | R@FPR5=%.3f R@FPR10=%.3f FPR@R90=%s"
            % (bb, a, m["accuracy"], m["f1_pos"], m["auc"],
               m["recall_at_fpr_5"], m["recall_at_fpr_10"],
               "n/a" if m["fpr_for_recall_90"] is None else "%.3f" % m["fpr_for_recall_90"]))

    return {"backbone": bb, "model": cfg["model"], "emb_dir": cfg["dir"],
            "n_train_synth": int(len(Esyn)), "n_train_real_it": int(len(y_it_tr)),
            "xgb_params": p, "arms": arms}


# ------------------------------------------------------------------ report / plot
def build_md(payload):
    cfg = payload["config"]
    L = []
    L.append("# Training on LLM-generated Italian, testing on the REAL Italian toxic test")
    L.append("")
    L.append("Small instruct generator **`%s`** with the cross-lingual rewrite prompt "
             "(`paraphrase_prompts.py`) produces Italian training text from the **English** toxic "
             "train (capped to **%d** rows, seed %d). EuroBERT **mean** embeddings feed a fixed "
             "XGBoost toxicity classifier, evaluated on the **real Italian** toxic test (%d rows)."
             % (cfg["generator"], cfg["en_cap"], cfg["seed"], cfg["n_it_test"]))
    L.append("")
    L.append("- Prompt: cross-lingual rewrite (EN -> IT), `paraphrase_prompts.build_messages("
             "\"crosslingual\", src=\"English\", tgt=\"Italian\")`.")
    L.append("- Decoding: greedy (`do_sample=False`), bf16, batch %d, max_new_tokens %d, "
             "`enable_thinking=False` when supported." % (cfg["gen_batch"], cfg["gen_max_new"]))
    L.append("- Data: `data/toxi_clean/` (binary `is_toxic`). EN train 24,000 -> cap %d; "
             "real IT train %d; IT val %d; **real IT test %d** (positive rate %.3f)."
             % (cfg["en_cap"], cfg["n_it_train"], cfg["n_it_val"], cfg["n_it_test"],
                cfg["it_test_pos_rate"]))
    L.append("- Features: **RAW** EuroBERT mean (bf16 forward -> fp32), read from "
             "`emb_bf16/toxic/<bb>/`; synthetic-IT embedded at runtime with the SAME pooling "
             "(no `perlang_z`, since the synthesis study must not use the target language's stats).")
    L.append("- XGBoost (CPU, fixed): `binary:logistic`, hist, max_bin 64, cpu, n_jobs 2, seed 0, "
             "depth 8, mcw 5, lr 0.1, n 400. Trained on each arm's own rows.")
    L.append("- Arms: **synth_it** (synthetic IT), **real_it** (ceiling), **en_transfer** (EN), "
             "**mixed** (EN + synthetic IT).")
    L.append("")
    L.append("## Example generations (EN source -> generated IT)")
    L.append("")
    if payload.get("gen_samples"):
        for e in payload["gen_samples"]:
            L.append("- **EN:** %s" % e["en"])
            L.append("  - gen IT: _%s_" % e["gen_it"])
        if all("ref_it" in e for e in payload["gen_samples"]):
            L.append("")
            L.append("_(No real-Italian reference column is shown: the real IT test is the "
                     "evaluation set, not a parallel reference.)_")
    L.append("")
    for bb in payload["backbones"]:
        bob = payload["backbones"][bb]
        L.append("## Backbone %s (%s, %d-d)" % (bb, bob["model"], BACKBONES[bb]["hidden"]))
        L.append("")
        L.append("| arm | accuracy | F1 (pos) | AUC | F1@val-thr |")
        L.append("|---|---|---|---|---|")
        for a in ARMS:
            m = bob["arms"][a]
            L.append("| %s | %.4f | %.4f | %.4f | %.4f |"
                     % (a, m["accuracy"], m["f1_pos"], m["auc"], m["f1_pos_val_thr"]))
        L.append("")
        L.append("**Recall-first view** (threshold-free; train sizes: synth %d, real-IT %d):"
                 % (bob["n_train_synth"], bob["n_train_real_it"]))
        L.append("")
        L.append("| arm | R@FPR5% | R@FPR10% | FPR@R90 |")
        L.append("|---|---|---|---|")
        for a in RECALL_ARMS:
            m = bob["arms"][a]
            L.append("| %s | %s | %s | %s |"
                     % (a, _f(m["recall_at_fpr_5"]), _f(m["recall_at_fpr_10"]),
                        _f(m["fpr_for_recall_90"])))
        s = bob["arms"]["synth_it"]["f1_pos"]
        c = bob["arms"]["real_it"]["f1_pos"]
        e = bob["arms"]["en_transfer"]["f1_pos"]
        x = bob["arms"]["mixed"]["f1_pos"]
        r = s / c if c else float("nan")
        L.append("")
        L.append("- synthetic / real-IT ceiling (F1) = **%.3f / %.3f = %.2f**." % (s, c, r))
        L.append("- vs EN-transfer (%+.3f); mixed %.3f (%+.3f vs ceiling, %+.3f vs EN-transfer)."
                 % (s - e, x, x - c, x - e))
        L.append("")
    L.append("![synth_toxic_it](synth_toxic_it.png)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(payload["verdict"])
    L.append("")
    return "\n".join(L)


def _f(v, nd=3):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def plot(payload, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bks = list(payload["backbones"])
    fig, axes = plt.subplots(1, len(bks) + 1, figsize=(6.0 * (len(bks) + 1), 5.0), squeeze=False)
    metrics = [("accuracy", "accuracy", "#4c72b0"),
               ("f1_pos", "F1 (pos)", "#dd8452"),
               ("auc", "AUC", "#55a868")]
    w = 0.25
    for ax, bk in zip(axes[0], bks):
        bob = payload["backbones"][bk]
        x = np.arange(len(ARMS))
        for j, (mk, label, col) in enumerate(metrics):
            vals = [bob["arms"][a][mk] for a in ARMS]
            bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=col)
            for r in bars:
                ax.annotate("%.3f" % r.get_height(),
                            (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title("EuroBERT-%s mean, XGB on real IT test" % bk, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    # recall-first panel (primary backbone bused 210m)
    ax = axes[0][-1]
    bk = bks[0]
    bob = payload["backbones"][bk]
    x = np.arange(len(RECALL_ARMS))
    fpr_vals = [bob["arms"][a]["fpr_for_recall_90"] for a in RECALL_ARMS]
    fpr_vals = [0.0 if v is None else v for v in fpr_vals]
    r5 = [bob["arms"][a]["recall_at_fpr_5"] or 0.0 for a in RECALL_ARMS]
    r10 = [bob["arms"][a]["recall_at_fpr_10"] or 0.0 for a in RECALL_ARMS]
    w = 0.25
    b1 = ax.bar(x - w, r5, w, label="R@FPR5%", color="#8172b3")
    b2 = ax.bar(x, r10, w, label="R@FPR10%", color="#937860")
    b3 = ax.bar(x + w, fpr_vals, w, label="FPR@R90 (lower better)", color="#da8bc3")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.annotate("%.2f" % r.get_height(),
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(RECALL_ARMS, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("Recall-first (%s), real IT test" % bk, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Synthetic-Italian toxicity training vs real-Italian ceiling (real Italian test)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------ launch / main
def _wait_for_gpu(min_free_mb, timeout_s):
    """Poll until the GPU has >= min_free_mb free (survives a concurrently busy GPU)."""
    import torch
    t0 = time.time()
    while True:
        free_b, _total_b = torch.cuda.mem_get_info()
        free_mb = free_b / 1024.0 / 1024.0
        if free_mb >= min_free_mb:
            log("  gpu free now %.0f MB (>= %d) -> proceed" % (free_mb, min_free_mb))
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError("GPU wait timed out after %ds (free %.0f MB < %d MB)"
                               % (timeout_s, free_mb, min_free_mb))
        log("  waiting for GPU: %.0f MB free < %d MB required" % (free_mb, min_free_mb))
        write_status({"phase": "wait_gpu", "free_mb": round(free_mb), "need_mb": min_free_mb})
        time.sleep(20)


def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, LOG_PATH, REPO], cwd=REPO)
    print("launched detached; log=%s" % LOG_PATH)


def main(argv=None):
    global _SMOKE, _DEVICE, GEN_BATCH, GEN_MAX_NEW
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Synthetic-Italian toxicity classifier study.")
    ap.add_argument("--generator", default=DEFAULT_GENERATOR,
                    help="small instruct generator HF id (default %s)" % DEFAULT_GENERATOR)
    ap.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                    help="comma list from {%s} (default all)" % ",".join(BACKBONES))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--en-cap", type=int, default=DEFAULT_EN_CAP,
                    help="cap of English train rows to generate (default %d)" % DEFAULT_EN_CAP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="cap subsample seed")
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH,
                    help="generation batch size (default %d)" % GEN_BATCH)
    ap.add_argument("--max-new", type=int, default=GEN_MAX_NEW,
                    help="max_new_tokens per generation (default %d)" % GEN_MAX_NEW)
    ap.add_argument("--wait-gpu-mb", type=int, default=0,
                    help="min free GPU MiB before loading a model (0 = do not wait)")
    ap.add_argument("--wait-gpu-sec", type=int, default=7200,
                    help="max seconds to wait for GPU memory (default 7200)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="rebuild synth_toxic_it.md/.png from synth_toxic_it.json")
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
        if args.en_cap != DEFAULT_EN_CAP:
            passthrough += ["--en-cap", str(args.en_cap)]
        if args.seed != DEFAULT_SEED:
            passthrough += ["--seed", str(args.seed)]
        if args.wait_gpu_mb:
            passthrough += ["--wait-gpu-mb", str(args.wait_gpu_mb)]
            passthrough += ["--wait-gpu-sec", str(args.wait_gpu_sec)]
        launch_detached(passthrough)
        return 0

    _SMOKE = args.smoke
    GEN_BATCH = args.gen_batch
    GEN_MAX_NEW = args.max_new
    bks = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for b in bks:
        if b not in BACKBONES:
            raise SystemExit("unknown backbone %r (choose from %r)" % (b, list(BACKBONES)))

    import pandas as pd
    import torch
    _DEVICE = args.device
    if _DEVICE == "auto":
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    log("start pid=%d device=%s smoke=%s generator=%s backbones=%s en_cap=%d seed=%d"
        % (os.getpid(), _DEVICE, _SMOKE, args.generator, bks, args.en_cap, args.seed))
    write_status({"phase": "start", "device": _DEVICE, "generator": args.generator,
                  "backbones": bks, "en_cap": args.en_cap})

    if _DEVICE == "cuda" and args.wait_gpu_mb > 0:
        _wait_for_gpu(args.wait_gpu_mb, args.wait_gpu_sec)

    def read_split(lang, split):
        df = pd.read_parquet(os.path.join(DATA_DIR, "%s_%s.parquet" % (lang, split)))
        return df["text"].astype(str).tolist(), df["is_toxic"].astype(int).to_numpy()

    en_tr_text, y_en_all = read_split("en", "train")
    it_tr_text, y_it_tr = read_split("it", "train")
    it_va_text, y_it_va = read_split("it", "validation")
    it_te_text, y_it_te = read_split("it", "test")
    n_en = len(y_en_all)

    # --- choose the EN cap (deterministic subsample) ---
    cap = n_en if _SMOKE else min(args.en_cap, n_en)
    if _SMOKE:
        cap = min(16, n_en)
    rng = np.random.default_rng(args.seed)
    rows_used = np.sort(rng.choice(n_en, size=cap, replace=False))
    en_use = [en_tr_text[i] for i in rows_used]
    y_en_tr = y_en_all[rows_used]
    log("data: EN train=%d (cap=%d, pos=%.3f) | IT train=%d val=%d test=%d (test pos=%.3f)"
        % (n_en, cap, float(y_en_tr.mean()), len(y_it_tr), len(y_it_va), len(y_it_te),
           float(y_it_te.mean())))

    write_status({"phase": "generate", "n": cap, "generator": args.generator, "device": _DEVICE})
    it_synth = generate_italian(en_use, args.generator)
    if len(it_synth) != cap:
        raise ValueError("generation length %d != requested %d" % (len(it_synth), cap))
    n_empty = sum(1 for s in it_synth if not s.strip())
    log("generated %d Italian rows (%d empty)" % (len(it_synth), n_empty))

    idxs = ([0, cap // 4, cap // 2, (3 * cap) // 4, cap - 1] if cap >= 5 else list(range(cap)))
    gen_samples = [{"en": en_use[i], "gen_it": it_synth[i]} for i in idxs]

    payload = {
        "config": {
            "dataset": "toxi_clean (FredZhang7/toxi-text-3M, leakage-corrected)",
            "task": "binary is_toxic", "generator": args.generator,
            "prompt": "paraphrase_prompts.build_messages('crosslingual', src='English', tgt='Italian')",
            "gen_batch": GEN_BATCH, "gen_max_new": GEN_MAX_NEW, "decode": "greedy",
            "en_cap": int(cap), "seed": int(args.seed),
            "n_en_train": int(n_en), "n_it_train": int(len(y_it_tr)),
            "n_it_val": int(len(y_it_va)), "n_it_test": int(len(y_it_te)),
            "it_test_pos_rate": float(y_it_te.mean()),
            "feature": "RAW EuroBERT mean (bf16 forward -> fp32), emb_bf16/toxic/<bb>/",
            "xgb": XGB_PARAMS, "device": _DEVICE, "smoke": _SMOKE,
            "python": sys.version.split()[0],
        },
        "gen_samples": gen_samples,
        "backbones": {},
    }

    # load the stored real/EN embeddings once per backbone (shared across arms)
    for bb in bks:
        write_status({"phase": "backbone %s" % bb, "device": _DEVICE})
        en_tr_emb = load_stored(bb, "en", "train")[rows_used]
        it_tr_emb = load_stored(bb, "it", "train")
        it_va_emb = load_stored(bb, "it", "validation")
        it_te_emb = load_stored(bb, "it", "test")
        assert en_tr_emb.shape[0] == cap
        assert it_te_emb.shape[0] == len(y_it_te)
        res = run_backbone(bb, en_tr_emb, y_en_tr, it_tr_emb, y_it_tr,
                           it_va_emb, y_it_va, it_te_emb, y_it_te, it_synth, smoke=_SMOKE)
        payload["backbones"][bb] = res

    # verdict from the primary (first) backbone
    primary = bks[0]
    arms = payload["backbones"][primary]["arms"]
    s, c = arms["synth_it"]["f1_pos"], arms["real_it"]["f1_pos"]
    e, x = arms["en_transfer"]["f1_pos"], arms["mixed"]["f1_pos"]
    ratio = s / c if c else float("nan")
    if ratio >= 0.9:
        verdict = ("On EuroBERT-%s, synthetic-Italian training reaches **%.0f%%** of the "
                   "real-Italian ceiling (F1 %.3f vs %.3f) -- it **approaches** the ceiling."
                   % (primary, ratio * 100, s, c))
    elif ratio >= 0.7:
        verdict = ("On EuroBERT-%s, synthetic-Italian training reaches **%.0f%%** of the "
                   "real-Italian ceiling (F1 %.3f vs %.3f) -- close, but a clear gap remains."
                   % (primary, ratio * 100, s, c))
    else:
        verdict = ("On EuroBERT-%s, synthetic-Italian training reaches only **%.0f%%** of the "
                   "real-Italian ceiling (F1 %.3f vs %.3f) -- a large gap remains."
                   % (primary, ratio * 100, s, c))
    verdict += (" It changes EN-transfer by %+.3f F1 (%.3f) and mixing EN + synthetic IT gives "
                "%.3f (%+.3f vs ceiling, %+.3f vs EN-transfer)."
                % (s - e, e, x, x - c, x - e))
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
