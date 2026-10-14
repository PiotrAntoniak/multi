"""gen_compare.py -- compare THREE small instruct LLM generators for English->Italian
synthetic TRAINING text, on the SAME downstream classifier, for TWO datasets.

Goal
----
For each dataset we (1) generate Italian training text from the ENGLISH training
text with THREE small instruct LLMs, using the cross-lingual rewrite template from
``paraphrase_prompts.py`` / ``PARAPHRASE_TEMPLATE.md``; (2) embed the synthetic
Italian with EuroBERT **mean**; (3) train the SAME XGBoost classifier on each
generator's synthetic Italian; (4) evaluate on the **REAL Italian** test.  We also
report two reference rows: ``real_it`` (train on real Italian -> ceiling) and
``en_transfer`` (train on the English source -> cross-lingual floor).

Generators (bf16, greedy, chat template, ``enable_thinking=False`` when supported)
----------------------------------------------------------------------------------
* ``Qwen/Qwen3.5-0.8B``
* ``Qwen/Qwen3.5-2B``
* ``HuggingFaceTB/SmolLM2-1.7B-Instruct``   (FinePhrase's best ~1B family)

Datasets
--------
1. **FLORES sent** (12-class): English train+val (997+504=1501) -> Italian; evaluate on
   the real Italian test (505).  English source = the ``en`` column of those rows;
   real Italian reference = the ``it`` column (parallel) for eyeballing generation.
2. **Toxicity** (``data/toxi_clean``, binary ``is_toxic``): English train capped to
   ``--en-cap`` (default 8000, seed 0) -> Italian; evaluate on the real Italian test
   (2310).

Classifier
----------
EuroBERT-**210m** mean embeddings -> XGBoost.  FLORES: ``multi:softprob`` (12-class);
toxic: ``binary:logistic``.  hist, max_bin 64, cpu, n_jobs 2, seed 0, depth 8, mcw 5,
lr 0.1, n 400.  610m is optional (``--backbones 210m,610m``).  210m is primary.

Feature pools (match the stored caches so synthetic and real vectors share a space)
-----------------------------------------------------------------------------------
* FLORES: fp32 forward, plain tokenization truncated to 511, masked mean
  (== ``extract_embeddings.py`` mean; read from ``embeddings[/_610]/mean/emb_<lang>.npy``).
* toxic: bf16 forward, max_length 512, masked mean
  (== ``backbone_grid._embed_mean``; read from ``emb_bf16/toxic/<bb>/emb_<lang>_<split>.npy``).

Process model
-------------
Generators run **sequentially, one on GPU at a time** (load -> generate -> free ->
next).  Generation is cached per (generator, dataset).  A GPU free-memory guard
(``--wait-gpu-mb``) lets the run survive a concurrently busy GPU.  Long run is
launched DETACHED (hidden) with PID + log + status files; nothing outside the four
``gen_compare.*`` outputs and the ``%TEMP%\\opencode\\gen_compare`` cache is written.

Usage
-----
    python gen_compare.py                 # foreground
    python gen_compare.py --launch        # detached hidden run via launch_hidden.vbs
    python gen_compare.py --smoke         # tiny plumbing check (no outputs written)
    python gen_compare.py --datasets flores --generators Qwen/Qwen3.5-0.8B
    python gen_compare.py --from-json     # rebuild .md/.png from gen_compare.json
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

import train_flores12 as T                                      # noqa: E402
from paraphrase_prompts import build_messages as _build_msgs     # noqa: E402

OUT_JSON = os.path.join(REPO, "gen_compare.json")
OUT_MD = os.path.join(REPO, "gen_compare.md")
OUT_PNG = os.path.join(REPO, "gen_compare.png")

TEMP = os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp")
OPCODE = os.path.join(TEMP, "opencode")
CACHE = os.path.join(OPCODE, "gen_compare")
os.makedirs(CACHE, exist_ok=True)
LOG_PATH = os.path.join(CACHE, "gen_compare.log")
PID_PATH = os.path.join(CACHE, "gen_compare.pid")
STATUS_PATH = os.path.join(CACHE, "gen_compare_status.json")

DATA_DIR = os.path.join(REPO, "data", "toxi_clean")

# ------------------------------------------------------------------ configuration
GENERATORS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
GEN_SHORT = {
    "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B": "Qwen3.5-2B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
}

# id -> (EuroBERT model id, FLORES mean root, toxic mean root, hidden size)
BACKBONES = {
    "210m": {"model": "EuroBERT/EuroBERT-210m", "flores_dir": "embeddings",
             "toxic_dir": "emb_bf16/toxic/210m", "hidden": 768},
    "610m": {"model": "EuroBERT/EuroBERT-610m", "flores_dir": "embeddings_610",
             "toxic_dir": "emb_bf16/toxic/610m", "hidden": 1152},
}
DEFAULT_BACKBONES = ["210m"]
DATASETS = ["flores", "toxic"]

SRC_LANG = "English"
TGT_LANG = "Italian"

GEN_BATCH = 16            # fixed batch (step down only if OOM)
GEN_MAX_NEW = 384
EMB_BATCH = 64

# Batch is fixed at 16 (largest candidate to START is 16); step down only on OOM.
BATCH_CANDIDATES = [32, 24, 16, 12, 8, 4, 1]
START_BATCH = 16
DUTY = 1.0                # 1.0 = NO duty cycle / no sleeping (run at full speed)
MEM_FRACTION = 1.0        # 1.0 = NO per-process VRAM cap (use as much GPU memory as needed)

# FLORES: fp32, plain tokenization truncated to 511 (matches extract_embeddings.py mean).
FLORES_EMB_MAX_LEN = 511
# toxic: bf16, max_length 512 (matches backbone_grid._embed_mean).
TOXIC_EMB_MAX_LEN = 512

XGB_PARAMS = dict(tree_method="hist", max_bin=64, device="cpu", n_jobs=2, seed=0,
                  max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)

DEFAULT_EN_CAP = 8000
DEFAULT_SEED = 0

BASELINES = ["real_it", "en_transfer"]

_T0 = time.time()
_SMOKE = False
_DEVICE = "cpu"
_WAIT_GPU_MB = 7000
_WAIT_GPU_SEC = 7200

_COMPILE = True           # torch.compile the generator (reduce-overhead, dynamic)
_BENCH_COMPILE = True     # measure eager vs compiled throughput once per generator
_BENCH_N = 16
_DUTY = DUTY              # GPU duty cycle for generation (1.0 = no throttle)
_MEM_FRACTION = MEM_FRACTION
_BELOW_NORMAL = True      # lower process priority so the desktop stays responsive
_GEN_ONLY = False         # generation-only: stop after generating (no embed/train/outputs)

_GEN_MODEL = {}
_EMB_MODEL = {}
_GEN_STATS = {}           # (dataset, generator) -> {batch, rows_per_s, tok_per_s}
_BENCHED = set()          # generators already micro-benchmarked


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
def mc_metrics(y_true, probs, C):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(probs).argmax(1)
    labels = list(range(C))
    out = {"accuracy": float(accuracy_score(y_true, pred)),
           "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels,
                                      zero_division=0))}
    try:
        out["auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                         average="macro", labels=labels))
    except Exception:
        out["auc"] = float("nan")
    return out


def binary_metrics(y_true, probs, thr=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(probs, dtype=float)
    pred = (s >= thr).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "auc": float(roc_auc_score(y, s)),
            "threshold": float(thr)}


# ------------------------------------------------------------------ generation
def _wait_for_gpu(min_free_mb, timeout_s):
    """Poll until the GPU has >= min_free_mb free (survives a concurrently busy GPU)."""
    if _DEVICE != "cuda":
        return
    import torch
    t0 = time.time()
    warned = False
    while True:
        free_b, _total_b = torch.cuda.mem_get_info()
        free_mb = free_b / 1024.0 / 1024.0
        if free_mb >= min_free_mb:
            if warned:
                log("  gpu free now %.0f MB (>= %d) -> proceed" % (free_mb, min_free_mb))
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError("GPU wait timed out after %ds (free %.0f MB)"
                               % (timeout_s, free_mb))
        if not warned:
            log("  waiting for GPU: %.0f MB free < %d MB required" % (free_mb, min_free_mb))
            warned = True
        time.sleep(20)


def _next_batch(bs):
    """Next lower candidate below `bs` (or None if already the smallest)."""
    lower = [b for b in BATCH_CANDIDATES if b < bs]
    return max(lower) if lower else None


def _apply_gpu_throttle():
    """Cap this process's VRAM share and lower CPU priority so the desktop stays usable."""
    if _DEVICE == "cuda" and _MEM_FRACTION and _MEM_FRACTION < 1.0:
        try:
            import torch
            torch.cuda.set_per_process_memory_fraction(float(_MEM_FRACTION), 0)
            log("  VRAM cap: set_per_process_memory_fraction(%.2f)" % _MEM_FRACTION)
        except Exception as e:
            log("  set_per_process_memory_fraction failed: %r" % e)
    if _BELOW_NORMAL:
        try:
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL_PRIORITY_CLASS):
                log("  process priority set to BELOW_NORMAL")
            else:
                log("  SetPriorityClass returned 0")
        except Exception as e:
            try:                                              # fallback via psutil if present
                import psutil
                psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                log("  process priority set to BELOW_NORMAL (psutil)")
            except Exception as e2:
                log("  could not lower process priority: %r / %r" % (e, e2))


def _gpu_mem():
    """Return {alloc_mb, reserved_mb, free_mb} for the CUDA device (or {} on CPU)."""
    try:
        import torch
        if _DEVICE == "cuda" and torch.cuda.is_initialized():
            free_b, _tot = torch.cuda.mem_get_info()
            return {"alloc_mb": torch.cuda.memory_allocated() / 1048576.0,
                    "reserved_mb": torch.cuda.memory_reserved() / 1048576.0,
                    "free_mb": free_b / 1048576.0}
    except Exception:
        pass
    return {}


def _mem_str(m):
    if not m:
        return "n/a"
    return "alloc=%.0fMB reserved=%.0fMB free=%.0fMB" % (m["alloc_mb"], m["reserved_mb"],
                                                        m["free_mb"])


def _evict_others(loading):
    """Enforce single-model residency before a load (never generator + embedder together)."""
    if loading == "generator" and _EMB_MODEL:
        log("  residency: embedder resident -> evicting before generator load")
        _free_emb_models()
    if loading == "embedder" and _GEN_MODEL:
        for g in list(_GEN_MODEL):
            log("  residency: generator %s resident -> evicting before embedder load"
                % GEN_SHORT.get(g, g))
            _free_gen_model(g)


def _get_gen_model(generator):
    if generator not in _GEN_MODEL:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _evict_others("generator")
        _wait_for_gpu(_WAIT_GPU_MB, _WAIT_GPU_SEC)
        log("  GPU before loading generator %s: %s" % (GEN_SHORT.get(generator, generator),
                                                       _mem_str(_gpu_mem())))
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(generator)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            generator, torch_dtype=torch.bfloat16 if _DEVICE == "cuda" else torch.float32)
        base = base.to(_DEVICE).eval()
        try:
            base.generation_config.pad_token_id = tok.pad_token_id
        except Exception:
            pass
        entry = {"tok": tok, "eager": base, "run": base, "compiled": False}
        if _COMPILE and _DEVICE == "cuda":
            try:
                entry["run"] = torch.compile(base, mode="reduce-overhead", dynamic=True)
                entry["compiled"] = True
                log("  torch.compile applied to %s (mode=reduce-overhead, dynamic=True)"
                    % generator)
            except Exception as e:
                entry["run"] = base
                entry["compiled"] = False
                log("  torch.compile unavailable for %s: %r" % (generator, e))
        _GEN_MODEL[generator] = entry
        log("  loaded generator %s on %s (bf16=%s); resident models: 1 (gen=%s, emb=%d); %s"
            % (generator, _DEVICE, _DEVICE == "cuda", GEN_SHORT.get(generator, generator),
               len(_EMB_MODEL), _mem_str(_gpu_mem())))
    return _GEN_MODEL[generator]


def _free_gen_model(generator):
    if generator in _GEN_MODEL:
        before = _gpu_mem()
        entry = _GEN_MODEL.pop(generator)
        del entry
        if _DEVICE == "cuda":
            import torch
            torch.cuda.empty_cache()
        log("  freed generator %s; %s -> %s"
            % (GEN_SHORT.get(generator, generator), _mem_str(before), _mem_str(_gpu_mem())))


def _bench_compile(generator, prompts):
    """One-time measured comparison of eager vs compiled throughput (same prompts)."""
    import torch
    entry = _GEN_MODEL[generator]
    tok = entry["tok"]
    sample = prompts[:min(_BENCH_N, len(prompts))]

    def _run(model):
        enc = tok(sample, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(_DEVICE)
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            g = model.generate(**enc, do_sample=False, num_beams=1,
                               max_new_tokens=GEN_MAX_NEW, pad_token_id=tok.pad_token_id)
        return int(g[:, in_len:].numel())

    try:
        t = time.time()
        nt = _run(entry["eager"])
        dt = time.time() - t
        erps, etps = len(sample) / dt, nt / dt
    except Exception as e:
        log("  BENCH %s eager failed: %r" % (generator, e))
        return
    log("  BENCH %s eager: %.2f rows/s, %.0f tok/s (n=%d)"
        % (GEN_SHORT.get(generator, generator), erps, etps, len(sample)))
    if entry["compiled"]:
        try:
            _run(entry["run"])                      # compile + cudagraph warmup
            t = time.time()
            nt2 = _run(entry["run"])
            dt2 = time.time() - t
            crps, ctps = len(sample) / dt2, nt2 / dt2
            log("  BENCH %s compiled: %.2f rows/s, %.0f tok/s (n=%d)"
                % (GEN_SHORT.get(generator, generator), crps, ctps, len(sample)))
            _GEN_STATS.setdefault(generator, {})["bench"] = {
                "eager_rows_per_s": erps, "eager_tok_per_s": etps,
                "compiled_rows_per_s": crps, "compiled_tok_per_s": ctps,
                "n": len(sample)}
            if crps < erps:
                log("  BENCH %s compiled is slower (%.2f < %.2f rows/s); keeping compiled "
                    "(toggle with --no-compile)" % (GEN_SHORT.get(generator, generator),
                                                    crps, erps))
        except Exception as e:
            entry["run"] = entry["eager"]
            entry["compiled"] = False
            if _DEVICE == "cuda":
                import torch as _t
                _t.cuda.empty_cache()
            log("  BENCH %s compiled failed (%r); fell back to eager"
                % (GEN_SHORT.get(generator, generator), e))
    _GEN_STATS.setdefault(generator, {})["bench_eager_rows_per_s"] = erps


def _build_prompts(tokenizer, texts):
    prompts = []
    for t in texts:
        messages = _build_msgs("crosslingual", text=str(t), src=SRC_LANG, tgt=TGT_LANG)
        try:
            p = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        except TypeError:
            p = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        prompts.append(p)
    return prompts


_THINK_RE = re.compile(r" thinking.*?</think>", re.DOTALL)


def _clean(s):
    s = _THINK_RE.sub(" ", s)
    if " thinking" in s:
        s = s.rsplit(" thinking", 1)[-1]
    s = s.replace("</think>", " ").replace("<|im_end|>", " ")
    s = s.replace("|im_end|", " ").strip()
    for pre in ("Italian:", "Italiano:", "Translation:", "Traduzione:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
    s = " ".join(s.split())
    return s


def _dump_partial(path, preds):
    """Atomically write a partial generation checkpoint (survives sleep/crash)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        log("  [warn] partial checkpoint write failed: %r" % e)


def generate_italian(texts, generator, dataset):
    """Generate Italian for `texts` with `generator`; cache outputs on disk (resumable).

    Adaptive batching: start at ``START_BATCH`` and step down the ``BATCH_CANDIDATES``
    ladder on ``torch.cuda.OutOfMemoryError`` (clearing the cache between attempts); a
    compiled-model failure (non-OOM) falls back to eager.  When ``_DUTY < 1`` the loop
    sleeps between batches (duty cycle).  **Robustness:** a partial checkpoint is flushed
    every few batches and on ANY interruption (machine sleep, GPU drop, kill), and a
    re-run resumes from it; the final cache is written only on completion.
    """
    cpath = os.path.join(CACHE, "gen_%s_%s_%d_%s.json"
                         % (generator.replace("/", "__"), dataset, len(texts),
                            _hash([generator, dataset] + list(texts))))
    ppath = cpath + ".partial.json"
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            out = json.load(f)
        log("  [cache] generations %s (%d lines)" % (os.path.basename(cpath), len(out)))
        return out

    preds = []
    start = 0
    if os.path.exists(ppath):                       # resume an interrupted run
        try:
            with open(ppath, "r", encoding="utf-8") as f:
                preds = json.load(f)
            start = len(preds)
            log("  [resume] %s partial has %d/%d rows -> continue at row %d"
                % (os.path.basename(cpath), start, len(texts), start))
        except Exception:
            preds, start = [], 0

    import torch
    entry = _get_gen_model(generator)
    tok = entry["tok"]
    prompts = _build_prompts(tok, texts)
    n = len(prompts)
    if _BENCH_COMPILE and _DEVICE == "cuda" and generator not in _BENCHED:
        _BENCHED.add(generator)
        _bench_compile(generator, prompts)

    bs = START_BATCH if START_BATCH in BATCH_CANDIDATES else max(BATCH_CANDIDATES)
    i = start
    n_tok = 0
    t_work = 0.0
    t_idle = 0.0
    t_gen0 = time.time()
    try:
        while i < n:
            batch = prompts[i:i + bs]
            t_b0 = time.time()
            try:
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024).to(_DEVICE)
                in_len = enc["input_ids"].shape[1]
                with torch.inference_mode():
                    gen = entry["run"].generate(**enc, do_sample=False, num_beams=1,
                                                max_new_tokens=GEN_MAX_NEW,
                                                pad_token_id=tok.pad_token_id)
                new_tokens = gen[:, in_len:]
                preds.extend(_clean(s) for s in tok.batch_decode(new_tokens,
                                                                 skip_special_tokens=True))
                n_tok += int(new_tokens.numel())
                i += len(batch)
                t_work += time.time() - t_b0
                if (i // max(1, bs)) % 5 == 0 or i == n:
                    el = time.time() - t_gen0
                    log("    gen %s/%s %d/%d bs=%d (%.1fs, %.2f rows/s, %.0f tok/s, "
                        "gpu-busy %.2f, duty=%.2f)"
                        % (GEN_SHORT.get(generator, generator), dataset, i, n, bs, el,
                           i / el if el else 0.0, n_tok / el if el else 0.0,
                           t_work / el if el else 0.0, _DUTY))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                nxt = _next_batch(bs)
                if nxt is None:
                    raise
                log("    OOM at bs=%d -> stepping down to %d (position %d/%d)"
                    % (bs, nxt, i, n))
                bs = nxt
                continue
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    torch.cuda.empty_cache()
                    nxt = _next_batch(bs)
                    if nxt is None:
                        raise
                    log("    OOM(RuntimeError) at bs=%d -> %d (position %d/%d)"
                        % (bs, nxt, i, n))
                    bs = nxt
                    continue
                if entry["compiled"]:
                    entry["run"] = entry["eager"]
                    entry["compiled"] = False
                    if _DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    log("    compiled generate failed (%r); falling back to eager, retrying"
                        % e)
                    continue
                raise
            # checkpoint + status every few batches
            if (i // max(1, bs)) % 5 == 0 or i == n:
                _dump_partial(ppath, preds)
                write_status({"phase": "generate", "generator": generator, "dataset": dataset,
                              "rows_done": int(i), "rows_total": int(n)})
            # duty cycle (disabled at _DUTY == 1.0)
            if _DUTY < 1.0 and i < n:
                idle = (time.time() - t_b0) * (1.0 - _DUTY) / max(_DUTY, 1e-9)
                if idle > 0:
                    time.sleep(idle)
                    t_idle += idle
    except BaseException as e:                          # sleep / GPU drop / kill / error
        _dump_partial(ppath, preds)
        log("  [guard] generation interrupted at %d/%d (%r); flushed partial -> %s"
            % (i, n, e, os.path.basename(ppath)))
        try:
            write_status({"phase": "interrupted", "generator": generator, "dataset": dataset,
                          "rows_done": int(i), "rows_total": int(n)})
        except Exception:
            pass
        raise

    el = time.time() - t_gen0
    busy = t_work / el if el else 0.0
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    try:
        os.remove(ppath)
    except OSError:
        pass
    log("  [save ] generations %s (bs=%d, %.2f rows/s, %.0f tok/s, gpu-busy %.2f, "
        "duty=%.2f, idle %.0fs)"
        % (os.path.basename(cpath), bs, n / el if el else 0.0, n_tok / el if el else 0.0,
           busy, _DUTY, t_idle))
    _GEN_STATS.setdefault(generator, {})[(dataset)] = {
        "batch": int(bs), "rows_per_s": (n / el if el else 0.0),
        "tok_per_s": (n_tok / el if el else 0.0), "n": int(n),
        "duty": float(_DUTY), "gpu_busy_fraction": float(busy),
        "wall_s": float(el)}
    return preds


# ------------------------------------------------------------------ embeddings
def _get_emb_model(model_id, dtype):
    key = (model_id, str(dtype))
    if key not in _EMB_MODEL:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _evict_others("embedder")
        _wait_for_gpu(_WAIT_GPU_MB, _WAIT_GPU_SEC)
        log("  GPU before loading embedder %s: %s" % (model_id, _mem_str(_gpu_mem())))
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        if dtype in ("bf16", "fp16") and _DEVICE == "cuda":
            # bf16 forward (wide-range 16-bit, per project directive); store fp32 arrays.
            model = model.to(torch.bfloat16)
        else:
            model = model.to(torch.float32)
        model = model.to(_DEVICE).eval()
        _EMB_MODEL[key] = (tok, model)
        log("  loaded embedder %s (hidden=%d) on %s (%s); resident models: 1 "
            "(gen=%d, emb=%d); %s"
            % (model_id, model.config.hidden_size, _DEVICE, dtype, len(_GEN_MODEL),
               len(_EMB_MODEL), _mem_str(_gpu_mem())))
    return _EMB_MODEL[key]


def _free_emb_models():
    if _EMB_MODEL:
        before = _gpu_mem()
        _EMB_MODEL.clear()
        if _DEVICE == "cuda":
            import torch
            torch.cuda.empty_cache()
        log("  freed embedder(s); %s -> %s" % (_mem_str(before), _mem_str(_gpu_mem())))


def embed_mean(texts, model_id, dtype, max_len):
    """Masked mean over the last hidden state (length-sorted, then restored)."""
    import torch
    tok, model = _get_emb_model(model_id, dtype)
    n = len(texts)
    if n == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    order = np.argsort(np.array([len(t) for t in texts], dtype=np.int64), kind="stable")
    sorted_texts = [texts[i] for i in order]
    acc = []
    with torch.inference_mode():
        for i in range(0, n, EMB_BATCH):
            batch = tok(sorted_texts[i:i + EMB_BATCH], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
            ids, mask = batch["input_ids"].to(_DEVICE), batch["attention_mask"].to(_DEVICE)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(h.dtype)
            acc.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
                       .astype(np.float32))
    e_sorted = np.concatenate(acc, 0).astype(np.float32)
    out = np.empty_like(e_sorted)
    out[order] = e_sorted
    return out


def ensure_synth_emb(texts, bb, dataset, model_id, dtype, max_len):
    tag = "synthemb_%s_%s_%d_%s" % (bb, dataset, len(texts), _hash(texts))
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        log("  [cache] synthetic-IT embeddings %s" % tag)
        return np.load(path)
    log("  embedding synthetic IT %s (%d texts)" % (tag, len(texts)))
    emb = embed_mean(texts, model_id, dtype, max_len)
    np.save(path, emb)
    return emb


# ------------------------------------------------------------------ samples / ranking
def pick_samples(d, gen_it, dataset, k=5):
    n = len(d["en_src"])
    idxs = ([0, n // 4, n // 2, (3 * n) // 4, n - 1] if n >= k else list(range(n)))
    out = []
    for i in idxs[:k]:
        e = {"en": d["en_src"][i], "gen_it": gen_it[i]}
        if dataset == "flores":
            e["ref_it"] = d["it_ref"][i]
        out.append(e)
    return out


PRIMARY = {"flores": ("macro_f1", "macro-F1"), "toxic": ("f1", "F1 (positive)")}


def rank_generators(res, dataset):
    key = PRIMARY[dataset][0]
    items = [(g, res["generators"][g][key]) for g in res["generators"]
             if key in res["generators"][g] and not res["generators"][g].get("skipped")]
    items.sort(key=lambda t: -t[1])
    return items


# ------------------------------------------------------------------ report / plot
def build_md(payload):
    cfg = payload["config"]
    L = []
    L.append("# Generator comparison: English -> Italian synthetic training text")
    L.append("")
    L.append("Three small instruct LLMs generate **Italian** training text from the **English** "
             "training text with the cross-lingual rewrite template "
             "(`paraphrase_prompts.build_messages('crosslingual', src='English', tgt='Italian')`). "
             "EuroBERT-**%s** mean embeddings feed the SAME fixed XGBoost classifier, trained on "
             "each generator's synthetic Italian and evaluated on the **real Italian** test."
             % ",".join(payload["backbones"]))
    L.append("")
    L.append("- Generators: %s." % ", ".join("`%s`" % g for g in payload["generators"]))
    L.append("- Decoding: greedy (`do_sample=False`), bf16, batch %d, max_new_tokens %d, "
             "`enable_thinking=False` when supported." % (cfg["gen_batch"], cfg["gen_max_new"]))
    L.append("- XGBoost (CPU, fixed): hist, max_bin 64, cpu, n_jobs 2, seed 0, depth 8, mcw 5, "
             "lr 0.1, n 400  (FLORES `multi:softprob` 12-class; toxic `binary:logistic`).")
    L.append("- Datasets: **FLORES sent** (EN train+val 997+504=1501 -> IT; real IT test 505); "
             "**toxic** (`data/toxi_clean`; EN train capped to %d, seed %d -> IT; real IT test %d)."
             % (cfg["toxic_en_cap"], cfg["seed"], cfg["toxic_n_test"]))
    L.append("- Reference rows: **real_it** (train on real Italian = ceiling) and **en_transfer** "
             "(train on the English source = cross-lingual floor).")
    L.append("- Generation: adaptive batch from `%s` (step down on OOM); `torch.compile` "
             "(reduce-overhead, dynamic) = %s; **duty cycle = %s** (GPU busy ~that fraction "
             "of wall time); VRAM cap = %s; below-normal priority = %s."
             % (cfg.get("start_batch"), cfg.get("torch_compile"), cfg.get("duty"),
                cfg.get("mem_fraction"), cfg.get("below_normal_priority")))
    L.append("")

    # ---- generation throughput ----
    gs = payload.get("gen_stats", {})
    if gs:
        L.append("## Generation throughput (bf16)")
        L.append("")
        L.append("| generator | dataset | batch | duty | gpu-busy | rows/s | tok/s |")
        L.append("|---|---|---|---|---|---|---|")
        for g in payload["generators"]:
            st = gs.get(g, {})
            for ds in payload.get("completed_datasets", payload["datasets"]):
                s = st.get(ds)
                if s:
                    L.append("| %s | %s | %d | %.2f | %.2f | %.2f | %.0f |"
                             % (GEN_SHORT.get(g, g), ds, s["batch"],
                                s.get("duty", float("nan")),
                                s.get("gpu_busy_fraction", float("nan")),
                                s["rows_per_s"], s["tok_per_s"]))
        L.append("")
        for g in payload["generators"]:
            b = gs.get(g, {}).get("bench")
            if b:
                L.append("- **%s** micro-bench (n=%d, same prompts): eager %.2f rows/s "
                         "(%.0f tok/s) vs compiled %.2f rows/s (%.0f tok/s)."
                         % (GEN_SHORT.get(g, g), b["n"], b["eager_rows_per_s"],
                            b["eager_tok_per_s"], b["compiled_rows_per_s"],
                            b["compiled_tok_per_s"]))
        L.append("")

    for ds in payload.get("completed_datasets", payload["datasets"]):
        dss = payload["results"][ds]
        primary, plabel = PRIMARY[ds]
        L.append("## Dataset: %s%s" % (ds, " (12-class, accuracy/macro-F1/AUC)"
                                       if ds == "flores" else " (binary, accuracy/F1/AUC)"))
        L.append("")
        for bb in payload["backbones"]:
            res = dss[bb]
            m0, m1, m2 = ("accuracy", "macro_f1", "auc") if ds == "flores" else \
                         ("accuracy", "f1", "auc")
            L.append("### Backbone EuroBERT-%s (%s)" % (bb, BACKBONES[bb]["model"]))
            L.append("")
            L.append("| generator | accuracy | %s | AUC |" % plabel)
            L.append("|---|---|---|---|")
            for g in payload["generators"]:
                m = res["generators"].get(g)
                if m is None or m.get("skipped"):
                    note = m.get("skipped") if isinstance(m, dict) else None
                    L.append("| %s | _skipped%s_ | - | - |"
                             % (GEN_SHORT.get(g, g), (" (%s)" % note) if note else ""))
                    continue
                L.append("| %s | %.4f | %.4f | %.4f |"
                         % (GEN_SHORT.get(g, g), m[m0], m[m1], m[m2]))
            for b in BASELINES:
                m = res["baselines"][b]
                L.append("| %s (%s) | %.4f | %.4f | %.4f |"
                         % (b, "ceiling" if b == "real_it" else "floor", m[m0], m[m1], m[m2]))
            L.append("")
            order = rank_generators(res, ds)
            ceil = res["baselines"]["real_it"][primary]
            floor = res["baselines"]["en_transfer"][primary]
            L.append("- Ranking by %s: %s." % (plabel,
                     " > ".join("%s (%.4f)" % (GEN_SHORT.get(g, g), v) for g, v in order)))
            best_g, best_v = order[0]
            L.append("- Best generator **%s** = %.4f %s vs real-IT ceiling %.4f "
                     "(%.0f%% of ceiling) and EN-transfer floor %.4f."
                     % (GEN_SHORT.get(best_g, best_g), best_v, plabel, ceil,
                        100.0 * best_v / ceil if ceil else float("nan"), floor))
            L.append("")
        L.append("### Sample generations (%s)" % ds)
        L.append("")
        for g in payload["generators"]:
            if g not in payload["samples"].get(ds, {}):
                continue
            L.append("**%s**" % GEN_SHORT.get(g, g))
            L.append("")
            for e in payload["samples"][ds][g]:
                L.append("- EN: %s" % e["en"])
                L.append("  - IT (%s): _%s_" % (GEN_SHORT.get(g, g), e["gen_it"]))
                if "ref_it" in e:
                    L.append("  - FLORES IT ref: %s" % e["ref_it"])
            L.append("")

    L.append("![gen_compare](gen_compare.png)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    for ds in payload.get("completed_datasets", payload["datasets"]):
        L.append("- **%s**: %s" % (ds, payload["verdicts"][ds]))
    L.append("")
    return "\n".join(L)


def plot(payload, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dss = payload.get("completed_datasets", payload["datasets"])
    fig, axes = plt.subplots(1, len(dss), figsize=(7.2 * len(dss), 5.2), squeeze=False)
    gens = [g for g in payload["generators"]
            if any((payload["results"].get(d, {}).get(payload["backbones"][0], {})
                    .get("generators", {}).get(g) or {}).get(PRIMARY[d][0]) is not None
                   for d in dss)]
    for ax, ds in zip(axes[0], dss):
        bb = payload["backbones"][0]
        res = payload["results"][ds][bb]
        if ds == "flores":
            metrics = [("accuracy", "accuracy", "#4c72b0"),
                       ("macro_f1", "macro-F1", "#dd8452"),
                       ("auc", "AUC", "#55a868")]
        else:
            metrics = [("accuracy", "accuracy", "#4c72b0"),
                       ("f1", "F1 (pos)", "#dd8452"),
                       ("auc", "AUC", "#55a868")]
        gg = [g for g in gens if (res["generators"].get(g) or {}).get(metrics[0][0]) is not None]
        x = np.arange(len(gg))
        w = 0.25
        for j, (mk, label, col) in enumerate(metrics):
            vals = [res["generators"][g][mk] for g in gg]
            bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=col)
            for r in bars:
                ax.annotate("%.3f" % r.get_height(),
                            (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=6)
        # ceiling lines (real_it) per metric, dashed, same colour
        for j, (mk, label, col) in enumerate(metrics):
            c = res["baselines"]["real_it"][mk]
            ax.axhline(c, color=col, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([GEN_SHORT.get(g, g) for g in gg], rotation=20, ha="right",
                           fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title("%s (EuroBERT-%s mean, real IT test)\ndashed = real-IT ceiling"
                     % (ds, bb), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Synthetic-Italian training per generator vs real-Italian ceiling "
                 "(real Italian test)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log("wrote %s" % out_png)


# ------------------------------------------------------------------ launch / main
def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, LOG_PATH, REPO], cwd=REPO)
    print("launched detached; log=%s" % LOG_PATH)


def write_outputs(payload):
    """Write gen_compare.json/.md/.png from the payload (call incrementally)."""
    if _SMOKE:
        return
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_jd, ensure_ascii=False)
    log("wrote %s" % OUT_JSON)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(build_md(payload))
    log("wrote %s" % OUT_MD)
    plot(payload, OUT_PNG)


def _verdict_for(ds, res):
    order = rank_generators(res, ds)
    primary, plabel = PRIMARY[ds]
    ceil = res["baselines"]["real_it"][primary]
    floor = res["baselines"]["en_transfer"][primary]
    best_g, best_v = order[0]
    ratio = best_v / ceil if ceil else float("nan")
    return ("best generator **%s** reaches %.4f %s = **%.0f%%** of the real-Italian ceiling "
            "(%.4f); EN-transfer floor %.4f. Ranking: %s."
            % (GEN_SHORT.get(best_g, best_g), best_v, plabel, 100.0 * ratio, ceil, floor,
               " > ".join("%s (%.4f)" % (GEN_SHORT.get(g, g), v) for g, v in order)))


def _train_eval(bb, ds, d, gen_outs, generators):
    """Train + evaluate one dataset for one backbone: baselines once, then per generator."""
    if ds == "flores":
        f = flores_data_for(bb, d)
        res = {"generators": {}, "baselines": {}}
        res["baselines"]["real_it"] = _eval_flores(bb, f["X_it_tr"], f["y_tr"],
                                                   f["X_it_te"], f["y_te"])
        res["baselines"]["en_transfer"] = _eval_flores(bb, f["X_en_tr"], f["y_tr"],
                                                       f["X_it_te"], f["y_te"])
        res["baselines"]["en_transfer"]["n_train"] = int(len(f["y_tr"]))
        res["baselines"]["real_it"]["n_train"] = int(len(f["y_tr"]))
        for g in generators:
            Xsyn = ensure_synth_emb(gen_outs[(ds, g)], bb, ds, BACKBONES[bb]["model"],
                                    "float32", FLORES_EMB_MAX_LEN)
            mm = _eval_flores(bb, Xsyn, f["y_tr"], f["X_it_te"], f["y_te"])
            mm["n_train"] = int(len(f["y_tr"]))
            res["generators"][g] = mm
    else:
        f = toxic_data_for(bb, d)
        res = {"generators": {}, "baselines": {}}
        res["baselines"]["real_it"] = _eval_toxic(f["X_it_tr"], f["y_it_tr"],
                                                  f["X_it_te"], f["y_it_te"])
        res["baselines"]["en_transfer"] = _eval_toxic(f["X_en_tr"], f["y_en_tr"],
                                                      f["X_it_te"], f["y_it_te"])
        for g in generators:
            Xsyn = ensure_synth_emb(gen_outs[(ds, g)], bb, ds, BACKBONES[bb]["model"],
                                    "bf16", TOXIC_EMB_MAX_LEN)
            res["generators"][g] = _eval_toxic(Xsyn, f["y_en_tr"], f["X_it_te"], f["y_it_te"])
    return res


def main(argv=None):
    global _SMOKE, _DEVICE, GEN_BATCH, START_BATCH, GEN_MAX_NEW
    global _WAIT_GPU_MB, _WAIT_GPU_SEC, _COMPILE, _BENCH_COMPILE
    global _DUTY, _MEM_FRACTION, _BELOW_NORMAL, _GEN_ONLY
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Compare 3 generators for EN->IT synthetic data.")
    ap.add_argument("--generators", default=",".join(GENERATORS),
                    help="comma list of HF generator ids")
    ap.add_argument("--datasets", default=",".join(DATASETS), help="comma list from {flores,toxic}")
    ap.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                    help="comma list from {%s}" % ",".join(BACKBONES))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--en-cap", type=int, default=DEFAULT_EN_CAP, help="toxic EN train cap")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="toxic cap subsample seed")
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH,
                    help="initial generation batch (adaptive ladder may start at --start-batch)")
    ap.add_argument("--start-batch", type=int, default=START_BATCH,
                    help="largest batch to start from (default %d; steps down on OOM)"
                         % START_BATCH)
    ap.add_argument("--max-new", type=int, default=GEN_MAX_NEW)
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="disable torch.compile (reduce-overhead) of the generator")
    ap.add_argument("--no-bench-compile", dest="bench_compile", action="store_false",
                    help="skip the eager-vs-compiled micro-benchmark")
    ap.add_argument("--duty", type=float, default=DUTY,
                    help="GPU duty cycle for generation, 0<duty<=1 (default %.2f; 1.0=no "
                         "throttle). Sleeps between batches so GPU is busy ~duty of wall "
                         "time." % DUTY)
    ap.add_argument("--mem-fraction", type=float, default=MEM_FRACTION,
                    help="torch.cuda.set_per_process_memory_fraction cap (default %.2f; "
                         "1.0=no cap)" % MEM_FRACTION)
    ap.add_argument("--no-below-normal", dest="below_normal", action="store_false",
                    help="do not lower the process priority")
    ap.set_defaults(compile=True, bench_compile=True, below_normal=True)
    ap.add_argument("--gen-limit", type=int, default=0, help="limit generations per dataset (0=all)")
    ap.add_argument("--wait-gpu-mb", type=int, default=_WAIT_GPU_MB,
                    help="min free GPU MiB before loading a model (default %d)" % _WAIT_GPU_MB)
    ap.add_argument("--wait-gpu-sec", type=int, default=_WAIT_GPU_SEC,
                    help="max seconds to wait for GPU memory (default %d)" % _WAIT_GPU_SEC)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="no-op: completed caches and partial checkpoints are reused "
                         "automatically")
    ap.add_argument("--gen-only", dest="gen_only", action="store_true",
                    help="generate only: stop after generation (do not embed/train/write outputs)")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--from-json", action="store_true")
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
        passthrough = ["--generators", args.generators, "--datasets", args.datasets,
                       "--backbones", args.backbones, "--en-cap", str(args.en_cap),
                       "--seed", str(args.seed), "--gen-batch", str(args.gen_batch),
                       "--start-batch", str(args.start_batch),
                       "--max-new", str(args.max_new), "--wait-gpu-mb", str(args.wait_gpu_mb),
                       "--wait-gpu-sec", str(args.wait_gpu_sec)]
        if not args.compile:
            passthrough += ["--no-compile"]
        if not args.bench_compile:
            passthrough += ["--no-bench-compile"]
        passthrough += ["--duty", str(args.duty), "--mem-fraction", str(args.mem_fraction)]
        if not args.below_normal:
            passthrough += ["--no-below-normal"]
        if args.gen_only:
            passthrough += ["--gen-only"]
        if args.device != "auto":
            passthrough += ["--device", args.device]
        if args.gen_limit:
            passthrough += ["--gen-limit", str(args.gen_limit)]
        launch_detached(passthrough)
        return 0

    _SMOKE = args.smoke
    GEN_BATCH = args.gen_batch
    START_BATCH = args.start_batch
    GEN_MAX_NEW = args.max_new
    _WAIT_GPU_MB = args.wait_gpu_mb
    _WAIT_GPU_SEC = args.wait_gpu_sec
    _COMPILE = args.compile
    _BENCH_COMPILE = args.bench_compile
    _DUTY = args.duty
    _MEM_FRACTION = args.mem_fraction
    _BELOW_NORMAL = args.below_normal
    _GEN_ONLY = args.gen_only
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bks = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit("unknown dataset %r (choose from %r)" % (d, DATASETS))
    for b in bks:
        if b not in BACKBONES:
            raise SystemExit("unknown backbone %r (choose from %r)" % (b, list(BACKBONES)))

    import torch
    _DEVICE = args.device
    if _DEVICE == "auto":
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _apply_gpu_throttle()

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    log("start pid=%d device=%s smoke=%s duty=%.2f mem_frac=%.2f gen_only=%s alloc_conf=%s "
        "generators=%s datasets=%s "
        "backbones=%s" % (os.getpid(), _DEVICE, _SMOKE, _DUTY, _MEM_FRACTION, _GEN_ONLY,
                          os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"), generators,
                          datasets, bks))
    write_status({"phase": "start", "device": _DEVICE, "generators": generators,
                  "datasets": datasets, "backbones": bks})

    payload = {
        "config": {
            "generators": generators, "datasets": datasets, "backbones": bks,
            "prompt": "paraphrase_prompts.build_messages('crosslingual', src='English', "
                      "tgt='Italian')",
            "src_lang": SRC_LANG, "tgt_lang": TGT_LANG, "decode": "greedy",
            "gen_batch": GEN_BATCH, "start_batch": START_BATCH,
            "batch_candidates": BATCH_CANDIDATES,
            "torch_compile": bool(_COMPILE), "compile_mode": "reduce-overhead (dynamic)",
            "duty": float(_DUTY), "mem_fraction": float(_MEM_FRACTION),
            "below_normal_priority": bool(_BELOW_NORMAL),
            "gen_max_new": GEN_MAX_NEW, "seed": int(args.seed),
            "toxic_en_cap": int(args.en_cap), "xgb": XGB_PARAMS,
            "device": _DEVICE, "smoke": _SMOKE, "python": sys.version.split()[0],
        },
        "generators": generators, "datasets": datasets, "backbones": bks,
        "samples": {}, "results": {}, "verdicts": {}, "gen_stats": {},
    }

    # ---- data (backbone-independent structure; embeddings are per-backbone) ----
    data = {}
    if "flores" in datasets:
        write_status({"phase": "data flores"})
        # FLORES structure is backbone-independent; embeddings differ per backbone.
        sent = T.prepare_sentence()
        y = np.asarray(sent["y"], dtype=int)
        tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
        trva = np.concatenate([tr, va])
        cur = sent["cur"]
        en_src = cur.iloc[trva]["en"].astype(str).tolist()
        it_ref = cur.iloc[trva]["it"].astype(str).tolist()
        if _SMOKE:
            rows, seen = [], {}
            for r in trva:
                c = int(y[r])
                if seen.get(c, 0) < 4:
                    rows.append(int(r)); seen[c] = seen.get(c, 0) + 1
            trva = np.asarray(rows, dtype=int)
            en_src = cur.iloc[trva]["en"].astype(str).tolist()
            it_ref = cur.iloc[trva]["it"].astype(str).tolist()
        elif args.gen_limit:
            trva = trva[:args.gen_limit]
            en_src = cur.iloc[trva]["en"].astype(str).tolist()
            it_ref = cur.iloc[trva]["it"].astype(str).tolist()
        data["flores"] = {"y": y, "trva": trva, "te": te, "en_src": en_src, "it_ref": it_ref,
                          "sent": sent, "n_train": int(len(tr)), "n_val": int(len(va)),
                          "n_test": int(len(te)), "classes": sent["classes"]}
        payload["config"]["flores_n_test"] = int(len(te))

    if "toxic" in datasets:
        write_status({"phase": "data toxic"})
        import pandas as pd

        def read_split(lang, split):
            df = pd.read_parquet(os.path.join(DATA_DIR, "%s_%s.parquet" % (lang, split)))
            return df["text"].astype(str).tolist(), df["is_toxic"].astype(int).to_numpy()

        en_tr_text, y_en_all = read_split("en", "train")
        _, y_it_tr = read_split("it", "train")
        _, y_it_te = read_split("it", "test")
        n_en = len(y_en_all)
        cap = n_en if _SMOKE else min(args.en_cap, n_en)
        if _SMOKE:
            cap = min(64, n_en)
        rng = np.random.default_rng(args.seed)
        rows_used = np.sort(rng.choice(n_en, size=cap, replace=False))
        if args.gen_limit:
            rows_used = rows_used[:args.gen_limit]
        en_use = [en_tr_text[i] for i in rows_used]
        data["toxic"] = {"en_src": en_use, "rows_used": rows_used, "y_en_all": y_en_all,
                         "y_it_tr": y_it_tr, "y_it_te": y_it_te, "n_en": n_en,
                         "en_cap": int(len(rows_used)), "n_it_train": int(len(y_it_tr)),
                         "n_it_test": int(len(y_it_te))}
        payload["config"]["toxic_en_cap"] = int(len(rows_used))
        payload["config"]["toxic_n_test"] = int(len(y_it_te))

    # ---- dataset-major pipeline: FLORES fully, then toxicity ----
    # for dataset in datasets:  (generate ALL generators) -> (embed + train + evaluate)
    # single model resident at a time; write partial outputs after each dataset.
    payload["completed_datasets"] = []
    gen_outs = {}       # (dataset, generator) -> list[str]
    for ds in datasets:
        data_ds = data[ds]
        log("######## DATASET %s (%s-first ordering: %s) ########"
            % (ds, datasets[0], "FLORES-first" if datasets[0] == "flores" else "as-listed"))

        # 1) generate Italian for EVERY generator on this dataset (one model at a time)
        for g in generators:
            write_status({"phase": "generate", "generator": g, "dataset": ds,
                          "order": "dataset-major"})
            texts = data_ds["en_src"]
            log("generating %s: dataset=%s n=%d" % (GEN_SHORT.get(g, g), ds, len(texts)))
            out = generate_italian(texts, g, ds)
            if len(out) != len(texts):
                raise ValueError("generation length %d != requested %d" % (len(out), len(texts)))
            ne = sum(1 for s in out if not s.strip())
            log("  %s/%s: %d rows (%d empty)" % (GEN_SHORT.get(g, g), ds, len(out), ne))
            gen_outs[(ds, g)] = out
            _free_gen_model(g)          # one model resident at a time
        _free_emb_models()

        if _GEN_ONLY:                   # generation-only mode: stop before embed/train/outputs
            write_status({"phase": "generated", "dataset": ds, "gen_only": True})
            log("GEN-ONLY: generations for %s complete; skipping embed/train/outputs" % ds)
            continue

        # samples for this dataset
        payload["samples"][ds] = {}
        for g in generators:
            payload["samples"][ds][g] = pick_samples(
                {"en_src": data_ds["en_src"], "it_ref": data_ds.get("it_ref", [])},
                gen_outs[(ds, g)], ds, k=5)

        # 2) embed synthetic + train + evaluate for this dataset (embedder one at a time)
        for bb in bks:
            write_status({"phase": "train", "backbone": bb, "dataset": ds})
            log("==== backbone %s dataset %s ====" % (bb, ds))
            res = _train_eval(bb, ds, data_ds, gen_outs, generators)
            payload["results"].setdefault(ds, {})[bb] = res
            for g in generators:
                m = res["generators"][g]
                log("  [%s/%s] %-14s %s" % (bb, ds, GEN_SHORT.get(g, g),
                    ", ".join("%s=%.4f" % (k, m[k]) for k in ("accuracy", PRIMARY[ds][0], "auc"))))
            for b in BASELINES:
                m = res["baselines"][b]
                log("  [%s/%s] %-14s %s" % (bb, ds, b,
                    ", ".join("%s=%.4f" % (k, m[k]) for k in ("accuracy", PRIMARY[ds][0], "auc"))))
            _free_emb_models()

        # verdict + mark complete
        payload["verdicts"][ds] = _verdict_for(ds, payload["results"][ds][bks[0]])
        payload["completed_datasets"].append(ds)
        payload["gen_stats"] = {g: _GEN_STATS.get(g, {}) for g in generators}
        log("VERDICT %s: %s" % (ds, payload["verdicts"][ds]))

        # incremental outputs so the FLORES table is available before toxic finishes
        write_outputs(payload)
        write_status({"phase": "done %s" % ds, "completed": list(payload["completed_datasets"]),
                      "verdict": payload["verdicts"][ds]})

    write_status({"phase": "done", "completed": list(payload["completed_datasets"]),
                  "verdicts": payload["verdicts"]})
    return 0


# ---- backbone-specific eval wrappers (kept separate for clarity) ----
def flores_data_for(bb, d):
    cfg = BACKBONES[bb]
    root = os.path.join(REPO, cfg["flores_dir"])
    n_source = len(d["sent"]["git"])
    kept = d["sent"]["kept"]
    emb = {}
    for lg in ("en", "it"):
        arr = np.load(os.path.join(root, "mean", "emb_%s.npy" % lg))
        if arr.shape[0] != n_source:
            raise ValueError("expected %d rows in %s, got %s" % (n_source, root, arr.shape))
        emb[lg] = arr[kept].astype(np.float32)
    trva, te, y = d["trva"], d["te"], d["y"]
    return {"X_en_tr": emb["en"][trva], "X_it_tr": emb["it"][trva], "X_it_te": emb["it"][te],
            "y_tr": y[trva], "y_te": y[te]}


def toxic_data_for(bb, d):
    cfg = BACKBONES[bb]
    rows_used = d["rows_used"]

    def load_stored(lang, split):
        path = os.path.join(REPO, cfg["toxic_dir"], "emb_%s_%s.npy" % (lang, split))
        return np.asarray(np.load(path), dtype=np.float32)

    return {"X_en_tr": load_stored("en", "train")[rows_used],
            "X_it_tr": load_stored("it", "train"),
            "X_it_te": load_stored("it", "test"),
            "y_en_tr": d["y_en_all"][rows_used],
            "y_it_tr": d["y_it_tr"], "y_it_te": d["y_it_te"]}


def _eval_flores(bb, Xtr, ytr, Xte, yte):
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    if _SMOKE:
        p["n_estimators"] = 40
    p.update(objective="multi:softprob", num_class=12)
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xtr, ytr)
    return mc_metrics(yte, clf.predict_proba(Xte), 12)


def _eval_toxic(Xtr, ytr, Xte, yte):
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    if _SMOKE:
        p["n_estimators"] = 40
    p.update(objective="binary:logistic")
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xtr, ytr)
    return binary_metrics(yte, clf.predict_proba(Xte)[:, 1], 0.5)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        write_status({"phase": "error"})
        raise
