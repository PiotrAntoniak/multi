"""paraphrase_translate.py -- paraphrase (and translate) a CSV column with a local LLM.

One CLI, two backends -- picked by --backend auto|gguf|hf (auto: inferred from --model):
gguf = llama_cpp.Llama on a quantised .gguf; hf = transformers AutoModelForCausalLM on a
safetensors dir / HF id.

Mode from --src / --tgt / --translate-only (paraphrase first, then translate):
  --src != --tgt   paraphrase each input xN, translate every paraphrase -> N rows/input
  --src == --tgt   paraphrase xN in that language -> N rows/input
  --translate-only translate each input once -> exactly 1 row/input (needs --src != --tgt)

Templates are two-step so ONE KV prefix serves every language: a language-agnostic HEAD
(system + instructions) is encoded once and reused; only the per-row TAIL (language + text)
is evaluated.  --prefix-cache PATH persists that prefix (GGUF: pickled LlamaState; HF:
torch.saved cache) and reloads it to skip prefill.  Each save also writes metadata (sidecar
<cache>.meta.json + a row in --kvcache-meta, default kvcache_meta.csv); a later run verifies
backend, model, weights_type, n_ctx, step and prefix_sha1 and re-prefills on any mismatch
(or missing/corrupt cache) instead of reusing a stale cache -- it never crashes.

Output CSV = input columns + <tgt>_text + variant (0..N-1).  CPU by default; pass
--n-gpu-layers -1 (GGUF) or --device cuda (HF) for GPU.  Decode params are per step
(--para-* / --trans-*; --temp aliases --para-temp).

Usage:
  python paraphrase_translate.py --model models/Qwen3.5-2B-Q4_K_M.gguf --in-csv en.csv --out-csv pl.csv --tgt Polish
  python paraphrase_translate.py --backend hf --model Qwen/Qwen3.5-2B --device cuda --dtype bf16 --in-csv en.csv --out-csv it.csv --encoding latin-1
  python paraphrase_translate.py ... --prefix-cache cache.state
"""
import copy
import csv
import ctypes
import hashlib
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import click

# Templates: HEAD is language-agnostic (encoded once, KV-cached); TAIL holds language + text.
XLING_SYSTEM = (
    "You are a careful multilingual translator. You rewrite the user's text into a requested "
    "target language while preserving its meaning exactly. You never add, drop, summarise, or "
    "explain information. You output only the translated text, with no quotes or commentary."
)
XLING_HEAD = (
    "Rewrite the text below into the requested target language.\n"
    "- Preserve ALL factual content and entities (names, numbers, dates).\n"
    "- Do not add or remove information.\n"
    "- Do not replace proper nouns with different entities.\n"
    "- Output ONLY the translated text. No preamble, no quotes."
)
XLING_TAIL = (
    "Target language: {tgt}. Source language: {src}.\n\n"
    "{src} text:\n\"\"\" {text} \"\"\""
)
PARA_SYSTEM = (
    "You are a careful paraphrasing assistant. You restate text in different words while keeping "
    "the meaning identical. You output only the paraphrase."
)
PARA_HEAD = (
    "Paraphrase the text below. Keep the meaning identical; change the wording and sentence "
    "structure. Do not add or remove information. Output ONLY the paraphrase."
)
PARA_TAIL = "Language: {lang}.\n\nText:\n\"\"\" {text} \"\"\""

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.5-2B-Q4_K_M.gguf")
FALLBACK_TMPL = (           # GGUF only, and only if the .gguf has no chat_template in metadata
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SLOT = "\u0001__TAIL__\u0001"           # marks where the per-row tail begins inside the prefix
THINK_BLOCK = "<think>\n\n</think>\n\n"  # empty reasoning block for Qwen3.5 (thinking off)
HF_MAX_NEW = 1024                       # HF safety cap when max_new=0 (GGUF stops at n_ctx)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class Decode:
    """Per-step decoding parameters (one instance for paraphrase, one for translate)."""
    temp: float = 0.0          # 0 => greedy argmax
    top_p: float = 1.0         # 1.0 => disabled
    top_k: int = 0             # 0 => disabled
    max_new: int = 0           # 0 => until EOS (GGUF: or n_ctx; HF: or safety cap)
    rep_penalty: float = 1.0   # 1.0 => disabled


def clean(s):
    """Strip surrounding whitespace and a leading/trailing quote block (as in gguf_translate.py)."""
    s = s.strip()
    for q in ('"""', '"'):
        s = s[len(q):] if s.startswith(q) else s
        s = s[:-1] if s.endswith(q) else s
    return s.strip()


def strip_think(s):
    return _THINK_RE.sub("", s).strip()


def _warn(msg):
    click.echo(f"[prefix-cache] {msg}", err=True)


# --- prefix-cache metadata (sidecar <cache>.meta.json + a row in kvcache_meta.csv) ---
SCRIPT_NAME = "paraphrase_translate.py"
STEP_NAMES = {"para": "paraphrase", "xlat": "translate"}
KV_META_FIELDS = ["timestamp", "script", "backend", "model", "weights_type", "dtype",
                  "n_ctx", "step", "prefix_tokens", "prefix_sha1", "cache_file", "cache_bytes"]
KV_META_VERIFY = ("backend", "model", "weights_type", "n_ctx", "step", "prefix_sha1")  # must match
_GGML_FTYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
               10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
               15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K"}  # ggml llama_ftype
_QUANT_RE = re.compile(r"(IQ\d+_[A-Z0-9_]+|Q\d+_K_[SML]|Q\d+_K|Q\d+_\d+|Q\d+|BF16|F16|F32)", re.I)
_UNREADABLE = object()   # sentinel: sidecar exists but cannot be parsed


def _detect_weights_type(filename, file_type=None):
    """Quantization/weights kind: from GGUF metadata file_type if known, else the filename."""
    if file_type is not None:
        try:
            if int(file_type) in _GGML_FTYPE:
                return _GGML_FTYPE[int(file_type)]
        except (TypeError, ValueError):
            pass
    base = os.path.basename(filename)
    m = _QUANT_RE.search(base)
    return m.group(1).upper() if m else (base or "unknown")


def _norm_dtype(s):
    s = str(s).lower().replace("torch.", "")
    return {"bfloat16": "bf16", "float16": "f16", "float32": "f32", "half": "f16"}.get(s, s)


def _sha1_ids(ids):
    """Stable hash of a token-id sequence (the cached HEAD)."""
    h = hashlib.sha1()
    for t in ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def append_kvcache_meta(csv_path, meta):
    """Append one metadata row to the shared kvcache_meta.csv (create with header if missing)."""
    try:
        d = os.path.dirname(os.path.abspath(csv_path))
        if d:
            os.makedirs(d, exist_ok=True)
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=KV_META_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: meta.get(k, "") for k in KV_META_FIELDS})
    except OSError as e:
        _warn(f"failed to append {csv_path} ({e})")


def resolve_cache_paths(spec, steps, backend):
    """Map a --prefix-cache spec to {step: path} for the active steps (para before xlat).

    ``spec`` is one path (used directly for a single step; else treated as a directory, or as a
    base to which ``<step>`` is inserted) or a comma-separated list of per-step paths.
    """
    steps = [s for s in steps]
    if not spec or not steps:
        return {}
    ext = "pt" if backend == "hf" else "state"
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return {}
    if len(parts) >= len(steps):
        return dict(zip(steps, parts))
    p = parts[0]
    if os.path.isdir(p) or p.endswith((os.sep, "/", "\\")):
        return {k: os.path.join(p, f"{k}.{ext}") for k in steps}
    if len(steps) == 1:
        return {steps[0]: p}
    stem, dot, suf = p.rpartition(".")
    return ({k: f"{stem}.{k}.{suf}" for k in steps} if dot
            else {k: f"{p}.{k}" for k in steps})


def _read_sidecar(path):
    """The sidecar dict, None if absent, or _UNREADABLE if corrupt (never raises)."""
    side = path + ".meta.json"
    if not os.path.isfile(side):
        return None
    try:
        with open(side, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        _warn(f"{side}: unreadable metadata ({e}); re-prefilling")
        return _UNREADABLE


def _verify_meta(path, meta, current):
    """Warn and return False unless ``meta`` matches ``current`` on the required fields."""
    ok, bad = isinstance(meta, dict), None
    if ok:
        for k in KV_META_VERIFY:
            if meta.get(k) != current.get(k):
                ok, bad = False, k
                break
    if not ok:
        _warn(f"{path}: metadata mismatch on '{bad or 'meta'}' "
              f"(have {meta.get(bad) if isinstance(meta, dict) else None!r}, "
              f"need {current.get(bad)!r}); re-prefilling")
    return ok


def _record_saved(path, current, meta_csv, label):
    """Fill cache_file/cache_bytes, then write the sidecar + a kvcache_meta.csv row."""
    current["cache_file"] = os.path.abspath(path)
    current["cache_bytes"] = os.path.getsize(path)
    try:
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except OSError as e:
        _warn(f"failed to write {path}.meta.json ({e})")
    append_kvcache_meta(meta_csv or "kvcache_meta.csv", current)
    _warn(f"{path}: saved prefix {label} [{current['weights_type']}, "
          f"{current['prefix_tokens']} tok, {current['cache_bytes']} B]")


class _Backend:
    """Shared prefix-cache plumbing; subclasses supply the blob + prefill/decode hooks."""

    name = ""

    def _meta_row(self, key, n_pre, pre_ids, cache_file, model, dtype, n_ctx):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": SCRIPT_NAME, "backend": self.name, "model": model,
            "model_name": self.model_name(), "weights_type": self.weights_type(),
            "dtype": dtype, "n_ctx": n_ctx, "step": STEP_NAMES.get(key, key),
            "prefix_tokens": n_pre, "prefix_sha1": _sha1_ids(pre_ids),
            "cache_file": os.path.abspath(cache_file) if cache_file else "", "cache_bytes": 0,
        }

    def _reuse(self, path, current):
        """Return the persisted payload when the cache exists and metadata matches, else None."""
        if not path or not os.path.isfile(path):
            return None
        meta = _read_sidecar(path)
        if meta is _UNREADABLE:
            return None
        try:
            blob = self._blob_load(path)
        except Exception as e:
            _warn(f"{path}: load failed ({e}); re-prefilling")
            return None
        if meta is None:
            meta = blob.get("meta") if isinstance(blob, dict) else None
        return self._blob_payload(blob) if _verify_meta(path, meta, current) else None

    def _persist(self, path, payload, current, meta_csv, label):
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            self._blob_save(path, payload, current)
            _record_saved(path, current, meta_csv, label)
        except Exception as e:                               # caching is best-effort
            _warn(f"{path}: save failed ({e})")


# --------------------------------------------------------------------------- GGUF backend
def render(llm, messages):
    """Render chat messages to text with the model's GGUF chat template."""
    from llama_cpp import llama_chat_apply_template, llama_chat_message
    tmpl = (llm.metadata.get("tokenizer.chat_template") or FALLBACK_TMPL).encode()
    chat = (llama_chat_message * len(messages))()
    for i, m in enumerate(messages):
        chat[i].role, chat[i].content = m["role"].encode(), m["content"].encode()
    buf = ctypes.create_string_buffer(1 << 20)
    n = llama_chat_apply_template(tmpl, chat, len(messages), True, buf, len(buf))
    return buf.raw[:n].decode("utf-8")


class _GGUFPrompt:
    """Reload the cached HEAD state, evaluate the TAIL, sample the output."""

    def __init__(self, state, post, n_pre, seed=0):
        self.state, self.post, self.n_pre, self.last_n = state, post, n_pre, 0
        self.base_seed, self.calls = seed, 0

    def __call__(self, llm, tail, dec):
        llm.load_state(self.state)
        # load_state() restores llama.cpp's RNG seed; re-seed per call so sampling varies
        self.calls += 1
        llm.set_seed(self.base_seed + self.calls)
        head = llm.tokenize(tail.encode() + self.post, add_bos=False, special=True)
        llm.eval(head)
        n = self.n_pre + len(head)
        hard_cap = llm.n_ctx() - 1
        stop_at = min(n + dec.max_new, hard_cap) if dec.max_new > 0 else hard_cap
        out = []
        while n < stop_at:                       # stop at EOS, max_new, or when context is full
            tok = llm.sample(temp=dec.temp, top_p=dec.top_p, top_k=dec.top_k,
                             repeat_penalty=dec.rep_penalty)
            if tok == llm.token_eos():
                break
            out.append(tok)
            llm.eval([tok])
            n += 1
        self.last_n = len(out)
        return clean(strip_think(llm.detokenize(out).decode("utf-8", "replace")))


class GGUFBackend(_Backend):
    """llama.cpp backend: one saved KV state per step, optionally persisted."""

    name = "gguf"

    def __init__(self, model_path, n_gpu_layers=0, n_ctx=0, seed=0):
        self.model_path, self.n_gpu_layers, self.n_ctx, self.seed = model_path, n_gpu_layers, n_ctx, seed
        self.llm, self._prompts, self.total_tokens, self.total_time = None, {}, 0, 0.0

    def load(self):
        from llama_cpp import Llama
        self.llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                         n_gpu_layers=self.n_gpu_layers, verbose=False)
        self.llm.set_seed(self.seed)
        # Thinking-capable templates (Qwen3.5) need the empty think block appended, else the
        # model may leak an untagged reasoning preamble.
        self.append_think = "<think>" in (self.llm.metadata.get("tokenizer.chat_template") or "")
        return self

    def model_name(self):
        return os.path.basename(self.model_path)

    def weights_type(self):
        ft = self.llm.metadata.get("general.file_type") if self.llm else None
        return _detect_weights_type(self.model_path, ft)

    def _blob_load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _blob_payload(self, blob):
        return blob["state"]

    def _blob_save(self, path, payload, current):
        with open(path, "wb") as f:
            pickle.dump({"meta": current, "state": payload}, f)

    def prefix(self, key, system, head, cache_path=None, meta_csv=None):
        """Encode the language-agnostic HEAD once; cache its KV state (optionally persisted)."""
        llm = self.llm
        full = render(llm, [{"role": "system", "content": system},
                            {"role": "user", "content": head + "\n\n" + SLOT}])
        i = full.index(SLOT)
        post = full[i + len(SLOT):].encode()
        if self.append_think:
            post += THINK_BLOCK.encode()
        pre_ids = llm.tokenize(full[:i].encode(), add_bos=True, special=True)
        current = self._meta_row(key, len(pre_ids), pre_ids, cache_path,
                                 os.path.abspath(self.model_path), self.weights_type(),
                                 int(llm.n_ctx()))
        state = self._reuse(cache_path, current) if cache_path else None
        if state is None:
            llm.reset()
            llm.eval(pre_ids)
            state = llm.save_state()
            if cache_path:
                self._persist(cache_path, state, current, meta_csv, "state")
        else:
            _warn(f"{cache_path}: reused cached prefix ({len(pre_ids)} tokens), skipped prefill")
        self._prompts[key] = _GGUFPrompt(state, post, len(pre_ids), self.seed)

    def generate(self, key, tail, dec):
        t0 = time.perf_counter()
        out = self._prompts[key](self.llm, tail, dec)
        self.total_time += time.perf_counter() - t0
        self.total_tokens += self._prompts[key].last_n
        return out


# --------------------------------------------------------------------------- HF backend
class HFBackend(_Backend):
    """transformers backend: the KV cache is deep-copied per call, optionally persisted."""

    name = "hf"

    def __init__(self, model_id_or_path, device="cpu", dtype="bf16", seed=0):
        self.model_id_or_path, self.device, self.dtype, self.seed = model_id_or_path, device, dtype, seed
        self.tok = self.model = self.torch = self.eos_id = None
        self.max_new = HF_MAX_NEW
        self._caches, self._lens, self._posts, self._pres = {}, {}, {}, {}
        self.total_tokens, self.total_time = 0, 0.0

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(self.seed)
        self.tok = AutoTokenizer.from_pretrained(self.model_id_or_path)
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": None}[self.dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path, **({} if dt is None else {"dtype": dt}))
        self.model.to(self.device).eval()
        self.eos_id = self.tok.eos_token_id
        return self

    def model_name(self):
        return os.path.basename(self.model_id_or_path.rstrip("/\\"))

    def _param_dtype(self):
        try:
            return _norm_dtype(next(self.model.parameters()).dtype)
        except Exception:
            return _norm_dtype(self.dtype)

    def weights_type(self):
        cfg = self.model.config
        dt = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
        return _norm_dtype(dt) if dt is not None else self._param_dtype()

    def _blob_load(self, path):
        return self.torch.load(path, weights_only=False, map_location=self.device)

    def _blob_payload(self, blob):
        cache = blob["cache"] if isinstance(blob, dict) else blob
        try:
            cache = cache.to(self.device)
        except Exception:
            pass
        return cache

    def _blob_save(self, path, payload, current):
        self.torch.save({"meta": current, "cache": payload}, path)

    def prefix(self, key, system, head, cache_path=None, meta_csv=None):
        """Tokenise the language-agnostic HEAD, forward it once, cache (+ optionally persist) KV."""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": head + "\n\n" + SLOT}]
        try:  # Qwen3.5: disable the reasoning block at the source
            full = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                tokenize=False, enable_thinking=False)
        except TypeError:  # templates that don't take enable_thinking
            full = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        i = full.index(SLOT)
        pre, post = full[:i], full[i + len(SLOT):]
        pre_ids = self.tok(pre, add_special_tokens=False)["input_ids"]
        n_pre = len(pre_ids)
        current = self._meta_row(key, n_pre, pre_ids, cache_path, self.model_id_or_path,
                                 self._param_dtype(),
                                 int(getattr(self.model.config, "max_position_embeddings", 0) or 0))
        cache = self._reuse(cache_path, current) if cache_path else None
        if cache is None:
            with self.torch.inference_mode():
                out = self.model(input_ids=self.torch.tensor([pre_ids], device=self.device),
                                 use_cache=True)
            cache = out.past_key_values
            if cache_path:
                self._persist(cache_path, cache, current, meta_csv, "cache")
        else:
            _warn(f"{cache_path}: reused cached prefix ({n_pre} tokens), skipped prefill")
        self._caches[key], self._lens[key] = cache, n_pre
        self._posts[key], self._pres[key] = post, list(pre_ids)

    @staticmethod
    def _rep_penalty(torch, logits, seen, penalty):
        """Apply a repetition penalty to ``seen`` token ids (no-op when penalty == 1.0)."""
        if penalty == 1.0 or not seen:
            return logits
        idx = torch.tensor(sorted(set(seen)), device=logits.device)
        vals = logits.index_select(0, idx)
        vals = torch.where(vals < 0, vals * penalty, vals / penalty)
        logits = logits.clone()
        logits[idx] = vals
        return logits

    @staticmethod
    def _top_k_top_p(torch, logits, top_k, top_p):
        """Top-k then top-p (nucleus) filtering of a 1-D logits vector."""
        if top_k and top_k > 0:
            k = min(int(top_k), logits.shape[-1])
            kth = torch.topk(logits, k)[0][-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p and top_p < 1.0:
            sorted_logits, order = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1) - probs   # keep tokens up to cumulative top_p
            sorted_logits = sorted_logits.masked_fill(cum > top_p, float("-inf"))
            logits = torch.empty_like(logits).scatter_(0, order, sorted_logits)
        return logits

    def generate(self, key, tail, dec):
        torch = self.torch
        t0 = time.perf_counter()
        cache = copy.deepcopy(self._caches[key])
        npre = self._lens[key]
        new_ids = self.tok(tail + self._posts[key], add_special_tokens=False)["input_ids"]

        def _fwd(ids, n_start):
            n = n_start + len(ids)
            with torch.inference_mode():
                res = self.model(
                    input_ids=torch.tensor([ids], device=self.device),
                    past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(n_start, n, device=self.device),
                    attention_mask=torch.ones((1, n), dtype=torch.long, device=self.device))
            return res.logits[0, -1].float(), n

        logits, n = _fwd(new_ids, npre)
        seen = list(self._pres[key]) + list(new_ids)
        out_tokens = []
        cap = dec.max_new if dec.max_new > 0 else self.max_new   # 0 => until EOS (safety-capped)
        for _ in range(cap):
            lg = self._rep_penalty(torch, logits, seen, dec.rep_penalty)
            if dec.temp and dec.temp > 0:                        # sampling
                lg = self._top_k_top_p(torch, lg / dec.temp, dec.top_k, dec.top_p)
                tok = int(torch.multinomial(torch.softmax(lg, dim=-1), 1))
            else:                                                # greedy
                tok = int(lg.argmax())
            if tok == self.eos_id:
                break
            out_tokens.append(tok)
            seen.append(tok)
            logits, n = _fwd([tok], n)
        self.total_tokens += len(out_tokens)
        self.total_time += time.perf_counter() - t0
        return clean(strip_think(self.tok.decode(out_tokens, skip_special_tokens=True)))


# --------------------------------------------------------------------------- selection + CLI
def detect_backend(model):
    """auto-detect: .gguf file -> gguf; dir with config.json / HF id -> hf."""
    if model.lower().endswith(".gguf"):
        return "gguf"
    if os.path.isdir(model):
        return "gguf" if any(f.endswith(".gguf") for f in os.listdir(model)) else "hf"
    return "gguf" if os.path.isfile(model) else "hf"


def build_backend(backend, model, device, dtype, n_gpu_layers, n_ctx, seed):
    if backend == "gguf":
        return GGUFBackend(model, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HFBackend(model, device=device, dtype=dtype, seed=seed)


def _norm(s):
    """Normalised form for dedup/equality (case-, punctuation- and whitespace-insensitive)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip().lower()


def _accept(p, src, min_ratio, max_ratio, seen):
    """Accept a paraphrase only if non-empty, sane length, not identical to the input, and new."""
    if not p:
        return False
    ratio = len(p) / max(1, len(src))
    if ratio < min_ratio or ratio > max_ratio:      # reject truncations / rambles
        return False
    k = _norm(p)
    return bool(k) and k != _norm(src) and k not in seen


def paraphrase_candidates(eng, lang, text, n, dec, min_ratio=0.6, max_ratio=2.0, max_tries=4):
    """Up to N accepted paraphrases (length-filtered, != input, deduplicated) with refill.

    One generation per attempt; refills until N are accepted or the budget (N * max_tries) is
    spent; pads with '' if it cannot reach N."""
    if not text:
        return [""] * n
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * max_tries:
        tries += 1
        p = eng.generate("para", PARA_TAIL.format(lang=lang, text=text), dec)
        if _accept(p, text, min_ratio, max_ratio, seen):
            out.append(p)
            seen.add(_norm(p))
    return out + [""] * (n - len(out))


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="path to a .gguf OR HF id / local safetensors dir")
@click.option("--backend", type=click.Choice(["auto", "gguf", "hf"]), default="auto",
              show_default=True, help="backend; auto-detected from --model when 'auto'")
@click.option("--in-csv", "in_csv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-csv", "out_csv", required=True)
@click.option("--text-col", default=None, help="input column (default: 'text' or first)")
@click.option("--out-col", default=None, help="output column (default: <tgt>_text)")
@click.option("--src", default="English", help="language of the input text")
@click.option("--tgt", default="Italian", help="target language")
@click.option("--translate-only", is_flag=True,
              help="translate src->tgt directly, no paraphrasing (1 row/input); needs --src != --tgt")
@click.option("--prefix-cache", "prefix_cache", default=None,
              help="persist/reuse the prefix KV cache: one path, a dir, or '<para>,<xlat>' paths")
@click.option("--kvcache-meta", "kvcache_meta", default="kvcache_meta.csv", show_default=True,
              help="CSV where each saved prefix cache appends a metadata row")
@click.option("-n", "--n-paraphrases", default=4, show_default=True)
@click.option("--para-min-ratio", default=0.6, show_default=True,
              help="min len(out)/len(in) to accept a paraphrase")
@click.option("--para-max-ratio", default=2.0, show_default=True,
              help="max len(out)/len(in) to accept a paraphrase")
@click.option("--para-max-tries", default=4, show_default=True,
              help="refill budget: generation attempts per input is N x this")
@click.option("--para-temp", "--temp", "para_temp", default=0.8, show_default=True,
              help="paraphrase temperature (alias: --temp); 0 = greedy")
@click.option("--trans-temp", default=0.0, show_default=True,
              help="translation temperature; 0 = greedy (default)")
@click.option("--para-top-p", default=1.0, show_default=True, help="paraphrase nucleus sampling (1.0 = off)")
@click.option("--para-top-k", default=50, show_default=True, help="paraphrase top-k (0 = off)")
@click.option("--trans-top-p", default=1.0, show_default=True, help="translation nucleus sampling (1.0 = off)")
@click.option("--trans-top-k", default=0, show_default=True, help="translation top-k (0 = off / greedy)")
@click.option("--para-max-new", default=0, show_default=True, help="paraphrase max new tokens (0 = until EOS)")
@click.option("--trans-max-new", default=0, show_default=True, help="translation max new tokens (0 = until EOS)")
@click.option("--para-rep-penalty", default=1.0, show_default=True, help="paraphrase repetition penalty")
@click.option("--trans-rep-penalty", default=1.0, show_default=True, help="translation repetition penalty")
@click.option("--seed", default=0, show_default=True, help="RNG seed")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="cpu",
              show_default=True, help="HF device (CPU by default)")
@click.option("--dtype", type=click.Choice(["bf16", "fp16", "auto"]), default="bf16",
              show_default=True, help="HF model dtype")
@click.option("--n-gpu-layers", default=0, show_default=True,
              help="GGUF: 0=CPU (default), -1=all layers on GPU")
@click.option("--n-ctx", default=0, show_default=True, help="GGUF context size (0 = model default)")
@click.option("--limit", default=0, help="only first N rows (0 = all)")
@click.option("--encoding", default="utf-8-sig", show_default=True,
              help="input CSV encoding (e.g. latin-1 for FLORES)")
def main(model, backend, in_csv, out_csv, text_col, out_col, src, tgt, translate_only,
         prefix_cache, kvcache_meta, n_paraphrases, para_min_ratio, para_max_ratio,
         para_max_tries, para_temp, trans_temp, para_top_p, para_top_k, trans_top_p, trans_top_k,
         para_max_new, trans_max_new, para_rep_penalty, trans_rep_penalty, seed, device, dtype,
         n_gpu_layers, n_ctx, limit, encoding):
    """Paraphrase (and translate) a CSV column with a local GGUF or safetensors LLM."""
    if backend == "auto":
        backend = detect_backend(model)
    if backend == "gguf" and not os.path.isfile(model):
        sys.exit(f"model not found: {model}\n  put a Qwen-2B GGUF there or pass --model PATH")

    try:
        with open(in_csv, encoding=encoding, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except UnicodeDecodeError:
        with open(in_csv, encoding="latin-1", newline="") as fh:
            rows = list(csv.DictReader(fh))
        click.echo(f"note: {in_csv} not decodable as {encoding}; fell back to latin-1")
    if not rows:
        sys.exit("input CSV is empty")
    text_col = text_col or ("text" if "text" in rows[0] else next(iter(rows[0])))
    out_col = out_col or f"{tgt.lower()}_text"
    translate = src.lower() != tgt.lower()
    if translate_only and not translate:
        sys.exit("--translate-only needs --src != --tgt")
    if limit:
        rows = rows[:limit]

    para_dec = Decode(temp=para_temp, top_p=para_top_p, top_k=para_top_k,
                      max_new=para_max_new, rep_penalty=para_rep_penalty)
    trans_dec = Decode(temp=trans_temp, top_p=trans_top_p, top_k=trans_top_k,
                       max_new=trans_max_new, rep_penalty=trans_rep_penalty)
    active = ([] if translate_only else ["para"]) + (["xlat"] if translate else [])  # step order
    cache_paths = resolve_cache_paths(prefix_cache, active, backend)

    mode = "translate-only" if translate_only else ("paraphrase+translate" if translate else "paraphrase")
    click.echo(f"backend={backend} model={model} device={device} dtype={dtype} rows={len(rows)} "
               f"{src}->{tgt} mode={mode} n={n_paraphrases} seed={seed}\n"
               f"  para : temp={para_dec.temp} top_p={para_dec.top_p} top_k={para_dec.top_k} "
               f"max_new={para_dec.max_new} rep={para_dec.rep_penalty}\n"
               f"  trans: temp={trans_dec.temp} top_p={trans_dec.top_p} top_k={trans_dec.top_k} "
               f"max_new={trans_dec.max_new} rep={trans_dec.rep_penalty}\n"
               f"  prefix-cache: {cache_paths or 'off'}")
    eng = build_backend(backend, model, device, dtype, n_gpu_layers, n_ctx, seed).load()

    # one language-agnostic prefix per step (shared across all languages)
    if translate:
        eng.prefix("xlat", XLING_SYSTEM, XLING_HEAD, cache_paths.get("xlat"), kvcache_meta)
    if not translate_only:          # skip the paraphrase prefix entirely when it is unused
        eng.prefix("para", PARA_SYSTEM, PARA_HEAD, cache_paths.get("para"), kvcache_meta)
    para_lang = src if translate else tgt

    fields = list(rows[0]) + [c for c in (out_col, "variant") if c not in rows[0]]
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in rows:
            text = (r.get(text_col) or "").strip()
            if translate_only:                      # direct translation: exactly 1 row/input
                out = eng.generate("xlat", XLING_TAIL.format(src=src, tgt=tgt, text=text),
                                   trans_dec) if text else ""
                w.writerow({**r, out_col: out, "variant": 0})
                fh.flush()
                continue
            cands = paraphrase_candidates(eng, para_lang, text, n_paraphrases, para_dec,
                                          para_min_ratio, para_max_ratio, para_max_tries)
            for v in range(n_paraphrases):
                out = cands[v] if v < len(cands) else ""
                if translate and out:
                    out = eng.generate("xlat", XLING_TAIL.format(src=src, tgt=tgt, text=out),
                                       trans_dec)
                w.writerow({**r, out_col: out, "variant": v})
                fh.flush()
    click.echo(f"wrote {out_csv}")
    if eng.total_time > 0:
        click.echo(f"generated {eng.total_tokens} tokens in {eng.total_time:.1f}s "
                   f"({eng.total_tokens / eng.total_time:.2f} tok/s)")


if __name__ == "__main__":
    main()
