# paraphrase_translate.py — paraphrase → translate (one file, both model formats)

Generates synthetic **target-language** text from a CSV column, optionally translating.
Two modes, chosen by `--src` vs `--tgt`:

| condition | pipeline | output |
|---|---|---|
| `--src != --tgt` (e.g. English→Polish) | paraphrase the source ×N, then translate each paraphrase into the target | N rows/input |
| `--src == --tgt` (text already in target) | paraphrase ×N in that language | N rows/input |

Output CSV = original columns + `<tgt>_text` + `variant` (0…N-1).

## Logic and "why X, not Y"

**Why paraphrase-then-translate (not translate-only, not translate-then-paraphrase).**
One source sentence gives many training rows, and the variation is created where the model
is strongest (its own language) before translation. `translate-then-paraphrase` needs target
text first and flattens register; `translate-only` gives no diversity. See `PARAPHRASE_TEMPLATE.md`.

**Why paraphrase before translate (source-side).** You only have English; the LLM's paraphrase
quality is best in English. Diversify in English, then translate — cheaper and higher quality
than asking for many diverse target-language rewrites in one shot.

**Why temperature 0 for translate, 0.8 for paraphrase.**
Translation is deterministic — higher temperature lowers BLEU/meaning (see README "research").
Paraphrasing wants diversity, so it samples (0.7–1.0; FinePhrase uses 1.0). `--temp` sets the
paraphrase step; translation is always greedy.

**Why no max-token setting.** Generation stops at EOS, so nothing is truncated mid-sentence and
short/long inputs both work. The only bound is the context window.

**Why KV-prefix caching.** The template's fixed part (system + instructions, ~140 tokens) is
identical for every row. We encode it **once** and reuse its KV cache, so each row only computes
its own new tokens. Same idea, two mechanisms:
- **GGUF** (`llama-cpp-python`): `reset()+eval(prefix)+save_state()` once; per row `load_state()`
  then eval only the new tokens.
- **safetensors** (`transformers`): forward the prefix with `use_cache=True`, keep the
  `past_key_values`, `deepcopy` it per row, forward only the new tokens.

With `--prefix-cache PATH` the prefix KV is also **persisted to disk** (GGUF: `LlamaState` is
picklable — its KV is a `bytes` blob; HF: `torch.save(past_key_values)`), so a later run **skips
prefill entirely**. Each cache carries **metadata** — model, **weights type/quant** (e.g. `Q4_K_M`,
`BF16`), dtype, `n_ctx`, which template step, prefix token count + `prefix_sha1` — written to a
sidecar `.meta.json` and appended to **`kvcache_meta.csv`**. On load the metadata is **validated**;
a mismatch (different model/quant/context) means the cache is ignored and the prefix is refilled.
Because the HEAD is language-agnostic, one cached prefix serves every language pair.

**Why a two-step template (common head + language/text tail).** Templates are split into a
**HEAD** (system prompt + instructions, containing **no** language name) and a **TAIL**
(`Target language: … / Source language: …` + the text). Only the HEAD is prefilled and KV-cached,
so **one** cached prefix serves **every** language — EN→IT, EN→PL, … never re-prefill the
instructions. The language lives in the tail.

**Why support both GGUF and safetensors.** GGUF = quantized, small, fast on CPU/GPU via llama.cpp.
safetensors = the original HF weights (full precision, any HF model). Same CLI either way;
`--backend auto` picks from the file extension.

**Why templates are inline (no `paraphrase_prompts` import).** So the script is one self-contained
file you can copy anywhere; no prompt module to ship alongside.

**Why one model resident / greedy decode / `enable_thinking=False`.** Keep memory low; greedy for
faithful translation; disable Qwen3.5-style reasoning so output is the translation, not a ` thinking`
trace (any leftover ` thinking…</think>` is stripped).

**Why the `variant` column (rows, not columns).** Augmentation multiplies *examples*; N rows per
input drop straight into a training set. A `variant` index keeps provenance.

**Diversity (`--para-sep`, default `||`).** One call asks the model for all N paraphrases,
explicitly *different*, joined by the marker; the output is split on it and deduplicated, and if
the model returns too few, single calls top up. `--para-sep ''` falls back to N separate calls.
Either way, the GGUF backend re-seeds per call (`load_state` otherwise restores the sampler seed and
every variant comes out identical).

**Why CPU by default (`--n-gpu-layers 0`).** Works anywhere; pass `-1` for GPU (needs a
CUDA-built `llama-cpp-python`), or `--device cuda` for the HF backend.

## Quick use
```powershell
python paraphrase_translate.py --in-csv en.csv --out-csv pl.csv --tgt Polish
python paraphrase_translate.py --in-csv pl.csv --out-csv pl_para.csv --src Polish --tgt Polish
python paraphrase_translate.py --model Qwen3.5-2B-Q4_K_M.gguf --n-gpu-layers -1 --in-csv en.csv --out-csv pl.csv --tgt Polish
```

`gguf_translate.py` is the GGUF-only predecessor of this script.
