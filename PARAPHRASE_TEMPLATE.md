# Paraphrase / cross-lingual rewrite templates

Templates for generating **synthetic training text** (paraphrase and English→target rewrites) with a
small instruct LLM. Grounded in the literature reviewed for this project:

- **FinePhrase** (Niklaus et al. 2026, arXiv:2604.13977): a **~1B-class generator** is optimal
  (SmolLM2-1.7B beat Gemma 3 / Llama 3.2 / Qwen 3 / Granite 3.1 / Falcon 3 at 1B; >1B gives no gain);
  **prompt design matters more than model size**; **structured formats** (table / faq / tutorial / math)
  beat plain paraphrase for pretraining data.
- **Self-Translate-Train** (arXiv:2407.00454): let the model translate its own labeled training data into
  the target language and fine-tune on the (filtered) output — beats zero-shot.
- **To Translate or Not to Translate** (arXiv:2311.09404): **keep the clean source-language data in the
  mix** (`TRG+SRC`); round-trip augmentation helps.
- **Geometric Filtering of LLM-Generated Samples** (arXiv:2608.13866): filter generated samples by their
  **distance to real class examples in the embedding space** (simple Euclidean filter wins).
- **SemEval-2026 Task 9** (Calderón-Reyes et al.): translated examples preserve the *label*; greedy
  selection over augmented sources.

---

## 0. Conventions

- `{TEXT}` — the input text (one document/sentence).
- `{SRC}` / `{TGT}` — source / target **language names** (e.g. `English`, `Italian`).
- `{LABEL}` — the task label(s), **preserved unchanged** (do not paraphrase labels).
- Use the model's **chat template**; set `enable_thinking=False` for Qwen3-style models.
- **Decoding:** cross-lingual rewrite → **greedy** (`do_sample=False`); paraphrase diversity → **sampling**
  (`temperature=0.7–1.0, top_p=1.0, top_k=50`), `max_new_tokens` ≥ input length × 1.5, stop at EOS.

---

## 1. Cross-lingual rewrite (English → target) — primary template

**System**
```
You are a careful multilingual data generator. You rewrite the user's text into {TGT} while preserving
its meaning exactly. You never add, drop, summarise, or explain information. You output only the {TGT}
text, with no quotes, labels, or commentary.
```

**User**
```
Rewrite the following {SRC} text into natural, fluent {TGT}.
- Preserve ALL factual content and entities (names, numbers, dates).
- Do not add or remove information.
- Do not translate proper nouns into different entities.
- Output ONLY the {TGT} text. No preamble, no quotes.

{SRC} text:
""" {TEXT} """
```

> Preserves the task label (e.g. an article's topic) because nothing is added or removed. This is the
> `translate-train` / `self-translate-train` generator.

---

## 2. Monolingual paraphrase (same language, different surface form)

**System**
```
You are a careful paraphrasing assistant. You restate text in different words while keeping the meaning
identical. You output only the paraphrase.
```

**User**
```
Paraphrase the following {LANG} text. Keep the meaning identical; change the wording and sentence
structure. Do not add or remove information. Output ONLY the paraphrase.

Text:
""" {TEXT} """
```

**Diverse variant (K samples):** prepend a style hint so repeated sampling differs, e.g. one of
`"Use a formal register."`, `"Use a simple, everyday register."`, `"Restructure the sentences."`,
`"Use different synonyms throughout."`.

---

## 3. Structured pedagogical rewrite (FinePhrase-style) — for *pretraining-style* corpora

FinePhrase found these four formats beat plain paraphrase. One template per format; `{TEXT}` is the
source document.

```
TABLE:     Rewrite the passage as a well-formed Markdown table that captures the same information.
FAQ:       Turn the passage into a list of question–answer pairs covering the same content.
TUTORIAL:  Rewrite the passage as a short step-by-step tutorial that teaches the same material.
MATH:      Write a math word problem whose solution requires the facts in the passage, then solve it.
```
(Use only if the goal is general-purpose synthetic pretraining text; for **labeled** classification
data use Section 1/2 so the label survives.)

---

## 4. Back-translation / round-trip (robustness, label-preserving)

Generate `{TEXT}` → `{TGT}` → `{SRC}` with Section 1 twice (swapping `{SRC}`/`{TGT}`); fine-tune on the
round-tripped text. Matches the source data's MT idiosyncrasies (paper #3), which helps at test time.

---

## 5. Filtering (do this — the generator will err)

Small LLMs produce some bad outputs (our runs mistranslated e.g. "The Sun" → *"L'urna"*). Apply:

1. **Termination / repetition:** drop outputs that do not end with EOS or repeat n-grams.
2. **Length ratio:** drop if `len(out) / len(in)` outside ~`[0.6, 2.0]` (per-token estimates fine).
3. **Dedup:** drop duplicates (exact and near).
4. **Embedding-space geometric filter** (arXiv:2608.13866): embed the generated text with the task
   encoder (e.g. EuroBERT mean), and keep a sample only if its distance to the **real** examples of its
   class is within a threshold (simple Euclidean filter beat multi-criteria alternatives). Optionally
   **soft-weight** samples by that distance.
5. **Mix in the clean source data** (`SRC + generated`), never synthetic-only (arXiv:2311.09404).

---

## 6. Minimal usage example (transformers, chat template)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

mid = "HuggingFaceTB/SmolLM3-3B"            # or HuggingFaceTB/SmolLM2-1.7B-Instruct (FinePhrase's pick)
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda").eval()

SYS = ("You are a careful multilingual data generator. You rewrite the user's text into {tgt} while "
       "preserving its meaning exactly. Output only the {tgt} text, no quotes or commentary.")
USR = ("Rewrite the following {src} text into natural, fluent {tgt}. Preserve ALL factual content and "
       "entities. Do not add or remove information. Output ONLY the {tgt} text.\n\n{src} text:\n\"\"\" {text} \"\"\"")

def rewrite(text, src="English", tgt="Italian", sample=False):
    msgs = [{"role": "system", "content": SYS.format(tgt=tgt)},
            {"role": "user",   "content": USR.format(src=src, tgt=tgt, text=text)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=False).to(model.device)
    kw = dict(max_new_tokens=min(1024, int(len(text.split()) * 2.2) + 32), do_sample=sample,
              temperature=0.7 if sample else None, top_p=1.0, top_k=50)
    out = model.generate(ids, **{k: v for k, v in kw.items() if v is not None})
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
```

---

## 7. Appendix — FinePhrase's own prompt templates (verbatim, for reference)

The FinePhrase repo (`github.com/huggingface/finephrase/prompts/`) ships a **library of rephrasing
prompt templates** organized by strategy: `rewire/` (guided rewrite), `nemotron/` (wikipedia-style,
distill, diverse QA pairs, extract knowledge, knowledge list), `dspy/` (GEPA-optimized), `format/`
(table/math/faq/tutorial), `beyondweb/`. Conventions: a `[TEXT]` slot, an explicit output constraint,
and often a priming suffix. Two verbatim examples (attribution: Niklaus et al. 2026, Apache-2.0 repo):

```
# prompts/rephrase.md
Rephrase the low-quality web document below into higher quality, more educational content.
Make sure the rephrased document can stand on its own; do NOT reference the input text.
Output only the rephrased text, nothing else.

Document:
[TEXT]

Here's a rephrased version of the provided text, aiming for a higher quality and more
educational content, without referencing the original input:
```

```
# prompts/nemotron/wikipedia_style_rephrasing.md
For the following paragraph give me a diverse paraphrase of the same in high quality English
language as in sentences on Wikipedia. Begin your answer on a separate line with
"Here is a paraphrased version:".

Text:
[TEXT]
```

Use these for *pretraining-style* rephrasing; for **labeled** classification data use Section 1/2
(label-preserving translation/paraphrase). Other FinePhrase prompts: `nemotron/distill.md`,
`nemotron/extract_knowledge.md`, `nemotron/diverse_qa_pairs.md`, `nemotron/knowledge_list.md`,
`rewire/guided_rewrite_corrected.md`, `dspy/5-max_full_evals.md` (see the repo's `prompts/` tree).

## 8. Prompt-design checklist (from FinePhrase)

- Prefer **one clear instruction** over many constraints; state the **output format explicitly**
  ("Output only the …"), which was the single biggest lever.
- **Structure** (table/faq/tutorial/math) > plain paraphrase for pretraining corpora.
- Don't bother scaling the generator past ~1B; **invest in the prompt**.
- Keep `temperature` in the mid-range for diversity without drift; use greedy for faithful translation.
