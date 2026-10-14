"""gguf_translate.py -- paraphrase (+translate) a CSV text column with a local GGUF model.

Mode is chosen from --src / --tgt / --translate-only:
  * --src != --tgt        paraphrase the source N times, then translate each into the target -> N rows
  * --src == --tgt        paraphrase N times in that language -> N rows
  * --translate-only      translate the source once -> 1 row

Templates are TWO-STEP so the KV prefix is shared across every language:
  * a COMMON head (system + instruction, NO language-specific text) -> encoded once, KV-cached;
  * a per-row TAIL = {language instruction + text} -> the only thing evaluated per request.
So one cached prefix serves all source/target languages; the language lives in the tail.

Prompt templates are defined inline (no external prompt module). Generation runs until EOS unless
a *-max-new is set. CPU by default (pass --n-gpu-layers -1 for GPU).

Usage:
  python gguf_translate.py --in-csv en.csv --out-csv it.csv                       # EN -> IT
  python gguf_translate.py --in-csv en.csv --out-csv pl.csv --tgt Polish          # EN -> PL
  python gguf_translate.py --in-csv en.csv --out-csv pl.csv --tgt Polish --translate-only
  python gguf_translate.py --in-csv pl.csv --out-csv pl_para.csv --src Polish --tgt Polish
"""
import csv
import ctypes
import os
import sys

import click
from llama_cpp import Llama, llama_chat_apply_template, llama_chat_message

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.5-2B-Q4_K_M.gguf")

FALLBACK_TMPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SLOT = "\u0001__TAIL__\u0001"  # marks where the per-row tail begins when building the prefix

# --- templates (two-step: HEAD has no language; TAIL has language + text) ---
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
PARA_TAIL = (
    "Language: {lang}.\n\n"
    "Text:\n\"\"\" {text} \"\"\""
)


def render(llm, messages):
    """Render chat messages to text with the model's GGUF chat template."""
    tmpl = (llm.metadata.get("tokenizer.chat_template") or FALLBACK_TMPL).encode()
    chat = (llama_chat_message * len(messages))()
    for i, m in enumerate(messages):
        chat[i].role, chat[i].content = m["role"].encode(), m["content"].encode()
    buf = ctypes.create_string_buffer(1 << 20)
    n = llama_chat_apply_template(tmpl, chat, len(messages), True, buf, len(buf))
    return buf.raw[:n].decode("utf-8")


def clean(s):
    s = s.strip()
    for q in ('"""', '"'):
        s = s[len(q):] if s.startswith(q) else s
        s = s[:-1] if s.endswith(q) else s
    return s.strip()


class Prompt:
    """Two-step chat template: a language-agnostic HEAD encoded once (KV-cached) + a per-row TAIL."""

    def __init__(self, llm, system, head):
        full = render(llm, [{"role": "system", "content": system},
                            {"role": "user", "content": head + "\n\n" + SLOT}])
        i = full.index(SLOT)
        pre = llm.tokenize(full[:i].encode(), add_bos=True, special=True)
        self.post = full[i + len(SLOT):].encode()
        llm.reset()
        llm.eval(pre)
        self.n_pre = len(pre)
        self.state = llm.save_state()

    def __call__(self, llm, tail, temp, top_p=1.0, top_k=0, max_new=0, rep_penalty=1.0):
        llm.load_state(self.state)
        head = llm.tokenize(tail.encode() + self.post, add_bos=False, special=True)
        llm.eval(head)
        n = self.n_pre + len(head)
        out = []
        while n < llm.n_ctx() - 1 and (max_new <= 0 or len(out) < max_new):
            tok = llm.sample(temp=temp, top_p=top_p, top_k=top_k, repeat_penalty=rep_penalty)
            if tok == llm.token_eos():
                break
            out.append(tok)
            llm.eval([tok])
            n += 1
        return clean(llm.detokenize(out).decode("utf-8", "replace"))


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="path to a .gguf model")
@click.option("--in-csv", "in_csv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-csv", "out_csv", required=True)
@click.option("--text-col", default=None, help="input column (default: 'text' or first)")
@click.option("--out-col", default=None, help="output column (default: <tgt>_text)")
@click.option("--src", default="English", help="language of the input text")
@click.option("--tgt", default="Italian", help="target language")
@click.option("--translate-only", is_flag=True, help="translate only, no paraphrasing (1 row/input)")
@click.option("-n", "--n-paraphrases", default=4, show_default=True)
@click.option("--para-temp", default=0.8, show_default=True)
@click.option("--para-top-p", default=1.0, show_default=True)
@click.option("--para-top-k", default=50, show_default=True)
@click.option("--para-max-new", default=0, show_default=True, help="0 = until EOS")
@click.option("--trans-temp", default=0.0, show_default=True)
@click.option("--trans-top-p", default=1.0, show_default=True)
@click.option("--trans-top-k", default=0, show_default=True, help="0 = disabled (greedy)")
@click.option("--trans-max-new", default=0, show_default=True, help="0 = until EOS")
@click.option("--rep-penalty", default=1.0, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--n-ctx", default=0, show_default=True, help="context size (0 = model default)")
@click.option("--n-gpu-layers", default=0, help="0=CPU (default), -1=all layers on GPU", show_default=True)
@click.option("--limit", default=0, help="only first N rows (0 = all)")
def main(model, in_csv, out_csv, text_col, out_col, src, tgt, translate_only, n_paraphrases,
         para_temp, para_top_p, para_top_k, para_max_new, trans_temp, trans_top_p, trans_top_k,
         trans_max_new, rep_penalty, seed, n_ctx, n_gpu_layers, limit):
    """Paraphrase (and translate) a CSV column with a local GGUF model."""
    if not os.path.isfile(model):
        sys.exit(f"model not found: {model}\n  put a GGUF there or pass --model PATH")
    with open(in_csv, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("input CSV is empty")
    text_col = text_col or ("text" if "text" in rows[0] else next(iter(rows[0])))
    out_col = out_col or f"{tgt.lower()}_text"
    translate = src.lower() != tgt.lower()
    if translate_only and not translate:
        sys.exit("--translate-only needs --src != --tgt")
    if limit:
        rows = rows[:limit]

    llm = Llama(model_path=model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    llm.set_seed(seed)

    # one language-agnostic prefix per step (shared across all languages)
    para_lang = src if translate else tgt
    para = None if translate_only else Prompt(llm, PARA_SYSTEM, PARA_HEAD)
    xlat = Prompt(llm, XLING_SYSTEM, XLING_HEAD) if translate else None

    fields = list(rows[0]) + [c for c in (out_col, "variant") if c not in rows[0]]
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in rows:
            text = (r.get(text_col) or "").strip()
            if translate_only:
                out = xlat(llm, XLING_TAIL.format(src=src, tgt=tgt, text=text),
                           trans_temp, trans_top_p, trans_top_k, trans_max_new,
                           rep_penalty) if text else ""
                w.writerow({**r, out_col: out, "variant": 0})
                fh.flush()
                continue
            for v in range(n_paraphrases):
                out = para(llm, PARA_TAIL.format(lang=para_lang, text=text),
                           para_temp, para_top_p, para_top_k, para_max_new,
                           rep_penalty) if text else ""
                if xlat and out:
                    out = xlat(llm, XLING_TAIL.format(src=src, tgt=tgt, text=out),
                               trans_temp, trans_top_p, trans_top_k, trans_max_new,
                               rep_penalty)
                w.writerow({**r, out_col: out, "variant": v})
                fh.flush()
    click.echo(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
