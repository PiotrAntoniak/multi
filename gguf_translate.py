"""gguf_translate.py -- paraphrase (+translate) a CSV text column with a local GGUF model.

Mode is picked from --src vs --tgt:
  * --src != --tgt  (e.g. English -> Italian): paraphrase the source N times, then
    translate each paraphrase into the target language.   ->  N output rows per input.
  * --src == --tgt  (text is already in the target language): paraphrase N times
    in that language.                                     ->  N output rows per input.

Templates come from `paraphrase_prompts` (the same ones gen_compare.py uses). Each
template's fixed prompt prefix is encoded ONCE and its KV cache is reused for every
call, so only the new text is evaluated per request.

Runs on CPU by default (pass --n-gpu-layers -1 to offload). Generation runs until the
model emits EOS -- there is no max-token setting.

Usage:
  python gguf_translate.py --in-csv en.csv --out-csv it.csv           # EN -> IT (defaults)
  python gguf_translate.py --in-csv it.csv --out-csv it_para.csv --src Italian --tgt Italian
  python gguf_translate.py --model path/to/other-Q4_K_M.gguf --in-csv en.csv --out-csv it.csv
"""
import csv
import ctypes
import os
import sys

import click
from llama_cpp import Llama, llama_chat_apply_template, llama_chat_message

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paraphrase_prompts import PARA_SYSTEM, PARA_USER, XLING_SYSTEM, XLING_USER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.5-2B-Q4_K_M.gguf")

FALLBACK_TMPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SLOT = "\u0001__TEXT__\u0001"  # marks where the text goes when building the prefix


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
    """A chat template whose fixed prefix is encoded once and its KV cache reused."""

    def __init__(self, llm, system, user):
        full = render(llm, [{"role": "system", "content": system},
                            {"role": "user", "content": user.replace("{text}", SLOT)}])
        i = full.index(SLOT)
        pre = llm.tokenize(full[:i].encode(), add_bos=True, special=True)
        self.post = full[i + len(SLOT):].encode()
        llm.reset()
        llm.eval(pre)
        self.n_pre = len(pre)
        self.state = llm.save_state()

    def __call__(self, llm, text, temp):
        llm.load_state(self.state)
        head = llm.tokenize(text.encode() + self.post, add_bos=False, special=True)
        llm.eval(head)
        n = self.n_pre + len(head)
        out = []
        while n < llm.n_ctx() - 1:               # stop at EOS or when context is full
            tok = llm.sample(temp=temp)
            if tok == llm.token_eos():
                break
            out.append(tok)
            llm.eval([tok])
            n += 1
        return clean(llm.detokenize(out).decode("utf-8", "replace"))


def para_templates(lang):
    """Paraphrase templates with {text} left unfilled (Prompt fills it)."""
    return PARA_SYSTEM.format(lang=lang), PARA_USER.format(lang=lang, text="{text}")


def xling_templates(src, tgt):
    """Cross-lingual templates with {text} left unfilled."""
    return (XLING_SYSTEM.format(tgt=tgt),
            XLING_USER.format(src=src, tgt=tgt, text="{text}"))


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="path to a .gguf model")
@click.option("--in-csv", "in_csv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-csv", "out_csv", required=True)
@click.option("--text-col", default=None, help="input column (default: 'text' or first)")
@click.option("--out-col", default=None, help="output column (default: <tgt>_text)")
@click.option("--src", default="English", help="language of the input text")
@click.option("--tgt", default="Italian", help="target language")
@click.option("-n", "--n-paraphrases", default=4, show_default=True)
@click.option("--temp", default=0.8, show_default=True, help="sampling temperature for paraphrases")
@click.option("--n-ctx", default=0, show_default=True, help="context size (0 = model default)")
@click.option("--n-gpu-layers", default=0, help="0=CPU (default), -1=all layers on GPU", show_default=True)
@click.option("--limit", default=0, help="only first N rows (0 = all)")
def main(model, in_csv, out_csv, text_col, out_col, src, tgt, n_paraphrases, temp, n_ctx,
         n_gpu_layers, limit):
    """Paraphrase (and translate) a CSV column with a local GGUF model."""
    if not os.path.isfile(model):
        sys.exit(f"model not found: {model}\n  put a Qwen-2B GGUF there or pass --model PATH")
    with open(in_csv, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("input CSV is empty")
    text_col = text_col or ("text" if "text" in rows[0] else next(iter(rows[0])))
    out_col = out_col or f"{tgt.lower()}_text"
    translate = src.lower() != tgt.lower()
    if limit:
        rows = rows[:limit]

    llm = Llama(model_path=model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    llm.set_seed(0)

    para = Prompt(llm, *para_templates(src if translate else tgt))
    xlat = Prompt(llm, *xling_templates(src, tgt)) if translate else None

    fields = list(rows[0]) + [c for c in (out_col, "variant") if c not in rows[0]]
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in rows:
            text = (r.get(text_col) or "").strip()
            for v in range(n_paraphrases):
                out = para(llm, text, temp) if text else ""
                if xlat and out:
                    out = xlat(llm, out, 0.0)
                w.writerow({**r, out_col: out, "variant": v})
                fh.flush()
    click.echo(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
