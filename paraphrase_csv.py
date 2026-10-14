"""Minimal CSV -> paraphrases generator.

Edit the two paths below, then run::

    python paraphrase_csv.py

It reads a CSV with a text column, generates one paraphrase (or a translation, if you set
MODE="crosslingual") per row with a local instruct LLM, and writes a CSV with the original text plus the
generation. Model path and file paths are HARD-CODED on purpose (minimal, no CLI needed).

Requires: torch + transformers (and the model cached / downloadable).
"""
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from paraphrase_prompts import build_messages  # prompt templates live in paraphrase_prompts.py

# --------------------------------------------------------------------------- #
# HARD-CODED SETTINGS -- edit these
# --------------------------------------------------------------------------- #
MODEL_PATH = r"HuggingFaceTB/SmolLM3-3B"   # local path OR HF id (e.g. r"C:\models\SmolLM3-3B")
IN_CSV     = r"input.csv"                  # CSV with a text column
OUT_CSV    = r"paraphrases.csv"
TEXT_COL   = "text"                        # name of the text column in IN_CSV
MODE       = "paraphrase"                  # "paraphrase" (same language) or "crosslingual"
LANG       = "English"                     # paraphrase: the text's language
SRC, TGT   = "English", "Italian"          # crosslingual: source / target language names
MAX_NEW_TOKENS = 384
BATCH      = 16
# --------------------------------------------------------------------------- #

EVAL_KW = dict(do_sample=False)  # greedy; set do_sample=True, temperature=0.7 for diversity


def main():
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(IN_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    texts = [r[TEXT_COL] for r in rows]
    print(f"{len(texts)} rows from {IN_CSV} | model={MODEL_PATH} ({device}, bf16) | mode={MODE}")

    out = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        prompts = []
        for t in chunk:
            msgs = (build_messages("crosslingual", t, src=SRC, tgt=TGT) if MODE == "crosslingual"
                    else build_messages("paraphrase", t, lang=LANG))
            try:   # Qwen3-style: disable the reasoning block
                p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)
            except Exception:  # templates that don't take enable_thinking (e.g. SmolLM3)
                p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompts.append(p)
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, **EVAL_KW)
        for j, t in enumerate(chunk):
            text = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append({TEXT_COL: t, "paraphrase": text.strip()})   # change key if you like
        print(f"  {min(i + BATCH, len(texts))}/{len(texts)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[TEXT_COL, "paraphrase"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
