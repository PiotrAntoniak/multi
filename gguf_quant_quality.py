"""gguf_quant_quality.py -- EN->IT translation QUALITY of GGUF quantizations.

For each GGUF in ``models/Qwen3.5-2B-*.gguf`` (Q4_K_M, Q8_0, BF16):

  * load fully on GPU (``n_gpu_layers=-1`` -> offload all layers to CUDA, sm_86 here;
    ``--n-threads`` stays modest as the CPU only assists sampling),
  * greedy-decode (temp 0) English -> Italian with the repo cross-lingual prompt
    (``paraphrase_prompts.build_messages("crosslingual", ...)`` rendered through the model's
    chat template -- the ``Prompt`` class from ``gguf_translate.py`` is reused, so the fixed
    system+user prefix is encoded once and its KV cache reused across sentences),
  * score the hypotheses against the FLORES Italian reference with **sacrebleu**
    (``corpus_chrf`` word_order=0 -> chrF, char_order=6 word_order=2 -> chrF++, ``corpus_bleu``).

Writes ``gguf_quant_quality.json`` + ``gguf_quant_quality.md`` and prints progress (flushed).

Run detached, e.g.::

    Start-Process -FilePath C:\\Users\\pedro\\anaconda3\\python.exe `
      -ArgumentList '-u', 'gguf_quant_quality.py' -WindowStyle Hidden `
      -RedirectStandardOutput $log -RedirectStandardError $err

Note: the model has ``n_ctx_train = 262144``; ``--n-ctx`` defaults to 2048 so the KV cache
stays small.
"""
import argparse
import csv
import json
import os
import sys
import time

from llama_cpp import Llama
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paraphrase_prompts import build_messages          # noqa: E402
from gguf_translate import clean, Prompt               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA = os.path.join(HERE, "flores200_en_it_de_fr.csv")   # latin-1, cols: ...,en,it,...
DEFAULT_MODELS = [
    ("Q4_K_M", os.path.join(HERE, "models", "Qwen3.5-2B-Q4_K_M.gguf")),
    ("Q8_0",   os.path.join(HERE, "models", "Qwen3.5-2B-Q8_0.gguf")),
    ("BF16",   os.path.join(HERE, "models", "Qwen3.5-2B-BF16.gguf")),
]

SRC, TGT = "English", "Italian"

# Qwen3.5 chat template disables reasoning by prefilling an EMPTY think block when
# enable_thinking is false. The C template engine in llama.cpp omits that prefill, so we
# add it ourselves after "<|im_start|>assistant\n" to keep greedy decoding on the answer
# (no leaked reasoning, much faster).
NO_THINK = b"<think>\n\n</think>\n\n"


def log(msg):
    print(msg, flush=True)


def load_flores(path, n):
    """Return (en, it) lists of the first ``n`` non-empty rows. File is latin-1 encoded."""
    ens, its = [], []
    with open(path, encoding="latin-1", newline="") as fh:
        for row in csv.DictReader(fh):
            en = (row.get("en") or "").strip()
            it = (row.get("it") or "").strip()
            if not en or not it:
                continue
            ens.append(en)
            its.append(it)
            if len(ens) >= n:
                break
    return ens, its


def postprocess(hyp):
    """Keep only the final answer; drop any leaked reasoning block / stray quotes."""
    if not hyp:
        return ""
    if "</think>" in hyp:
        hyp = hyp.rsplit("</think>", 1)[1]
    elif "<think>" in hyp:
        hyp = hyp.split("<think>", 1)[0]
    return clean(hyp).strip()


def build_translator(llm):
    """Prompt whose fixed prefix is cached (reuses gguf_translate.Prompt)."""
    # Pass the literal slot so Prompt can split it out and reuse the prefix KV cache;
    # this is equivalent to build_messages(..., text=<EN>) with <EN> filled in per sentence.
    msgs = build_messages("crosslingual", text="{text}", src=SRC, tgt=TGT)
    promp = Prompt(llm, msgs[0]["content"], msgs[1]["content"])
    promp.post = promp.post + NO_THINK       # force the no-reasoning assistant prefix
    return promp


def translate(promp, llm, text, temp=0.0):
    """Greedy decode one sentence; return (hypothesis, n_generated_tokens, seconds)."""
    llm.load_state(promp.state)
    head = llm.tokenize(text.encode() + promp.post, add_bos=False, special=True)
    llm.eval(head)
    n = promp.n_pre + len(head)
    n_ctx = llm.n_ctx()
    out = []
    t0 = time.time()
    while n < n_ctx - 1:                        # stop at EOS or when context is full
        tok = llm.sample(temp=temp)
        if tok == llm.token_eos():
            break
        out.append(tok)
        llm.eval([tok])
        n += 1
    dt = time.time() - t0
    return postprocess(llm.detokenize(out).decode("utf-8", "replace")), len(out), dt


def score(hyps, refs):
    """chrF (word_order 0), chrF++ (word_order 2), BLEU vs the human reference."""
    return {
        "chrf": sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=0).score,
        "chrfpp": sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2).score,
        "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
    }


def run_model(tag, path, ens, its, args):
    size_mb = os.path.getsize(path) / 1e6
    dev = "GPU (all layers, CUDA)" if args.n_gpu_layers != 0 else "CPU"
    log(f"[{tag}] loading {os.path.basename(path)} ({size_mb:.0f} MB) on {dev} "
        f"(n_gpu_layers={args.n_gpu_layers}, n_threads={args.n_threads}, n_ctx={args.n_ctx}) ...")
    llm = Llama(model_path=path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                n_threads=args.n_threads, verbose=False)
    llm.set_seed(0)
    promp = build_translator(llm)

    hyps, toks, gen_secs = [], 0, 0.0
    examples = []
    t_all = time.time()
    for i, en in enumerate(ens, 1):
        hyp, nt, dt = translate(promp, llm, en)
        hyps.append(hyp)
        toks += nt
        gen_secs += dt
        if i <= 3:
            examples.append({"en": en, "hyp": hyp, "ref": its[i - 1]})
        if i % 10 == 0 or i == len(ens):
            rate = toks / gen_secs if gen_secs > 0 else 0.0
            log(f"[{tag}] {i}/{len(ens)}  {rate:.1f} tok/s  last: {hyp[:70]!r}")
    wall = time.time() - t_all

    sc = score(hyps, its)
    res = {
        "tag": tag,
        "file": os.path.basename(path),
        "size_mb": round(size_mb, 1),
        "n": len(ens),
        "chrf": sc["chrf"],
        "chrfpp": sc["chrfpp"],
        "bleu": sc["bleu"],
        "tok_per_s": toks / gen_secs if gen_secs > 0 else None,
        "gen_seconds": round(gen_secs, 1),
        "wall_seconds": round(wall, 1),
        "total_tokens": toks,
        "examples": examples,
        "hyps": args.keep_hyps and hyps or None,
    }
    log(f"[{tag}] DONE  chrF={sc['chrf']:.2f}  chrF++={sc['chrfpp']:.2f}  BLEU={sc['bleu']:.2f}  "
        f"tok/s={res['tok_per_s']:.1f}  wall={wall:.0f}s")
    del llm
    return res


def write_report(results, out_dir):
    js = os.path.join(out_dir, "gguf_quant_quality.json")
    md = os.path.join(out_dir, "gguf_quant_quality.md")

    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"config": {"src": SRC, "tgt": TGT,
                              "n": results[0]["n"] if results else 0,
                              "decoding": "greedy (temp 0), until EOS",
                              "prompt": "crosslingual"},
                   "results": [{k: v for k, v in r.items() if k != "examples"} for r in results],
                   "examples": {r["tag"]: r["examples"] for r in results}},
                  fh, ensure_ascii=False, indent=2)

    lines = ["# GGUF quantization translation quality (EN->IT, FLORES)",
             "",
             f"- data: `{os.path.basename(DEFAULT_DATA)}` (latin-1), first "
             f"{results[0]['n'] if results else 0} sentences",
             f"- prompt: `crosslingual` (`paraphrase_prompts.build_messages`), model chat template",
             "- decoding: greedy (temp 0) until EOS, full GPU offload (n_gpu_layers=-1)",
             "- metrics: sacrebleu chrF (char_order 6, word_order 0), chrF++ (word_order 2), BLEU",
             "",
             "| quant | size MB | chrF | chrF++ | BLEU | avg tok/s |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['tag']} | {r['size_mb']:.0f} | {r['chrf']:.2f} | {r['chrfpp']:.2f} | "
                     f"{r['bleu']:.2f} | {r['tok_per_s']:.1f} |")
    lines += ["", "## Examples", ""]
    for r in results:
        lines.append(f"### {r['tag']}")
        for ex in r["examples"]:
            lines.append(f"- **EN**: {ex['en']}")
            lines.append(f"  - **hyp**: {ex['hyp']}")
            lines.append(f"  - **ref**: {ex['ref']}")
        lines.append("")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return js, md


def main():
    ap = argparse.ArgumentParser(description="EN->IT translation quality of GGUF quantizations.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=150, help="number of EN sentences (default 150)")
    ap.add_argument("--n-ctx", dest="n_ctx", type=int, default=2048)
    ap.add_argument("--n-threads", dest="n_threads", type=int, default=4)
    ap.add_argument("--n-gpu-layers", dest="n_gpu_layers", type=int, default=-1,
                    help="layers to offload to GPU (-1 = all; 0 = CPU only; default -1)")
    ap.add_argument("--out-dir", dest="out_dir", default=HERE)
    ap.add_argument("--models", nargs="*", default=None,
                    help="optional tag=path overrides; default: the three Qwen3.5-2B quants")
    ap.add_argument("--keep-hyps", action="store_true", help="also store all hypotheses in the JSON")
    ap.add_argument("--resume", action="store_true",
                    help="skip models already present in gguf_quant_quality.partial.json")
    args = ap.parse_args()

    if args.models:
        models = []
        for spec in args.models:
            tag, _, path = spec.partition("=")
            models.append((tag, path))
    else:
        models = DEFAULT_MODELS

    for tag, path in models:
        if not os.path.isfile(path):
            sys.exit(f"model not found: {path}")

    ens, its = load_flores(args.data, args.n)
    log(f"data: {os.path.basename(args.data)}  -> {len(ens)} EN/IT sentence pairs")

    # Per-model checkpoint: a resumable run never redoes a model that already finished.
    partial = os.path.join(args.out_dir, "gguf_quant_quality.partial.json")
    done = {}
    if args.resume and os.path.isfile(partial):
        with open(partial, encoding="utf-8") as fh:
            for r in json.load(fh):
                done[r["tag"]] = r
        log(f"resume: {len(done)} model(s) already scored: {list(done)}")

    results = []
    t0 = time.time()
    for tag, path in models:
        if tag in done:
            results.append(done[tag])
            log(f"[{tag}] skipped (already scored: chrF={done[tag]['chrf']:.2f})")
            continue
        results.append(run_model(tag, path, ens, its, args))
        with open(partial, "w", encoding="utf-8") as fh:   # checkpoint after each model
            json.dump([{k: v for k, v in r.items() if k != "hyps"} for r in results],
                      fh, ensure_ascii=False, indent=2)
    js, md = write_report(results, args.out_dir)
    log(f"wrote {js}")
    log(f"wrote {md}")
    log(f"total wall: {time.time() - t0:.0f}s")

    log("")
    log("| quant | size MB | chrF | chrF++ | BLEU | tok/s |")
    log("|---|---|---|---|---|---|")
    for r in results:
        log(f"| {r['tag']} | {r['size_mb']:.0f} | {r['chrf']:.2f} | {r['chrfpp']:.2f} | "
            f"{r['bleu']:.2f} | {r['tok_per_s']:.1f} |")


if __name__ == "__main__":
    main()
