"""run_translate.py -- run the paraphrase->translate pipeline for both models, timed.

Runs gguf_translate.py (paraphrase EN, then translate EN->PL) on flores_en_pl.csv for
  A) Qwen3.5-2B            -> out_qwen35_2b_pl.csv
  B) Qwen2.5-1.5B-Instruct -> out_qwen25_15b_pl.csv
on the GPU, measuring wall-clock seconds per model and writing out_timing.json with an
approximate generation throughput (output tokens re-tokenized with the model's own vocab,
divided by wall seconds -- prompt prefix is KV-cached so wall time is generation-dominated).

Usage: python tools/run_translate.py
"""
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

IN_CSV = "flores_en_pl.csv"
N = 4
LIMIT = 25

JOBS = [
    ("Qwen3.5-2B", os.path.join("models", "Qwen3.5-2B-Q4_K_M.gguf"), "out_qwen35_2b_pl.csv"),
    ("Qwen2.5-1.5B-Instruct",
     os.path.join("models", "qwen2.5-1.5b-instruct-q4_k_m.gguf"), "out_qwen25_15b_pl.csv"),
]


def count_tokens(model_path, out_csv, col):
    from llama_cpp import Llama
    llm = Llama(model_path=os.path.join(ROOT, model_path), vocab_only=True, verbose=False)
    total = 0
    with open(os.path.join(ROOT, out_csv), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            txt = (row.get(col) or "").strip()
            if txt:
                total += len(llm.tokenize(txt.encode("utf-8")))
    del llm
    return total


def main():
    results = {}
    for name, model, out_csv in JOBS:
        cmd = [
            sys.executable, os.path.join(ROOT, "gguf_translate.py"),
            "--model", model, "--n-gpu-layers", "-1",
            "--in-csv", IN_CSV, "--text-col", "en", "--src", "English", "--tgt", "Polish",
            "-n", str(N), "--limit", str(LIMIT), "--out-csv", out_csv,
        ]
        print(f"[{name}] running: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        secs = time.time() - t0
        if rc != 0:
            results[name] = {"error": f"exit {rc}", "seconds": secs, "out_csv": out_csv}
            print(f"[{name}] FAILED rc={rc}", flush=True)
            continue
        tok = count_tokens(model, out_csv, "polish_text")
        results[name] = {
            "model": model, "out_csv": out_csv, "seconds": round(secs, 2),
            "output_tokens": tok, "tok_per_s": round(tok / secs, 2) if secs else None,
            "n_paraphrases": N, "limit": LIMIT,
        }
        print(f"[{name}] done in {secs:.1f}s, {tok} out-tokens, {tok/secs:.2f} tok/s", flush=True)
        time.sleep(5)

    with open(os.path.join(ROOT, "out_timing.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("wrote out_timing.json", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
