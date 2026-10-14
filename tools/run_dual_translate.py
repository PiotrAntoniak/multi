"""run_dual_translate.py -- EN->PL paraphrase->translate for both models (dual-backend script).

Runs paraphrase_translate.py (gguf backend, GPU, explicit n_ctx=4096 since the qwen35 arch
does not expose context_length to llama.cpp) on flores_en_pl.csv:
  A) Qwen3.5-2B            -> out_qwen35_2b_pl.csv
  B) Qwen2.5-3B-Instruct   -> out_qwen25_3b_pl.csv
Captures each run's self-reported generation throughput and writes out_timing.json.

Usage: python tools/run_dual_translate.py
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

IN_CSV = "flores_en_pl.csv"
N = 4
LIMIT = 25
N_CTX = 4096
TOK_RE = re.compile(r"generated (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")

JOBS = [
    ("Qwen3.5-2B", os.path.join("models", "Qwen3.5-2B-Q4_K_M.gguf"), "out_qwen35_2b_pl.csv"),
    ("Qwen2.5-3B-Instruct",
     os.path.join("models", "qwen2.5-3b-instruct-q4_k_m.gguf"), "out_qwen25_3b_pl.csv"),
]


def main():
    results = {}
    for name, model, out_csv in JOBS:
        cmd = [
            sys.executable, os.path.join(ROOT, "paraphrase_translate.py"),
            "--backend", "gguf", "--model", model, "--n-ctx", str(N_CTX), "--n-gpu-layers", "-1",
            "--in-csv", IN_CSV, "--text-col", "en", "--src", "English", "--tgt", "Polish",
            "-n", str(N), "--limit", str(LIMIT), "--out-csv", out_csv,
        ]
        print(f"[{name}] running: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        secs = time.time() - t0
        print(proc.stdout, flush=True)
        if proc.stderr.strip():
            print("STDERR:", proc.stderr[-2000:], flush=True)
        m = TOK_RE.search(proc.stdout or "")
        entry = {
            "model": model, "out_csv": out_csv, "rc": proc.returncode,
            "wall_seconds": round(secs, 2), "n_paraphrases": N, "limit": LIMIT, "n_ctx": N_CTX,
        }
        if m:
            entry.update(output_tokens=int(m.group(1)), gen_seconds=float(m.group(2)),
                         tok_per_s=float(m.group(3)))
        results[name] = entry
        print(f"[{name}] done rc={proc.returncode} wall={secs:.1f}s {m.group(0) if m else ''}",
              flush=True)
        time.sleep(5)

    with open(os.path.join(ROOT, "out_timing.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("wrote out_timing.json", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
