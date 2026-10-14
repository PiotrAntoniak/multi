"""fetch_gguf.py -- download the Qwen3.5-2B GGUF quantizations for the quant-vs-quality check.

Usage:  python tools/fetch_gguf.py [dest_dir]
Downloads Q4_K_M, Q8_0 and BF16 (raw 16-bit) from unsloth/Qwen3.5-2B-GGUF.
"""
import sys

from huggingface_hub import hf_hub_download

REPO = "unsloth/Qwen3.5-2B-GGUF"
FILES = ["Qwen3.5-2B-Q4_K_M.gguf", "Qwen3.5-2B-Q8_0.gguf", "Qwen3.5-2B-BF16.gguf"]


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "models"
    for f in FILES:
        print("fetching", f, flush=True)
        print("ok", hf_hub_download(REPO, f, local_dir=dest), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
