"""fetch_qwen25_3b_gguf.py -- download the Qwen2.5-3B-Instruct Q4_K_M GGUF.

Usage:  python tools/fetch_qwen25_3b_gguf.py [dest_dir]
"""
import sys

from huggingface_hub import hf_hub_download

REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "models"
    print("fetching", FILE, flush=True)
    print("ok", hf_hub_download(REPO, FILE, local_dir=dest), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
