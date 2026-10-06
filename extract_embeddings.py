"""Four pooling modes of EuroBERT-210m sentence embeddings.

EuroBERT-210m has no dedicated classification token and its tokenizer does not
prepend a leading special token.  This script materializes FOUR poolings of the
LAST hidden state in one pass, using TWO forwards per batch for exactness (the
plain forward serves mean/lead/eos; the prepended forward serves bos):

  * mean : masked mean over the PLAIN tokenization (content + appended end token)
  * lead : last hidden state at position 0 of the PLAIN forward (no prepend)
  * bos  : position 0 of a forward over the PLAIN ids with the leading special id
           128000 manually PREPENDED (+ a leading 1 in the attention mask)
  * eos  : last non-pad position of the PLAIN forward (the appended end token)

The plain tokenization is truncated to MAX_LEN-1 = 511 before prepending, so the
final (prepended) model input is <= MAX_LEN = 512.  Texts are the aligned
flores200_en_it_de_fr.csv rows, preserving order.

Saves float32 (2009, 768) arrays, finite-checked, incrementally (one file per
mode+lang, so an interruption loses at most the file in flight) to
multi/embeddings/<mode>/emb_<lang>.npy for lang in {en,it,de,fr}.

After extraction it verifies the fresh `bos` arrays against the pre-existing root
files multi/embeddings/emb_<lang>.npy (the historical bos variant): max absolute
difference per language.  If the worst difference is <= 1e-3 the root files are
MOVED into multi/embeddings/bos/ (replacing the fresh duplicates; root
meaning.npy is left untouched).  Otherwise the script STOPS and moves nothing.

fp32 GPU (CPU fallback), inference_mode, OMP/MKL/OpenBLAS/NumExpr threads=2,
batch 64, no detached processes / no windows.

Usage: python extract_embeddings.py [--model-id ID] [--data CSV] [--out-dir DIR]
                                    [--pools mean,lead,bos,eos] [--dtype float32|bfloat16]
                                    [--no-migrate-bos]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)

from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "flores200_en_it_de_fr.csv"
OUT_DIR = ROOT / "embeddings"
MODEL_ID = "EuroBERT/EuroBERT-210m"
LANGS = ["en", "it", "de", "fr"]
ALL_POOLS = ["mean", "lead", "bos", "eos"]
POOLS = list(ALL_POOLS)
BATCH, MAX_LEN = 64, 512
LEAD_FALLBACK = 128000


def read_texts(path):
    """Read aligned CSV -> ({lang: [texts]}, n_rows), preserving row order."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {lang: [r[lang] for r in rows] for lang in LANGS}, len(rows)


def leading_id(tokenizer):
    """Return the tokenizer's leading special id, falling back to 128000."""
    lead = tokenizer.bos_token_id
    if lead is None:
        lead = LEAD_FALLBACK
    if lead not in set(tokenizer.get_vocab().values()):
        print(f"FATAL: leading special id {lead} not in vocab", file=sys.stderr)
        sys.exit(2)
    return lead


@torch.inference_mode()
def forward_pools(texts, tokenizer, model, device, lead, pools):
    """Return {pool: (n, hidden) float32 array} for one list of texts.

    Two forwards per batch: the PLAIN forward (content + appended end token,
    truncated to MAX_LEN-1) yields mean/lead/eos; a PREPENDED forward over the
    same ids with `lead` prepended yields bos.  lead is taken from the plain
    forward, never approximated from the prepended one.

    Only the requested `pools` are computed and returned (e.g. ``["mean", "bos"]``
    skips the plain-forward lead/eos slices), and the extra prepended forward runs
    only when ``bos`` is requested.
    """
    acc = {k: [] for k in pools}
    for i in range(0, len(texts), BATCH):
        # PLAIN tokenization: content plus the tokenizer's appended end token.
        batch = tokenizer(texts[i:i + BATCH], padding=True, truncation=True,
                          max_length=MAX_LEN - 1, return_tensors="pt")
        ids, mask = batch["input_ids"], batch["attention_mask"]

        h = model(input_ids=ids.to(device),
                  attention_mask=mask.to(device)).last_hidden_state
        m = mask.to(h.device).unsqueeze(-1)
        if "mean" in acc:
            acc["mean"].append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
        if "lead" in acc:
            acc["lead"].append(h[:, 0, :].float().cpu().numpy())
        if "eos" in acc:
            last = (mask.sum(1) - 1).to(h.device)
            acc["eos"].append(h[torch.arange(h.shape[0], device=h.device), last]
                              .float().cpu().numpy())

        # PREPENDED forward for bos: leading id 128000 then the content tokens.
        if "bos" in acc:
            lead_col = torch.full((ids.shape[0], 1), lead, dtype=ids.dtype)
            p_ids = torch.cat([lead_col, ids], dim=1)
            p_mask = torch.cat([torch.ones_like(lead_col), mask], dim=1).to(torch.long)
            assert int(p_ids[:, 0].min()) == lead and int(p_ids[:, 0].max()) == lead, \
                "position 0 does not hold the leading id for every row"
            hb = model(input_ids=p_ids.to(device),
                       attention_mask=p_mask.to(device)).last_hidden_state
            acc["bos"].append(hb[:, 0, :].float().cpu().numpy())

    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in acc.items()}


def verify_and_migrate_bos(worst_limit=1e-3):
    """Compare fresh bos arrays to the old root bos files, then move them in.

    Returns (worst_diff, per_lang, moved).  If the worst difference exceeds
    `worst_limit`, prints a STOP banner and exits WITHOUT moving anything.
    """
    print("\n--- bos vs pre-existing root embeddings/emb_<lang>.npy (expect ~0) ---")
    per_lang = {}
    worst = 0.0
    for lang in LANGS:
        fresh = OUT_DIR / "bos" / f"emb_{lang}.npy"
        old = OUT_DIR / f"emb_{lang}.npy"
        if not old.exists():
            print(f"emb_{lang}: root file already moved (skip compare)")
            continue
        got = np.load(fresh).astype(np.float64)
        ref = np.load(old).astype(np.float64)
        diff = float(np.abs(got - ref).max())
        per_lang[lang] = diff
        worst = max(worst, diff)
        print(f"emb_{lang}: shape={got.shape} max abs diff = {diff:.3e}")

    if not per_lang:
        print("no root bos files present; nothing to compare or move")
        return worst, per_lang, []

    print(f"worst max abs diff over {len(per_lang)} languages = {worst:.3e}")
    if worst > worst_limit:
        print(f"STOP: worst diff {worst:.3e} > {worst_limit:.0e}; "
              f"root files left untouched.", file=sys.stderr)
        sys.exit(3)

    print("diff within tolerance -> moving root bos files into embeddings/bos/")
    moved = []
    for lang in LANGS:
        old = OUT_DIR / f"emb_{lang}.npy"
        if not old.exists():
            continue
        dest = OUT_DIR / "bos" / f"emb_{lang}.npy"
        os.replace(old, dest)
        moved.append(f"emb_{lang}.npy")
        print(f"moved embeddings/emb_{lang}.npy -> embeddings/bos/emb_{lang}.npy")
    return worst, per_lang, moved


def parse_args(argv=None):
    """CLI with defaults identical to the pre-argparse behavior."""
    ap = argparse.ArgumentParser(
        description="EuroBERT sentence-embedding pooling extractor.")
    ap.add_argument("--model-id", default="EuroBERT/EuroBERT-210m",
                    help="HF model id (default EuroBERT/EuroBERT-210m)")
    ap.add_argument("--data", default="flores200_en_it_de_fr.csv",
                    help="aligned input CSV (default flores200_en_it_de_fr.csv)")
    ap.add_argument("--out-dir", default="embeddings",
                    help="output root; writes <out>/<mode>/emb_<lang>.npy (default embeddings)")
    ap.add_argument("--pools", default="mean,lead,bos,eos",
                    help="comma-separated poolings to compute (default mean,lead,bos,eos)")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="model dtype; bfloat16 needs CUDA, else falls back to float32")
    ap.add_argument("--no-migrate-bos", action="store_true",
                    help="skip verify_and_migrate_bos() after extraction")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    global DATA, OUT_DIR, MODEL_ID, POOLS
    DATA = Path(args.data)
    if not DATA.is_absolute():
        DATA = ROOT / DATA
    OUT_DIR = Path(args.out_dir)
    if not OUT_DIR.is_absolute():
        OUT_DIR = ROOT / OUT_DIR
    MODEL_ID = args.model_id
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    unknown = [p for p in pools if p not in ALL_POOLS]
    if unknown:
        raise SystemExit("unknown pools %r (choose from %r)" % (unknown, ALL_POOLS))
    POOLS = pools

    t0 = time.time()
    texts, n = read_texts(DATA)
    print(f"{DATA.name}: {n} rows x {LANGS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32
    if args.dtype == "bfloat16":
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        else:
            print("WARNING: --dtype bfloat16 requested but CUDA is unavailable; "
                  "falling back to float32", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lead = leading_id(tokenizer)
    example = tokenizer(texts[LANGS[0]][0], add_special_tokens=True)["input_ids"]
    print(f"tokenizer auto position 0 = {example[0]} "
          f"({tokenizer.convert_ids_to_tokens(example[0])!r}); "
          f"prepended leading id = {lead} "
          f"({tokenizer.convert_ids_to_tokens(lead)!r}); "
          f"len={len(example)} -> {len(example) + 1}")

    model = AutoModel.from_pretrained(MODEL_ID).to(device).to(torch_dtype).eval()
    print(f"{MODEL_ID} on {device}, dtype={torch_dtype}, "
          f"hidden_size = {model.config.hidden_size}, "
          f"batch={BATCH}, max_length={MAX_LEN} (plain tokenize to {MAX_LEN - 1}, "
          f"prepend +1), pools={POOLS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode in POOLS:
        (OUT_DIR / mode).mkdir(parents=True, exist_ok=True)

    norms = {mode: {} for mode in POOLS}
    for lang in LANGS:
        pools_out = forward_pools(texts[lang], tokenizer, model, device, lead, POOLS)
        for mode in POOLS:
            arr = pools_out[mode].astype(np.float32)
            path = OUT_DIR / mode / f"emb_{lang}.npy"
            np.save(path, arr)
            finite = bool(np.isfinite(arr).all())
            norms[mode][lang] = float(np.linalg.norm(arr, axis=1).mean())
            print(f"{mode}/emb_{lang}: shape={arr.shape} dtype={arr.dtype} "
                  f"all_finite={finite} -> {path}")
            if not finite:
                print(f"FATAL: non-finite values in {path}", file=sys.stderr)
                sys.exit(1)

    print("\n--- scale sanity (mean L2 norm per mode over each language) ---")
    for mode in POOLS:
        vals = [norms[mode][lang] for lang in LANGS]
        pretty = " ".join(f"{lang}={norms[mode][lang]:.3f}" for lang in LANGS)
        print(f"{mode}: mean over langs={np.mean(vals):.3f}  |  {pretty}")

    if "bos" in POOLS and not args.no_migrate_bos:
        verify_and_migrate_bos()
    elif "bos" not in POOLS:
        print("bos not requested; skipping verify_and_migrate_bos()")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
