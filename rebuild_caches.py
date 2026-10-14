"""Rebuild cached artifacts lost in the 2026-10-11 cleanup (see EXPERIMENTS.md RECOVERY NOTE).

This script ONLY creates files. It never deletes or moves anything. It is resumable:
every target is skipped when it already exists (and, for .npy, is finite); progress and
status are written after each artifact.

Artifacts produced (all under this script's directory `multi/`):

  1) EuroBERT-210m sentence embeddings (mean + bos pooling) for the sentiment/emotion lines
       sst2_emb/{mean,bos}_{train,val,test}.npy      from sst2.csv   (column `sentence`)
       emotion_emb/{mean,bos}_{train,val,test}.npy   from emotion.csv (column `text`)

  2) Helsinki-NLP/opus-mt-en-{it,de,fr} translations of the VALIDATION and TEST splits
       translated/sst2_<lang>_{validation,test}.csv     (columns: text,label; row order kept)
       translated/emotion_<lang>_{validation,test}.csv  (columns: text,label; row order kept)

  3) EuroBERT-210m embeddings (mean + bos) of the translated text
       translated_emb/sst2_{mean,bos}_{lang}_{validation,test}.npy
       translated_emb/emotion_{mean,bos}_{lang}_{validation,test}.npy

Embedding recipe is IDENTICAL to `xgboost_sst2.py::embed_split`:
  * EuroBERT/EuroBERT-210m native AutoModel + AutoTokenizer, fp32, GPU.
  * batch 256, max_length 512.
  * mean : masked mean over all non-pad tokens of the PLAIN forward (the tokenizer's
           appended end token is included).
  * bos  : position 0 of the last hidden state of a forward over the PLAIN ids with the
           special id 128000 manually PREPENDED (+ a leading 1 in the attention mask).

Constraints: GPU for MT + embedding, torch threads = 2, resumable, no deletions.

Logs / PID / status are written to %TEMP%\\opencode\\:
  rebuild_caches.log, rebuild_caches.pid, rebuild_caches.status.json

Run detached (see launch_hidden.vbs):
  wscript //B //Nologo launch_hidden.vbs "<python> -u rebuild_caches.py"
"""
import os
import sys

# Thread caps must be set before numpy/torch import to take effect (torch threads=2).
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
from datetime import datetime

import numpy as np
import torch

torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# Paths, constants
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.join(os.environ.get("TEMP", BASE), "opencode")
os.makedirs(TEMP, exist_ok=True)
LOG_PATH = os.path.join(TEMP, "rebuild_caches.log")
PID_PATH = os.path.join(TEMP, "rebuild_caches.pid")
STATUS_PATH = os.path.join(TEMP, "rebuild_caches.status.json")

MODEL_ID = "EuroBERT/EuroBERT-210m"
BOS_ID = 128000
BATCH, MAX_LEN = 256, 512
MT_BATCH = 64
MT_MAX_LEN = 512
LANGS = ["it", "de", "fr"]
SENTIMENT_DATASETS = {          # name -> (csv, text column, out dir)
    "sst2": ("sst2.csv", "sentence", "sst2_emb"),
    "emotion": ("emotion.csv", "text", "emotion_emb"),
}
SPLIT_FILE = {"train": "train", "validation": "val", "test": "test"}

PID = os.getpid()
START = time.time()
STATUS = {"pid": PID, "started": datetime.now().isoformat(timespec="seconds"),
          "artifacts": [], "current": None}


# ---------------------------------------------------------------------------
# Logging / status
# ---------------------------------------------------------------------------
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def write_status(stage, **kw):
    STATUS["stage"] = stage
    STATUS["updated"] = datetime.now().isoformat(timespec="seconds")
    STATUS["elapsed_s"] = round(time.time() - START, 1)
    STATUS.update(kw)
    _atomic_json(STATUS_PATH, STATUS)


def note_artifact(path):
    rel = os.path.relpath(path, BASE)
    STATUS["artifacts"].append({"path": rel, "at": datetime.now().isoformat(timespec="seconds")})
    log(f"ARTIFACT ready: {rel}")
    write_status("artifact", current=rel)


# ---------------------------------------------------------------------------
# Cache helpers (create-only, resumable)
# ---------------------------------------------------------------------------
def cached_npy(path):
    """Return the array at `path` if it exists and is finite, else None."""
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
    except Exception as e:  # noqa: BLE001
        log(f"  [cache] unreadable {os.path.basename(path)} ({type(e).__name__}); recomputing")
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or not np.isfinite(arr).all():
        log(f"  [cache] non-finite/empty {os.path.basename(path)} {arr.shape}; recomputing")
        return None
    log(f"  [cache] hit {os.path.basename(path)} {arr.shape} {arr.dtype}")
    return arr


def csv_is_complete(path, expected_rows):
    """True if `path` exists, is non-empty and has `expected_rows` data rows."""
    if not os.path.exists(path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        log(f"  [cache] unreadable CSV {os.path.basename(path)} ({type(e).__name__})")
        return False
    ok = len(df) == expected_rows and "text" in df.columns and "label" in df.columns
    if not ok:
        log(f"  [cache] CSV {os.path.basename(path)} rows={len(df)} != {expected_rows}; rebuild")
    return ok


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite embeddings written to {path}")


# ---------------------------------------------------------------------------
# Embedding (identical pooling to xgboost_sst2.py::embed_split)
# ---------------------------------------------------------------------------
def _coerce_texts(texts):
    """Coerce every element to a plain str, replacing None/NaN with "".

    The tokenizer (and HF TextEncodeInput) rejects non-string elements; translated
    CSVs can carry NaN/None (and float coercion) in the `text` column.
    """
    out = []
    for t in texts:
        if t is None:
            out.append("")
        elif isinstance(t, float) and (t != t):          # float NaN
            out.append("")
        elif isinstance(t, np.floating) and bool(np.isnan(t)):
            out.append("")
        else:
            out.append(str(t))
    return out


def embed_texts(mode, texts, tokenizer, model, device, label=""):
    texts = _coerce_texts(texts)
    n = len(texts)
    n_batches = (n + BATCH - 1) // BATCH
    acc = []
    with torch.no_grad():
        for b in range(n_batches):
            chunk = texts[b * BATCH:(b + 1) * BATCH]
            max_len = MAX_LEN - 1 if mode == "bos" else MAX_LEN
            tok = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)
            if mode == "mean":
                h = model(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).to(h.dtype)
                pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
            elif mode == "bos":
                lead = torch.full((ids.shape[0], 1), BOS_ID, dtype=ids.dtype, device=device)
                p_ids = torch.cat([lead, ids], dim=1)
                p_mask = torch.cat([torch.ones_like(lead), mask], dim=1)
                h = model(input_ids=p_ids, attention_mask=p_mask).last_hidden_state
                pooled = h[:, 0, :].float().cpu().numpy()
            else:
                raise ValueError(f"unknown mode {mode!r}")
            acc.append(pooled.astype(np.float32))
            write_status("embedding", current=label, batch=b + 1, total_batches=n_batches,
                         rows_done=min(n, (b + 1) * BATCH), rows_total=n)
            if (b + 1) % 10 == 0 or b + 1 == n_batches:
                log(f"  [embed {label}] batch {b + 1}/{n_batches} "
                    f"({min(n, (b + 1) * BATCH)}/{n})")
    return np.concatenate(acc, 0).astype(np.float32) if acc else np.zeros((0, 0), np.float32)


def embed_cached(mode, texts, tokenizer, model, device, out_path, label):
    arr = cached_npy(out_path)
    if arr is not None:
        return arr
    log(f"  [embed {label}] computing {len(texts)} rows -> {os.path.relpath(out_path, BASE)}")
    arr = embed_texts(mode, texts, tokenizer, model, device, label=label)
    if arr.shape[0] != len(texts):
        raise RuntimeError(f"row mismatch {label}: {arr.shape[0]} != {len(texts)}")
    save_npy(out_path, arr)
    log(f"  [embed {label}] saved {arr.shape} {arr.dtype}")
    note_artifact(out_path)
    return arr


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate_texts(texts, mt_tok, mt_model, device):
    out = []
    n = len(texts)
    n_batches = (n + MT_BATCH - 1) // MT_BATCH
    with torch.no_grad():
        for b in range(n_batches):
            chunk = texts[b * MT_BATCH:(b + 1) * MT_BATCH]
            enc = mt_tok(chunk, return_tensors="pt", padding=True, truncation=True,
                         max_length=MT_MAX_LEN).to(device)
            gen = mt_model.generate(**enc, max_length=MT_MAX_LEN, num_beams=1, do_sample=False)
            out.extend(mt_tok.batch_decode(gen, skip_special_tokens=True))
            if (b + 1) % 10 == 0 or b + 1 == n_batches:
                log(f"    [MT] batch {b + 1}/{n_batches} ({len(out)}/{n})")
    if len(out) != n:
        raise RuntimeError(f"translation row mismatch: {len(out)} != {n}")
    return out


def translate_cached(texts, labels, mt_tok, mt_model, device, out_csv):
    if csv_is_complete(out_csv, len(texts)):
        log(f"  [MT] cache hit {os.path.relpath(out_csv, BASE)} ({len(texts)} rows)")
        return
    log(f"  [MT] translating {len(texts)} rows -> {os.path.relpath(out_csv, BASE)}")
    out = translate_texts(texts, mt_tok, mt_model, device)
    import pandas as pd
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tmp = out_csv + ".tmp"
    pd.DataFrame({"text": out, "label": labels}).to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    note_artifact(out_csv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    with open(PID_PATH, "w") as f:
        f.write(str(PID))
    log(f"=== rebuild_caches start pid={PID} base={BASE} ===")
    log(f"temp dir = {TEMP}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        log("WARNING: CUDA not available; running on CPU (slow)")
    write_status("start")
    torch.set_num_threads(2)

    import pandas as pd  # noqa: F401  (ensure pandas importable early)

    # ---- load data once ----
    data = {}
    for name, (csv, textcol, outdir) in SENTIMENT_DATASETS.items():
        df = pd.read_csv(os.path.join(BASE, csv))
        splits = {}
        for sp in ("train", "validation", "test"):
            sub = df[df["split"] == sp]
            splits[sp] = (list(sub[textcol]), np.asarray(sub["label"], dtype=np.int64))
        data[name] = dict(csv=csv, textcol=textcol, outdir=outdir, splits=splits)
        log(f"loaded {csv}: train={len(splits['train'][0])} "
            f"val={len(splits['validation'][0])} test={len(splits['test'][0])}")

    # ---- load EuroBERT once ----
    from transformers import AutoModel, AutoTokenizer
    log(f"loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(torch.float32).to(device).eval()
    log(f"{MODEL_ID} on {device} fp32, hidden={model.config.hidden_size}")

    # ================= PART 1: source-language sentiment embeddings ========
    write_status("part1_sentiment_embeddings")
    for name in SENTIMENT_DATASETS:
        outdir = os.path.join(BASE, data[name]["outdir"])
        os.makedirs(outdir, exist_ok=True)
        for mode in ("mean", "bos"):
            for sp in ("train", "validation", "test"):
                texts, _ = data[name]["splits"][sp]
                fname = f"{mode}_{SPLIT_FILE[sp]}.npy"
                embed_cached(mode, texts, tokenizer, model, device,
                             os.path.join(outdir, fname), f"{name}/{mode}/{sp}")
    log("PART 1 done (source sentiment/emotion embeddings)")

    # ================= PART 2: translations + translated embeddings ========
    write_status("part2_translations")
    mt_loaded = None
    from transformers import AutoModelForSeq2SeqLM
    for lang in LANGS:
        # ---- translations (validation + test only) ----
        mt_id = f"Helsinki-NLP/opus-mt-en-{lang}"
        log(f"loading {mt_id} ...")
        if mt_loaded is not None:
            del mt_loaded
            torch.cuda.empty_cache() if device == "cuda" else None
        mt_tok = AutoTokenizer.from_pretrained(mt_id)
        mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_id).to(device)
        mt_model = mt_model.half().eval() if device == "cuda" else mt_model.eval()
        mt_loaded = mt_model

        for name, d in SENTIMENT_DATASETS.items():
            for sp in ("validation", "test"):
                texts, labels = data[name]["splits"][sp]
                out_csv = os.path.join(BASE, "translated", f"{name}_{lang}_{sp}.csv")
                write_status("translating", current=f"{name}/{lang}/{sp}")
                translate_cached(texts, labels, mt_tok, mt_model, device, out_csv)

        # ---- embeddings of translated text (reuse EuroBERT) ----
        for name, d in SENTIMENT_DATASETS.items():
            for sp in ("validation", "test"):
                csv = os.path.join(BASE, "translated", f"{name}_{lang}_{sp}.csv")
                # Read the `text` column as pure str; never let the `label` column bleed in.
                tdf = pd.read_csv(csv, dtype=str, usecols=["text"]).fillna("")
                ttexts = tdf["text"].astype(str).tolist()
                for mode in ("mean", "bos"):
                    out_npy = os.path.join(BASE, "translated_emb",
                                           f"{name}_{mode}_{lang}_{sp}.npy")
                    embed_cached(mode, ttexts, tokenizer, model, device, out_npy,
                                 f"translated/{name}/{mode}/{lang}/{sp}")

    write_status("done")
    log(f"=== ALL DONE in {time.time() - START:.0f}s; {len(STATUS['artifacts'])} artifacts this run ===")
    log(f"total artifacts on disk check: sst2_emb/ emotion_emb/ translated/ translated_emb/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        log("FATAL: " + tb)
        write_status("error", error=f"{type(e).__name__}: {e}", traceback=tb)
        sys.exit(1)
