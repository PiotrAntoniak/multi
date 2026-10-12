"""
mt_compare.py
=============
Compare three BILINGUAL Italian/German/French -> English MT models against a
SINGLE MULTILINGUAL many -> English model, on FLORES (parallel, so gold English
is available).

Models
------
Bilingual   : Helsinki-NLP/opus-mt-it-en, opus-mt-de-en, opus-mt-fr-en
Multilingual: Helsinki-NLP/opus-mt-mul-en   (one many->en model handling it/de/fr)

Data
----
flores200_en_it_de_fr.csv  (columns: split,id,en,it,de,fr,...; 2006 rows)
The `en` column is the gold reference.  We translate the it/de/fr columns -> en
with BOTH families (each language gets a bilingual AND a multilingual result).

Pipeline
--------
transformers seq2seq MarianMT, fp16, greedy (num_beams=1), batch 64,
max_length=512, GPU, torch threads=2.  Outputs are cached in %TEMP%\opencode.

Metrics
-------
sacrebleu corpus BLEU and chrF (detokenized default settings), per language and
per family, plus the 3-language macro-average and the bilingual->multilingual
delta.

Outputs (create-only): mt_compare.json, mt_compare.md, mt_compare.png
"""

import os
import sys
import json
import time
import hashlib
import tempfile

import numpy as np
import pandas as pd

# ---- fixed runtime settings -------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

torch.set_num_threads(2)

# ---- configuration ----------------------------------------------------------
DATA_CSV   = "flores200_en_it_de_fr.csv"
N_ROWS     = None          # None => use ALL 2006 rows (state in report)
BATCH_SIZE = 64
MAX_LENGTH = 512
NUM_BEAMS  = 1             # greedy
USE_FP16   = True

LANGS = ["it", "de", "fr"]

BILINGUAL = {
    "it": "Helsinki-NLP/opus-mt-it-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}
MULTILINGUAL = {
    "it": "Helsinki-NLP/opus-mt-mul-en",
    "de": "Helsinki-NLP/opus-mt-mul-en",
    "fr": "Helsinki-NLP/opus-mt-mul-en",
}
FAMILIES = {"bilingual": BILINGUAL, "multilingual": MULTILINGUAL}

CACHE_DIR = os.path.join(tempfile.gettempdir(), "opencode")
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _key(model_name, texts):
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    safe = model_name.replace("/", "__")
    return os.path.join(CACHE_DIR, f"mtcache_{safe}_{len(texts)}_{h}.json")


def translate(model_name, texts):
    """Translate `texts` with `model_name`; cache the outputs on disk."""
    cpath = _key(model_name, texts)
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            out = json.load(f)
        print(f"    [cache] {model_name}  ({len(out)} lines)")
        return out

    print(f"    [load ] {model_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_FP16 and DEVICE == "cuda":
        model = model.half()
    model = model.to(DEVICE).eval()

    preds = []
    t0 = time.time()
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            gen = model.generate(**enc, num_beams=NUM_BEAMS,
                                 max_length=MAX_LENGTH, do_sample=False)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
            done = min(i + BATCH_SIZE, n)
            if (i // BATCH_SIZE) % 8 == 0 or done == n:
                print(f"      {done}/{n}  ({time.time()-t0:.1f}s)", flush=True)

    # free GPU memory before the next model
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"    [save ] {cpath}")
    return preds


def main():
    import sacrebleu  # imported late so install errors are obvious

    print(f"device={DEVICE}  fp16={USE_FP16}  threads={torch.get_num_threads()}")
    df = pd.read_csv(DATA_CSV)
    if N_ROWS:
        df = df.iloc[:N_ROWS].reset_index(drop=True)
    n_rows = len(df)
    refs = df["en"].astype(str).tolist()
    print(f"loaded {DATA_CSV}: {n_rows} rows (subset={bool(N_ROWS)})")

    # ---- translate ----------------------------------------------------------
    preds = {}  # preds[family][lang] = list[str]
    for family, models in FAMILIES.items():
        preds[family] = {}
        for lang in LANGS:
            model_name = models[lang]
            print(f"  {family:12s} {lang}  <- {model_name}")
            src = df[lang].astype(str).tolist()
            preds[family][lang] = translate(model_name, src)

    # ---- metrics ------------------------------------------------------------
    def corpus_scores(hyps):
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        return bleu, chrf

    results = {f: {} for f in FAMILIES}
    for family in FAMILIES:
        for lang in LANGS:
            bleu, chrf = corpus_scores(preds[family][lang])
            results[family][lang] = {"bleu": round(bleu, 4),
                                     "chrf": round(chrf, 4)}
            print(f"  {family:12s} {lang}: BLEU={bleu:6.2f}  chrF={chrf:6.2f}")

    # ---- macro-average + delta ---------------------------------------------
    summary = {}
    for family in FAMILIES:
        b = np.mean([results[family][l]["bleu"] for l in LANGS])
        c = np.mean([results[family][l]["chrf"] for l in LANGS])
        summary[family] = {"bleu": round(float(b), 4), "chrf": round(float(c), 4)}

    delta = {
        "bleu": round(summary["multilingual"]["bleu"] - summary["bilingual"]["bleu"], 4),
        "chrf": round(summary["multilingual"]["chrf"] - summary["bilingual"]["chrf"], 4),
    }

    out = {
        "data": DATA_CSV,
        "n_rows": n_rows,
        "subset": bool(N_ROWS),
        "device": DEVICE,
        "fp16": USE_FP16,
        "num_beams": NUM_BEAMS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "models": {
            "bilingual": BILINGUAL,
            "multilingual": MULTILINGUAL,
        },
        "per_language": results,      # results[family][lang] -> {bleu, chrf}
        "macro": summary,             # summary[family] -> {bleu, chrf}
        "delta_multilingual_minus_bilingual": delta,
        "delta_definition": "multilingual - bilingual (positive => multilingual better)",
    }
    with open("mt_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote mt_compare.json")

    # ---- plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(LANGS))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in [(axes[0], "bleu", "BLEU"), (axes[1], "chrf", "chrF")]:
        bil = [results["bilingual"][l][metric] for l in LANGS]
        mul = [results["multilingual"][l][metric] for l in LANGS]
        b1 = ax.bar(x - w / 2, bil, w, label="bilingual (it/de/fr-en)", color="#4c72b0")
        b2 = ax.bar(x + w / 2, mul, w, label="multilingual (mul-en)", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in LANGS])
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{title} on FLORES (n={n_rows})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        for bars in (b1, b2):
            for r in bars:
                ax.annotate(f"{r.get_height():.1f}",
                            (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=8)
    fig.suptitle("Bilingual vs multilingual -> EN MT on FLORES "
                 f"(greedy, fp16, n={n_rows})")
    fig.tight_layout()
    fig.savefig("mt_compare.png", dpi=150)
    print("wrote mt_compare.png")

    # ---- markdown report ----------------------------------------------------
    def row(family, lang):
        r = results[family][lang]
        return f"| {lang} | {r['bleu']:.2f} | {r['chrf']:.2f} |"

    verdict_dir = "matches / exceeds" if delta["bleu"] >= -0.5 else "falls short of"
    lines = []
    lines.append("# Bilingual vs multilingual -> EN MT on FLORES\n")
    lines.append(f"- Data: `{DATA_CSV}` — **{n_rows} rows** "
                 f"({'full set' if not N_ROWS else 'subset'}), gold reference = the **en** column.")
    lines.append(f"- Pipeline: MarianMT seq2seq, fp16={USE_FP16}, greedy "
                 f"(`num_beams={NUM_BEAMS}`), batch={BATCH_SIZE}, max_length={MAX_LENGTH}, "
                 f"device={DEVICE}, torch threads=2.")
    lines.append("- Bilingual: `opus-mt-it-en`, `opus-mt-de-en`, `opus-mt-fr-en`; "
                 "Multilingual: `opus-mt-mul-en` (single many→en model for it/de/fr).")
    lines.append("- Metrics: sacrebleu corpus **BLEU** and **chrF**, detokenized default settings.\n")

    lines.append("## Results (BLEU / chrF)\n")
    lines.append("| Language | Bilingual BLEU | Bilingual chrF | Multilingual BLEU | Multilingual chrF |")
    lines.append("|---|---|---|---|---|")
    for lang in LANGS:
        b = results["bilingual"][lang]
        m = results["multilingual"][lang]
        lines.append(f"| {lang} | {b['bleu']:.2f} | {b['chrf']:.2f} "
                     f"| {m['bleu']:.2f} | {m['chrf']:.2f} |")
    lines.append(f"| **macro (it,de,fr)** | **{summary['bilingual']['bleu']:.2f}** "
                 f"| **{summary['bilingual']['chrf']:.2f}** "
                 f"| **{summary['multilingual']['bleu']:.2f}** "
                 f"| **{summary['multilingual']['chrf']:.2f}** |")
    lines.append("")
    lines.append(f"**Delta (multilingual − bilingual):** "
                 f"BLEU {delta['bleu']:+.2f}, chrF {delta['chrf']:+.2f} "
                 f"(positive = multilingual better).\n")

    lines.append("## Verdict\n")
    lines.append(f"> The single multilingual `opus-mt-mul-en` model **{verdict_dir}** the three "
                 f"bilingual models on FLORES: macro BLEU "
                 f"{summary['multilingual']['bleu']:.2f} vs {summary['bilingual']['bleu']:.2f} "
                 f"({delta['bleu']:+.2f}), macro chrF "
                 f"{summary['multilingual']['chrf']:.2f} vs {summary['bilingual']['chrf']:.2f} "
                 f"({delta['chrf']:+.2f}).")
    lines.append("")
    lines.append("![Bilingual vs multilingual MT](mt_compare.png)\n")

    with open("mt_compare.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote mt_compare.md")

    # echo a compact summary to stdout
    print("\n=== SUMMARY ===")
    print(json.dumps({"per_language": results, "macro": summary, "delta": delta}, indent=2))


if __name__ == "__main__":
    main()
