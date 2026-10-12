"""clean_toxi.py -- build a LEAKAGE-CORRECTED, size-managed copy of `FredZhang7/toxi-text-3M`.

Problem.  The HF dataset `FredZhang7/toxi-text-3M` has train/test leakage: exact texts appear
in MORE THAN ONE split (translation artifacts).  Measured against the shipped train pool the
share of test texts also present in train is huge (en 90.1%, it ~99.9%, de ~84.7%, fr ~99.9%,
es ~99.8%).  The original per-split counts are also tiny for some languages (en: 525 test rows
vs a 2.5M train pool), so a plain 80/10/10 re-split of the shipped splits is not possible.

Fix (this builder).  For EVERY requested language:
    1. POOL all available rows from ALL shipped splits (train + validation + test).
    2. Deduplicate GLOBALLY per language (drop duplicate `text`).  Because a text is unique
       per language, no text can appear in two splits -> leakage is removed by construction.
    3. Re-split the deduped pool 80 / 10 / 10 (train / validation / test) with a fixed seed,
       with ceil() on the 10% parts so validation and test are each >= 10%.
    4. Optionally cap the TOTAL per language (`--total-cap`) to keep the committed copy small.
       The cap is applied to the pool BEFORE splitting, so the 80/10/10 proportions still hold.
Verified by reloading the parquet: 0 cross-split text overlap, split fractions ~0.8/0.1/0.1.

Data source (read-only):
    FredZhang7/toxi-text-3M
      train      : train/multilingual-train-deduplicated.csv  (2,880,667 rows)
      validation : validation/multilingual-validation(new).csv (29,641 rows)
      test       : test/multilingual-test.csv                  (63,812 rows)
    train/test are read from the local HF `datasets` arrow cache when present (fast, offline);
    every split falls back to `huggingface_hub.hf_hub_download` + pandas.  A split that cannot
    be obtained is skipped with a warning (the pool is then train+test, etc.).

Outputs (CREATE-ONLY except the stale-file cleanup noted below):
    <out>/<lang>_train.parquet
    <out>/<lang>_validation.parquet
    <out>/<lang>_test.parquet
    <out>/leakage_report.md
    All parquet: columns `text,is_toxic,lang`, gzip.  Stale `<lang>_*.parquet` files that do NOT
    match the {train,validation,test} triple are removed (this is the only deletion performed).

Usage:
    python clean_toxi.py --langs en,it,de,fr,es --out data/toxi_clean --seed 0 --total-cap 30000
    python clean_toxi.py --langs en,it,de,fr,es --out data/toxi_clean --seed 0 --total-cap 0   # no cap
    python clean_toxi.py --no-validation                                                        # pool train+test only
"""
import argparse
import glob
import math
import os
import time

DATASET = "FredZhang7/toxi-text-3M"
DEFAULT_LANGS = ["en", "it", "de", "fr", "es"]
SPLITS = ["train", "validation", "test"]
FILES = {
    "train": "train/multilingual-train-deduplicated.csv",
    "validation": "validation/multilingual-validation(new).csv",
    "test": "test/multilingual-test.csv",
}
COLS = ["text", "is_toxic", "lang"]

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ------------------------------------------------------------------ loading
def _hf_datasets_cache():
    c = os.environ.get("HF_DATASETS_CACHE")
    if c:
        return c
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")


def _load_from_arrow_cache(split):
    """Load train/test from the local HF `datasets` arrow cache. Returns DataFrame or None."""
    if split not in ("train", "test"):
        return None
    cache = _hf_datasets_cache()
    if not os.path.isdir(cache):
        return None
    roots = glob.glob(os.path.join(cache, "*toxi-text-3_m*"))
    for root in sorted(roots):
        if split == "test":
            pat = os.path.join(root, "*", "0.0.0", "*", "toxi-text-3_m-test.arrow")
        else:
            # sharded train only; excludes the stray non-sharded `toxi-text-3_m-train.arrow`
            pat = os.path.join(root, "*", "0.0.0", "*", "toxi-text-3_m-train-*-of-*.arrow")
        files = sorted(glob.glob(pat))
        if not files:
            continue
        from datasets import Dataset, concatenate_datasets
        dss = [Dataset.from_file(f) for f in files]
        ds = concatenate_datasets(dss) if len(dss) > 1 else dss[0]
        df = ds.to_pandas()
        log("loaded %-10s from arrow cache: %d rows (%d files)" % (split, len(df), len(files)))
        return df[COLS]
    return None


def _load_via_hub(split):
    """Download a split CSV via huggingface_hub and read it with pandas."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    log("downloading %-10s via hf_hub_download ..." % split)
    p = hf_hub_download(DATASET, FILES[split], repo_type="dataset")
    df = pd.read_csv(p)
    log("loaded %-10s from CSV: %d rows" % (split, len(df)))
    return df


def load_split(split):
    df = _load_from_arrow_cache(split)
    if df is None:
        df = _load_via_hub(split)
    df = df[COLS].copy()
    df["text"] = df["text"].astype(str)
    df["is_toxic"] = df["is_toxic"].astype(int)
    df["lang"] = df["lang"].astype(str)
    return df


# ------------------------------------------------------------------ splitting
def split_80_10_10(pool, seed):
    """Deterministic 80/10/10 split. val/test sizes use ceil(0.1*N) so both are >= 10%."""
    import numpy as np
    n = len(pool)
    n_test = max(1, int(math.ceil(0.10 * n)))
    n_val = max(1, int(math.ceil(0.10 * n)))
    n_train = n - n_val - n_test
    pos = np.random.default_rng(seed).permutation(n)
    tr = pool.iloc[pos[n_val + n_test:]].reset_index(drop=True)
    va = pool.iloc[pos[:n_val]].reset_index(drop=True)
    te = pool.iloc[pos[n_val:n_val + n_test]].reset_index(drop=True)
    return tr, va, te


# ------------------------------------------------------------------ build
def build(args):
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    out = args.out
    os.makedirs(out, exist_ok=True)

    split_names = [s for s in SPLITS if not (s == "validation" and args.no_validation)]
    loaded = {}
    for s in split_names:
        try:
            loaded[s] = load_split(s)
        except Exception as e:  # noqa: BLE001
            log("WARN: split %s unavailable (%r); pooling without it" % (s, e))
            loaded[s] = None
    if loaded.get("train") is None and loaded.get("test") is None:
        raise SystemExit("need at least one of train/test to pool")

    reports = {}
    written = []
    for L in langs:
        frames = [loaded[s][loaded[s]["lang"] == L] for s in split_names
                  if loaded[s] is not None]
        pool = (frames[0] if len(frames) == 1 else
                __import__("pandas").concat(frames, ignore_index=True))[COLS]
        pooled_n = len(pool)

        # 1) global dedup per language (unique text -> no cross-split leakage)
        pool = pool.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
        dedup_n = len(pool)
        dedup_drop = pooled_n - dedup_n

        # 2) optional TOTAL cap per language (before splitting; keeps 80/10/10)
        capped = False
        if args.total_cap and args.total_cap > 0 and len(pool) > args.total_cap:
            pool = pool.sample(n=int(args.total_cap), random_state=args.seed).reset_index(drop=True)
            capped = True
        pool_n = len(pool)

        # 3) re-split 80/10/10
        tr, va, te = split_80_10_10(pool, args.seed)

        # 4) write parquet
        paths = {}
        for name, df in (("train", tr), ("validation", va), ("test", te)):
            p = os.path.join(out, "%s_%s.parquet" % (L, name))
            df.to_parquet(p, engine="pyarrow", compression="gzip", index=False)
            written.append(p)
            paths[name] = p

        # 5) remove stale <lang>_*.parquet that are not part of the new triple
        expected = {"%s_train.parquet" % L, "%s_validation.parquet" % L, "%s_test.parquet" % L}
        for p in glob.glob(os.path.join(out, "%s_*.parquet" % L)):
            if os.path.basename(p) not in expected:
                os.remove(p)
                log("removed stale file %s" % p)

        # 6) verify 0 cross-split overlap
        st = {"train": set(tr["text"]), "validation": set(va["text"]), "test": set(te["text"])}
        overlap = (len(st["train"] & st["validation"]) + len(st["train"] & st["test"]) +
                   len(st["validation"] & st["test"]))
        assert overlap == 0, "cross-split overlap for %s: %d" % (L, overlap)

        reports[L] = dict(
            pooled_n=pooled_n, dedup_n=dedup_n, dedup_drop=dedup_drop, capped=capped,
            pool_n=pool_n,
            train=len(tr), val=len(va), test=len(te),
            frac_train=len(tr) / pool_n, frac_val=len(va) / pool_n, frac_test=len(te) / pool_n,
            pr_train=float(tr["is_toxic"].mean()),
            pr_val=float(va["is_toxic"].mean()),
            pr_test=float(te["is_toxic"].mean()),
            overlap=overlap,
        )
        log("%s: pooled=%d dedup=%d(-%d)%s  pool=%d  train=%d val=%d test=%d  "
            "frac=%.3f/%.3f/%.3f  overlap=%d"
            % (L, pooled_n, dedup_n, dedup_drop, " capped" if capped else "", pool_n,
               len(tr), len(va), len(te), len(tr) / pool_n, len(va) / pool_n, len(te) / pool_n,
               overlap))

    total = sum(os.path.getsize(p) for p in written)
    report_path = os.path.join(out, "leakage_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(args, langs, reports, total, written, split_names))
    log("wrote %s (%d files, %.2f MiB total)" % (report_path, len(written), total / 1024.0 / 1024.0))
    return reports, total


# ------------------------------------------------------------------ report
def render_report(args, langs, reports, total_bytes, written, split_names):
    cli = ("python clean_toxi.py --langs %s --out %s --seed %d --total-cap %d%s"
           % (",".join(langs), args.out, args.seed, args.total_cap,
              " --no-validation" if args.no_validation else ""))
    L = []
    L.append("# toxi-text-3M -- leakage-corrected 80/10/10 copy\n")
    L.append("Source: `%s` (read-only). Shipped splits pooled: `%s`.\n"
             % (DATASET, ", ".join(split_names)))
    L.append("**Method.** For each language: (1) pool ALL shipped splits "
             "(train + validation + test); (2) deduplicate GLOBALLY per language on `text`; "
             "(3) optionally cap the TOTAL per language (`--total-cap`, before splitting); "
             "(4) re-split 80/10/10 with seed `--seed` (ceil on the 10% parts so val and test are "
             "each >= 10%). A text is unique per language, so no text can appear in two splits -> "
             "0 cross-split leakage by construction (asserted).\n")
    L.append("## Per-language counts\n")
    L.append("| lang | pooled N | dedup drop | pool N (post-cap) | train | val | test | "
             "train/val/test frac | pos-rate tr/val/te | cross-split overlap |")
    L.append("|---|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|")
    for lg in langs:
        r = reports[lg]
        L.append("| %s | %d | %d | %d%s | %d | %d | %d | %.3f / %.3f / %.3f | "
                 "%.3f / %.3f / %.3f | %d |"
                 % (lg, r["pooled_n"], r["dedup_drop"], r["pool_n"],
                    " (capped)" if r["capped"] else "",
                    r["train"], r["val"], r["test"],
                    r["frac_train"], r["frac_val"], r["frac_test"],
                    r["pr_train"], r["pr_val"], r["pr_test"], r["overlap"]))
    L.append("")
    L.append("**Cross-split text overlap is 0 for every language** (asserted at build time; "
             "val and test are each >= 10% of the pooled per-language data).\n")
    L.append("## Caps and size\n")
    L.append("- `--total-cap` = %d per language total (0 = no cap); seed = %d.  "
             "A language whose deduped pool is below the cap keeps all of it.\n"
             % (args.total_cap, args.seed))
    L.append("- total size on disk = **%.2f MiB** (%.2f MB decimal; %d bytes) across %d parquet "
             "files (gzip) + this report.\n"
             % (total_bytes / 1024.0 / 1024.0, total_bytes / 1e6, total_bytes, len(written)))
    L.append("## Files\n")
    for p in written:
        L.append("- `%s` (%.1f KiB)" % (os.path.relpath(p, os.getcwd()).replace("\\", "/"),
                                        os.path.getsize(p) / 1024.0))
    L.append("")
    L.append("## Exact CLI used\n")
    L.append("```\n%s\n```\n" % cli)
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a leakage-corrected, 80/10/10 toxi copy.")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS),
                    help="comma-separated languages")
    ap.add_argument("--out", default=os.path.join("data", "toxi_clean"),
                    help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-cap", type=int, default=30000,
                    help="cap TOTAL rows per language before splitting (0 = no cap)")
    ap.add_argument("--no-validation", action="store_true",
                    help="do not pool the validation file (pool train+test only)")
    args = ap.parse_args(argv)

    reports, total = build(args)

    print("\n%-4s %12s %10s %12s %9s %9s %9s %10s %20s" %
          ("lang", "pooled", "dedup_drop", "pool", "train", "val", "test",
           "overlap", "pos tr/val/te"))
    for lg, r in reports.items():
        print("%-4s %12d %10d %12d %9d %9d %9d %10d  %.3f/%.3f/%.3f" %
              (lg, r["pooled_n"], r["dedup_drop"], r["pool_n"], r["train"], r["val"], r["test"],
               r["overlap"], r["pr_train"], r["pr_val"], r["pr_test"]))
    print("\ntotal size: %.2f MiB (%.2f MB, %d bytes)" % (total / 1024.0 / 1024.0, total / 1e6, total))


if __name__ == "__main__":
    main()
