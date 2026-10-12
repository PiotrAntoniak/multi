"""clean_toxi.py -- build a LEAKAGE-CORRECTED, size-managed copy of `FredZhang7/toxi-text-3M`.

Problem.  The HF dataset `FredZhang7/toxi-text-3M` has train/test leakage: exact texts appear
in BOTH the train and the test split (translation artifacts).  Measured on the raw files
(this machine, seed 0), for the final test sets the share of test texts that are also present
in that language's train pool is HUGE: en 90.1%, it ~99.9%, de ~84.7%, fr ~99.9%, es ~99.8%.
A model that trains on the raw train split can memorise the test texts -> inflated scores.

Fix (this builder).  For EVERY requested language, remove every test text from that language's
train pool, drop duplicate train texts, then cap.  The result is a leakage-free re-split:
0 overlap, verified and reported in `leakage_report.md`.

Data source (read-only):
    FredZhang7/toxi-text-3M
      train : train/multilingual-train-deduplicated.csv   (2,880,667 rows)
      test  : test/multilingual-test.csv                   (63,812 rows)
    Loaded from the local HF datasets arrow cache when present (fast, offline); otherwise
    falls back to `datasets.load_dataset(...)` (downloads on a fresh machine).

Outputs (CREATE-ONLY -- nothing existing is deleted or moved):
    <out>/<lang>_train.parquet   one per (lang, split), columns text,is_toxic,lang, gzip
    <out>/<lang>_test.parquet
    <out>/leakage_report.md      pre/post overlap per language + method + CLI

Usage:
    # size-managed committed copy (defaults; <out> defaults to data/toxi_clean)
    python clean_toxi.py --cap 20000 --test-cap 5000 --langs en,it,de,fr,es --out data/toxi_clean

    # reproduce the FULL cleaned dataset (no caps; only leakage removal + train dedup)
    python clean_toxi.py --cap 0 --test-cap 0 --langs en,it,de,fr,es --out data/toxi_clean_full

    # en-only "literal spec" behaviour (subsample EN train, do NOT remove EN test texts):
    python clean_toxi.py --en-subsample-only

Note: a cap of 0 (or negative) means "no cap" (keep everything left after cleaning).
"""
import argparse
import glob
import os
import sys
import time

DATASET = "FredZhang7/toxi-text-3M"
DEFAULT_LANGS = ["en", "it", "de", "fr", "es"]
FILES = {"train": "train/multilingual-train-deduplicated.csv",
         "test": "test/multilingual-test.csv"}
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
    """Load a split from the local HF `datasets` arrow cache. Returns DataFrame or None."""
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
        log("loaded %s from arrow cache: %d rows (%d shard files)" % (split, len(df), len(files)))
        return df[COLS]
    return None


def _load_via_datasets(split):
    """Fallback: download/convert via the `datasets` library."""
    from datasets import load_dataset
    log("loading %s via datasets.load_dataset (may download) ..." % split)
    ds = load_dataset(DATASET, data_files={split: FILES[split]})
    return ds[split].to_pandas()[COLS]


def load_split(split):
    df = _load_from_arrow_cache(split)
    if df is None:
        df = _load_via_datasets(split)
    df = df[COLS].copy()
    df["text"] = df["text"].astype(str)
    df["is_toxic"] = df["is_toxic"].astype(int)
    df["lang"] = df["lang"].astype(str)
    return df


# ------------------------------------------------------------------ capping
def cap(df, n, seed):
    """Return df capped to n rows (seed). n<=0 means no cap."""
    if n and n > 0 and len(df) > n:
        return df.sample(n=int(n), random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ build
def build(args):
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    out = args.out
    os.makedirs(out, exist_ok=True)

    train_all = load_split("train")
    test_all = load_split("test")

    reports = {}
    written = []
    for L in langs:
        tr = train_all[train_all["lang"] == L].copy()
        te = test_all[test_all["lang"] == L].copy()
        pool_before_rows = len(tr)
        pool_before_unique = int(tr["text"].nunique())

        # 1) finalise the test set (cap), then treat it as the leak source
        te = cap(te, args.test_cap, args.seed)[COLS]
        test_texts = set(te["text"].tolist())
        test_n = len(te)

        # 2) measure leak BEFORE correction (unique test texts present in the train pool)
        train_set_before = set(tr["text"].tolist())
        leaked_before = len(test_texts & train_set_before)
        leaked_pct_before = (100.0 * leaked_before / test_n) if test_n else 0.0

        # 3) correct: remove test texts from the train pool
        if args.en_subsample_only and L == "en":
            removed_leak = 0
        else:
            before = len(tr)
            tr = tr[~tr["text"].isin(test_texts)]
            removed_leak = before - len(tr)

        # 4) drop duplicate train texts
        before = len(tr)
        tr = tr.drop_duplicates(subset=["text"], keep="first")
        dropped_dups = before - len(tr)

        # 5) cap train
        tr = cap(tr, args.cap, args.seed)[COLS]

        # 6) verify post-condition
        train_set_after = set(tr["text"].tolist())
        leaked_after = len(test_texts & train_set_after)
        assert leaked_after == 0, "leak persists for %s: %d" % (L, leaked_after)

        # 7) write parquet (gzip)
        tr_path = os.path.join(out, "%s_train.parquet" % L)
        te_path = os.path.join(out, "%s_test.parquet" % L)
        tr.to_parquet(tr_path, engine="pyarrow", compression="gzip", index=False)
        te.to_parquet(te_path, engine="pyarrow", compression="gzip", index=False)
        written += [tr_path, te_path]

        reports[L] = dict(
            test_n=test_n,
            pool_before_rows=pool_before_rows,
            pool_before_unique=pool_before_unique,
            leaked_before=leaked_before,
            leaked_pct_before=leaked_pct_before,
            removed_leak=removed_leak,
            dropped_dups=dropped_dups,
            train_after=len(tr),
            leaked_after=leaked_after,
            train_toxic=int(tr["is_toxic"].sum()),
            test_toxic=int(te["is_toxic"].sum()),
        )
        log("%s: test=%d  leaked_before=%d (%.1f%%)  removed_leak=%d  dropped_dups=%d  "
            "train_after=%d  leaked_after=%d" %
            (L, test_n, leaked_before, leaked_pct_before, removed_leak, dropped_dups,
             len(tr), leaked_after))

    total = sum(os.path.getsize(p) for p in written)
    report_path = os.path.join(out, "leakage_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(args, langs, reports, total, written))
    log("wrote %s (%d files, %.2f MiB total)" % (report_path, len(written), total / 1024.0 / 1024.0))
    return reports, total, written


# ------------------------------------------------------------------ report
def render_report(args, langs, reports, total_bytes, written):
    cli = ("python clean_toxi.py --cap %d --test-cap %d --langs %s --out %s --seed %d%s"
           % (args.cap, args.test_cap, ",".join(langs), args.out, args.seed,
              " --en-subsample-only" if args.en_subsample_only else ""))
    L = []
    L.append("# toxi-text-3M -- leakage-corrected copy\n")
    L.append("Source: `%s` (read-only). "
             "train `%s` = 2,880,667 rows; test `%s` = 63,812 rows.\n"
             % (DATASET, FILES["train"], FILES["test"]))
    L.append("**Method.** For every language: (1) finalise the test set (cap `--test-cap`); "
             "(2) remove every test text from that language's train pool; "
             "(3) drop duplicate train texts; (4) cap train to `--cap` (seed `--seed`); "
             "(5) verify 0 remaining overlap. Test texts are treated as the leak source, so the "
             "post-condition is exactly 0 by construction and asserted.\n")
    L.append("> EN is **also** leaked (90.1% of its 525 test texts sit in the EN train pool), so "
             "the same removal is applied to EN by default. Pass `--en-subsample-only` to instead "
             "reproduce the literal 'subsample EN train only' behaviour (leaves EN leakage intact).\n")
    L.append("## Overlap per language (test texts found in the train pool)\n")
    L.append("| lang | test rows | train pool (rows / unique) | leaked test texts BEFORE | % BEFORE | "
             "removed from train | dup train dropped | train rows AFTER | leaked AFTER |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lg in langs:
        r = reports[lg]
        L.append("| %s | %d | %d / %d | %d | %.1f%% | %d | %d | %d | %d |"
                 % (lg, r["test_n"], r["pool_before_rows"], r["pool_before_unique"],
                    r["leaked_before"], r["leaked_pct_before"], r["removed_leak"],
                    r["dropped_dups"], r["train_after"], r["leaked_after"]))
    L.append("")
    L.append("**After correction every language has 0 leaked test texts** (asserted at build time).\n")
    L.append("## Caps and size\n")
    L.append("- train cap `--cap` = %d per language (0 = no cap); test cap `--test-cap` = %d "
             "per language (0 = no cap); seed = %d.  " % (args.cap, args.test_cap, args.seed))
    L.append("A test language with fewer rows than the cap keeps all of them "
             "(e.g. de test, en test).\n")
    L.append("- total `%s/` size on disk = **%.2f MiB** (%.2f MB decimal; %.0f bytes total) "
             "across %d parquet files (gzip).\n"
             % (args.out, total_bytes / 1024.0 / 1024.0, total_bytes / 1e6, total_bytes, len(written)))
    L.append("## Files\n")
    for p in written:
        L.append("- `%s` (%.1f KiB)" % (os.path.relpath(p, os.getcwd()).replace("\\", "/"),
                                        os.path.getsize(p) / 1024.0))
    L.append("")
    L.append("## Full-dataset reproduction\n")
    L.append("The size-managed copy above is a cap. To rebuild the FULL cleaned dataset "
             "(leakage removal + train dedup, no caps):\n")
    L.append("```\npython clean_toxi.py --cap 0 --test-cap 0 --langs %s --out data/toxi_clean_full\n```\n"
             % ",".join(langs))
    L.append("## Exact CLI used for this copy\n")
    L.append("```\n%s\n```\n" % cli)
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a leakage-corrected toxi-text-3M copy.")
    ap.add_argument("--cap", type=int, default=20000,
                    help="train rows per language after cleaning (0 = no cap)")
    ap.add_argument("--test-cap", type=int, default=5000,
                    help="test rows per language (0 = no cap)")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS),
                    help="comma-separated languages")
    ap.add_argument("--out", default=os.path.join("data", "toxi_clean"),
                    help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--en-subsample-only", action="store_true",
                    help="for EN only: subsample train, do NOT remove EN test texts (literal spec)")
    args = ap.parse_args(argv)

    reports, total, written = build(args)

    # stdout summary table
    print("\n%-4s %10s %16s %18s %8s %10s %12s" %
          ("lang", "test", "leaked_before", "pct_before", "removed", "dup_dropped", "train_after"))
    for lg, r in reports.items():
        print("%-4s %10d %16d %17.1f%% %8d %10d %12d" %
              (lg, r["test_n"], r["leaked_before"], r["leaked_pct_before"],
               r["removed_leak"], r["dropped_dups"], r["train_after"]))
    print("\ntotal size: %.2f MiB (%.2f MB, %d bytes)" % (total / 1024.0 / 1024.0, total / 1e6, total))


if __name__ == "__main__":
    main()
