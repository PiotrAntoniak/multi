"""Table 1 (aggregated) -- devtest retrieval protocol on URL-level FLORES rows.

Same protocol as table1.py, but applied to the AGGREGATED data (flores_agg.csv,
562 URL-level rows: one row per article with merged en/it/de/fr text).  For a
directed pair a->b with English on one side, the language-shift vector D is the
mean over DEV rows of (emb_a - emb_b); meaning cancels in that difference, leaving
the cross-lingual shift.  Retrieval is evaluated on the DEVTEST rows only: the
query is emb_a, the candidate keys are emb_b ("raw"); adding D to every key ("+D")
should move each key onto the query side.  Rows are L2-normalized so dot products
are cosine similarities, and the rank of query i is 1 + #{keys strictly closer
than its exact translation key i}.  This script reports, per pooling mode, the
min-max ranges over the THREE foreign-to-English directions (it->en, de->en,
fr->en) of top-1, MRR and (median) rank, before and after adding D, plus the
per-direction detail.  All other directions are out of scope.

Split resolution: the dev / devtest split is taken from the agg row's own `split`
column when present.  flores_agg.csv as produced does NOT carry a split column,
so the split falls back to the per-URL split of the source file
flores200_en_it_de_fr.csv (every source URL belongs to exactly one split).  An agg
URL with no / ambiguous source split is excluded and reported.

Usage: python table1_agg.py
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "flores_agg.csv"
SRC = ROOT / "flores200_en_it_de_fr.csv"
EMB_ROOT = ROOT / "embeddings_agg"
LANGS = ["en", "it", "de", "fr"]
MODES = ["mean", "bos"]
EPS = 1e-12


def l2norm(X):
    """Row-wise L2 normalization (all-zero rows stay zero)."""
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def ranks(sim):
    """1 + number of keys strictly closer to query i than its exact key i."""
    return 1 + (sim > np.diag(sim)[:, None]).sum(axis=1)


def direction_stats(embs, a, b, dev, test):
    """Top-1 %, MRR, mean rank and median rank for a->b, raw and after adding D."""
    D = (embs[a][dev] - embs[b][dev]).mean(axis=0)
    E, T, F = l2norm(embs[a][test]), l2norm(embs[b][test]), l2norm(embs[b][test] + D)
    out = []
    for r in (ranks(E @ T.T), ranks(E @ F.T)):
        out.append((100.0 * (r <= 1).mean(), float(np.mean(1.0 / r)),
                    float(r.mean()), float(np.median(r))))
    return out  # [raw(top1,mrr,meanrank,medrank), plusd(...)]


def load_split(rows, fields):
    """Return an array of split labels for the agg rows.

    Prefer the agg row's own `split` column; otherwise map each agg URL to the
    split of its (unique) source row.  Unmapped/ambiguous rows get "unknown".
    """
    if "split" in fields:
        return np.array([r["split"] for r in rows]), "flores_agg.csv:split column"
    by_url = {}
    with SRC.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            by_url.setdefault(r["URL"], set()).add(r["split"])
    split, unknown = [], []
    for i, r in enumerate(rows):
        s = by_url.get(r["URL"])
        if s and len(s) == 1:
            split.append(next(iter(s)))
        else:
            split.append("unknown")
            unknown.append(i)
    return np.array(split), f"{SRC.name}:per-URL ({len(unknown)} unknown)", unknown


def main():
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames
        rows = list(rdr)
    split, src, unknown = load_split(rows, fields)
    if unknown:
        print(f"WARNING: {len(unknown)} agg URL(s) without a unique source split "
              f"(indices {unknown}) -> excluded from dev/devtest.")
    dev, test = split == "dev", split == "devtest"
    dirs = [(a, "en") for a in LANGS if a != "en"]
    print(f"Table 1-agg - devtest protocol | split source: {src} | "
          f"dev={int(dev.sum())} devtest N={int(test.sum())} "
          f"| 3 foreign-to-English directions | chance rank {(test.sum() + 1) / 2:.1f}")

    result, detail, ran = {}, {}, []
    for mode in MODES:
        missing = [l for l in LANGS if not (EMB_ROOT / mode / f"emb_{l}.npy").exists()]
        if missing:
            print(f"SKIP mode '{mode}': missing {EMB_ROOT / mode} emb_*.npy for {missing}")
            continue
        embs = {l: np.load(EMB_ROOT / mode / f"emb_{l}.npy") for l in LANGS}
        ran.append(mode)
        dir_rows = []
        for a, b in dirs:
            st = direction_stats(embs, a, b, dev, test)
            dir_rows.append(st)
            detail[(mode, a, b)] = st

        def span(k, m):
            return min(s[k][m] for s in dir_rows), max(s[k][m] for s in dir_rows)

        result[mode] = {"top1raw": span(0, 0), "top1+D": span(1, 0), "mrrraw": span(0, 1),
                        "mrr+D": span(1, 1), "rankraw": span(0, 2), "rank+D": span(1, 2),
                        "medraw": span(0, 3), "med+D": span(1, 3)}

    print("| mode | top-1 raw | top-1 +D | MRR raw | MRR +D | median rank +D |")
    print("|---|---|---|---|---|---|")
    for mode in ran:
        c = result[mode]
        print(f"| {mode} | {c['top1raw'][0]:.1f}-{c['top1raw'][1]:.1f}% "
              f"| {c['top1+D'][0]:.1f}-{c['top1+D'][1]:.1f}% "
              f"| {c['mrrraw'][0]:.3f}-{c['mrrraw'][1]:.3f} "
              f"| {c['mrr+D'][0]:.3f}-{c['mrr+D'][1]:.3f} "
              f"| {c['med+D'][0]:.0f}-{c['med+D'][1]:.0f} |")

    print("\nPer-direction detail:")
    for mode in ran:
        for a, b in dirs:
            (t1r, mrrr, mr, mdr), (t1d, mrrd, mrd, mdd) = detail[(mode, a, b)]
            print(f"  [{mode}] {a}->{b}  top-1 {t1r:.1f}% (raw) / {t1d:.1f}% (+D)  "
                  f"MRR {mrrr:.3f} (raw) / {mrrd:.3f} (+D)  "
                  f"mean rank {mr:.1f} (raw) / {mrd:.1f} (+D)  "
                  f"median rank {mdr:.0f} (raw) / {mdd:.0f} (+D)")


if __name__ == "__main__":
    main()
