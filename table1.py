"""Table 1 -- devtest retrieval protocol, end to end (standalone: csv + numpy only).

Paragraph-length summary of the protocol:
  FLORES-200 en/it/de/fr rows are parallel across languages.  For a directed pair
  a->b, the language-shift vector D is the mean over DEV rows of (emb_a - emb_b);
  meaning cancels in that difference, leaving the cross-lingual shift.  Retrieval
  is evaluated on the DEVTEST rows only: the query is emb_a, the 1012 candidate
  keys are emb_b ("raw"); adding D to every key ("+D") should move each key onto
  the query side.  Rows are L2-normalized so dot products are cosine similarities,
  and the rank of query i is 1 + #{keys strictly closer than its exact translation
  key i}.  This script reports, per pooling mode, the min-max ranges over the 12
  ordered directions of top-1, MRR and mean rank, before and after adding D.

Usage: python table1.py
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "flores200_en_it_de_fr.csv"
EMB_ROOT = ROOT / "embeddings"
LANGS = ["en", "it", "de", "fr"]
MODES = ["mean", "lead", "bos", "eos"]
EPS = 1e-12

EXPECTED = {  # (top1 raw, top1 +D, mrr raw, mrr +D, rank raw, rank +D)
    "mean": ((27.1, 80.8), (78.8, 93.7), (0.485, 0.869), (0.836, 0.961), (2.1, 13.1), (1.2, 6.3)),
    "lead": ((7.3, 26.5), (10.5, 28.9), (0.118, 0.333), (0.144, 0.354), (181.2, 308.7), (179.4, 316.8)),
    "bos": ((2.5, 8.9), (4.4, 11.8), (0.044, 0.128), (0.075, 0.173), (208.4, 364.6), (180.5, 330.0)),
    "eos": ((1.8, 7.5), (9.1, 21.1), (0.043, 0.139), (0.154, 0.289), (155.7, 290.0), (105.7, 194.9)),
}


def l2norm(X):
    """Row-wise L2 normalization (all-zero rows stay zero)."""
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def ranks(sim):
    """1 + number of keys strictly closer to query i than its exact key i."""
    return 1 + (sim > np.diag(sim)[:, None]).sum(axis=1)


def direction_stats(embs, a, b, dev, test):
    """Top-1 %, MRR and mean rank for a->b, raw and after adding D."""
    D = (embs[a][dev] - embs[b][dev]).mean(axis=0)
    E, T, F = l2norm(embs[a][test]), l2norm(embs[b][test]), l2norm(embs[b][test] + D)
    out = []
    for r in (ranks(E @ T.T), ranks(E @ F.T)):
        out.append((100.0 * (r <= 1).mean(), float(np.mean(1.0 / r)), float(r.mean())))
    return out  # [raw(top1,mrr,rank), plusd(top1,mrr,rank)]


def main():
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        split = np.array([row["split"] for row in csv.DictReader(f)])
    dev, test = split == "dev", split == "devtest"
    dirs = [(a, b) for a in LANGS for b in LANGS if a != b]
    print(f"Table 1 - devtest protocol | dev={int(dev.sum())} devtest N={int(test.sum())} "
          f"| 12 ordered directions | chance rank {(test.sum() + 1) / 2:.1f}")

    result = {}
    for mode in MODES:
        embs = {l: np.load(EMB_ROOT / mode / f"emb_{l}.npy") for l in LANGS}
        rows = [direction_stats(embs, a, b, dev, test) for a, b in dirs]
        # rows[dir] = [before(top1,mrr,rank), after(top1,mrr,rank)]; take min/max per metric.
        def span(k, m):
            return min(s[k][m] for s in rows), max(s[k][m] for s in rows)

        cols = {"top1raw": span(0, 0), "top1+D": span(1, 0), "mrrraw": span(0, 1),
                "mrr+D": span(1, 1), "rankraw": span(0, 2), "rank+D": span(1, 2)}
        result[mode] = cols
        exp = tuple(cols[k] for k in ("top1raw", "top1+D", "mrrraw", "mrr+D", "rankraw", "rank+D"))
        for g, e in zip(exp, EXPECTED[mode]):
            assert abs(g[0] - e[0]) < 0.05 and abs(g[1] - e[1]) < 0.05, f"{mode} {g} != {e}"

    print("| mode | top-1 raw | top-1 +D | MRR raw | MRR +D | mean rank raw | mean rank +D |")
    print("|---|---|---|---|---|---|---|")
    for mode in MODES:
        c = result[mode]
        print(f"| {mode} | {c['top1raw'][0]:.1f}-{c['top1raw'][1]:.1f}% | {c['top1+D'][0]:.1f}-{c['top1+D'][1]:.1f}% "
              f"| {c['mrrraw'][0]:.3f}-{c['mrrraw'][1]:.3f} | {c['mrr+D'][0]:.3f}-{c['mrr+D'][1]:.3f} "
              f"| {c['rankraw'][0]:.1f}-{c['rankraw'][1]:.1f} | {c['rank+D'][0]:.1f}-{c['rank+D'][1]:.1f} |")
    print("\nverified: all values match the reference table")


if __name__ == "__main__":
    main()
