"""Paired mean/variance panels.

Each row is one chart type: left = mean version, right = variance version.
Rows a-f, then Pearson (g-i), then histogram (j) + statistics.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pc1_dimension_plots import _add_density, _add_pearson, COLORS


def read_rows(path, topk=16):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[:topk] if topk and topk < len(rows) else rows


def languages(rows):
    cols = list(rows[0].keys())
    langs = [c[:-len("_mean_before")] for c in cols
             if c.endswith("_mean_before") and c != "all_mean_before"]
    order = {c: i for i, c in enumerate(cols)}
    return sorted(langs, key=lambda x: order.get(x + "_mean_before", 999))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cross-json", required=True)
    ap.add_argument("--byk-mean", required=True)
    ap.add_argument("--byk-var", required=True)
    ap.add_argument("--loadings", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = read_rows(args.csv, 16)
    langs = languages(rows)
    n = len(rows)
    dim = [int(r["dim"]) for r in rows]
    aw = np.abs([float(r["pc1_w"]) for r in rows])

    def arr(key):
        return np.array([float(r[key]) for r in rows])

    mb = {lg: arr(lg + "_mean_before") for lg in langs}
    ma = {lg: arr(lg + "_mean_after3") for lg in langs}
    vb = {lg: arr(lg + "_var_before") for lg in langs}
    va = {lg: arr(lg + "_var_after3") for lg in langs}
    amb, ama = arr("all_mean_before"), arr("all_mean_after3")
    avb, ava = arr("all_var_before"), arr("all_var_after3")

    cross = json.load(open(args.cross_json, encoding="utf-8"))
    byk_m = list(csv.DictReader(open(args.byk_mean, encoding="utf-8")))
    byk_v = list(csv.DictReader(open(args.byk_var, encoding="utf-8")))
    loadings = np.load(args.loadings)

    fig = plt.figure(figsize=(14, 28))
    gs = fig.add_gridspec(8, 2, hspace=0.45, wspace=0.25)

    def diag(a, x, y, title, c=None, s=42):
        sc = a.scatter(x, y, c=(aw if c is None else c), cmap="viridis", s=s)
        m = max(x.max(), y.max())
        a.plot([0, m], [0, m], "k--", lw=1)
        a.set_title(title, fontsize=9)
        a.grid(alpha=.25)
        return sc

    # row 0: a (variance only; the global mean is invariant)
    a = fig.add_subplot(gs[0, :])
    sc = diag(a, avb, ava, "(a) all-rows variance, top-%d" % n)
    a.set_xlabel("before"); a.set_ylabel("after PC1-3")
    fig.colorbar(sc, ax=a, label="|PC1 loading|")

    # row 1: b
    for col, (data_b, data_a, title) in enumerate([
            (mb, ma, "(b mean) per-language means"),
            (vb, va, "(b var) per-language variance")]):
        a = fig.add_subplot(gs[1, col])
        for lg in langs:
            a.scatter(data_b[lg], data_a[lg], s=32, alpha=.75,
                      color=COLORS.get(lg), label=lg)
        lo = min(min(data_b[lg].min(), data_a[lg].min()) for lg in langs)
        hi = max(max(data_b[lg].max(), data_a[lg].max()) for lg in langs)
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_title(title, fontsize=9)
        a.legend(fontsize=7)
        a.grid(alpha=.25)

    # row 2: c (variance only)
    a = fig.add_subplot(gs[2, :])
    x = np.arange(n)
    a.bar(x - .2, avb, .4, label="before", color="#4c72b0")
    a.bar(x + .2, ava, .4, label="after PC1-3", color="#c44e52")
    a.set_xticks(x); a.set_xticklabels(dim, rotation=90, fontsize=6)
    a.set_title("(c) all-rows variance, top-%d" % n, fontsize=9)
    a.legend(fontsize=7); a.grid(axis="y", alpha=.25)

    # row 3: d
    a = fig.add_subplot(gs[3, 0])
    for lg in langs:
        a.plot(np.arange(1, n + 1), ma[lg] - mb[lg], "-o",
               color=COLORS.get(lg), label=lg)
    a.axhline(0, color="gray", ls=":", lw=1)
    a.set_title("(d mean) mean change per dimension", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.25)
    a = fig.add_subplot(gs[3, 1])
    for lg in langs:
        a.plot(np.arange(1, n + 1), va[lg] / vb[lg], "-o",
               color=COLORS.get(lg), label=lg)
    a.plot(np.arange(1, n + 1), ava / avb, "k-", lw=2, label="all")
    a.axhline(1, color="gray", ls=":", lw=1)
    a.set_title("(d var) variance retention", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.25)

    # row 4: e
    a = fig.add_subplot(gs[4, 0])
    Dm = np.stack([ma[lg] - mb[lg] for lg in langs])
    im = a.imshow(Dm, aspect="auto", cmap="RdBu_r",
                  vmin=-max(abs(Dm).max(), 1e-6), vmax=max(abs(Dm).max(), 1e-6))
    a.set_yticks(range(len(langs))); a.set_yticklabels(langs)
    a.set_xticks(range(n)); a.set_xticklabels(dim, rotation=90, fontsize=6)
    a.set_title("(e mean) mean change heatmap", fontsize=9)
    fig.colorbar(im, ax=a, label="after - before")
    a = fig.add_subplot(gs[4, 1])
    Dv = np.stack([va[lg] / vb[lg] for lg in langs])
    im = a.imshow(Dv, aspect="auto", cmap="magma", vmin=.5, vmax=1)
    a.set_yticks(range(len(langs))); a.set_yticklabels(langs)
    a.set_xticks(range(n)); a.set_xticklabels(dim, rotation=90, fontsize=6)
    a.set_title("(e var) variance retention", fontsize=9)
    fig.colorbar(im, ax=a, label="after / before")

    # row 5: f
    ks_m = [int(r["topk"]) for r in byk_m]
    a = fig.add_subplot(gs[5, 0])
    for lg in langs:
        a.plot(ks_m, [float(r[lg + "_before"]) for r in byk_m], "-o",
               color=COLORS.get(lg), label=lg)
        a.plot(ks_m, [float(r[lg + "_after3"]) for r in byk_m], "--o",
               color=COLORS.get(lg), alpha=.7)
    a.plot(ks_m, [float(r["all_before"]) for r in byk_m], "-ko", lw=2, label="all")
    a.plot(ks_m, [float(r["all_after3"]) for r in byk_m], "--ko", lw=2, alpha=.7)
    a.set_xticks(ks_m); a.set_xticklabels([str(k) for k in ks_m])
    a.set_title("(f mean) mean value vs top-k", fontsize=9)
    a.legend(fontsize=7, ncol=2); a.grid(alpha=.25)
    ks_v = [int(r["topk"]) for r in byk_v]
    a = fig.add_subplot(gs[5, 1])
    for lg in langs:
        a.plot(ks_v, [float(r[lg + "_before"]) for r in byk_v], "-o",
               color=COLORS.get(lg), label=lg)
        a.plot(ks_v, [float(r[lg + "_after3"]) for r in byk_v], "--o",
               color=COLORS.get(lg), alpha=.7)
    a.plot(ks_v, [float(r["all_before"]) for r in byk_v], "-ko", lw=2, label="all")
    a.plot(ks_v, [float(r["all_after3"]) for r in byk_v], "--ko", lw=2, alpha=.7)
    a.set_xticks(ks_v); a.set_xticklabels([str(k) for k in ks_v])
    a.set_title("(f var) mean variance vs top-k", fontsize=9)
    a.legend(fontsize=7, ncol=2); a.grid(alpha=.25)

    # row 6: Pearson
    cl = cross["langs"]
    order = [cl.index(lg) for lg in langs]
    specs = [(16, "(g)"), (64, "(h)"), (128, "(i)")]
    subgs = gs[6, :].subgridspec(1, 4, width_ratios=[1, 1, 1, .05], wspace=.35)
    vmax = .10 if args.dataset == "flores" else .02
    ims = []
    for kk, (k, lab) in enumerate(specs):
        B = np.array(cross["pearson_top%d_before" % k], dtype=float)[np.ix_(order, order)]
        Af = np.array(cross["pearson_top%d_after" % k], dtype=float)[np.ix_(order, order)]
        a = fig.add_subplot(subgs[0, kk])
        ims.append(_add_pearson(a, B, Af, langs, vmax=vmax))
        off = ~np.eye(len(langs), dtype=bool)
        a.set_title("%s Pearson r, top-%d (%.3f -> %.3f)" %
                    (lab, k, B[off].mean(), Af[off].mean()), fontsize=9)
    cax = fig.add_subplot(subgs[0, 3])
    fig.colorbar(ims[0], cax=cax, label="after - before (scale +/-%.2f)" % vmax)

    # row 7: histogram + stats
    sub2 = gs[7, :].subgridspec(1, 2, width_ratios=[2.3, 1.0], wspace=.12)
    a = fig.add_subplot(sub2[0, 0])
    _add_density(a, loadings, n)
    a.set_title("(j) PC1 loading histogram (counts)", fontsize=9)
    a2 = fig.add_subplot(sub2[0, 1])
    a2.axis("off")
    lines = ["MEAN / VAR STATS (top-%d)" % n, ""]
    for lg in langs:
        lines.append("%-3s mean %.3f -> %.3f" % (lg, mb[lg].mean(), ma[lg].mean()))
        lines.append("    var  %.3f -> %.3f" % (vb[lg].mean(), va[lg].mean()))
    lines.append("all mean %.3f -> %.3f" % (amb.mean(), ama.mean()))
    lines.append("all var  %.3f -> %.3f" % (avb.mean(), ava.mean()))
    a2.set_title("Statistics", fontsize=10, loc="left")
    a2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=8)

    fig.suptitle("%s PC1: means (left) and variances (right) before vs after PC1-3"
                 % args.dataset.upper(), fontsize=13)
    legend = [
        "Panel key:",
        "(a) all-rows variance before vs after; variance only.",
        "(b) per-language mean (left) and per-language variance (right), before vs after.",
        "(c) all-rows variance per dimension, before vs after; variance only.",
        "(d) mean change per dimension (left) and variance retention (right).",
        "(e) mean-change heatmap (left) and variance-retention heatmap (right).",
        "(f) mean value vs top-k (left) and mean variance vs top-k (right); solid = before, dashed = after.",
        "(g)-(i) Pearson r on the top-16/64/128 dims; each cell before (top) / after (bottom); shared scale.",
        "(j) PC1 loading histogram (counts).",
    ]
    fig.text(0.012, 0.006, "\n".join(legend), fontsize=7.6, va="bottom",
             ha="left", linespacing=1.35)
    fig.subplots_adjust(top=.95, bottom=.11, left=.06, right=.90)
    out = os.path.join(args.outdir, "%s_pc1_mean_var_panels_210m.png" % args.dataset)
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
