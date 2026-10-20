"""Plots for the leading-PC variance analysis.

Outputs
-------
<dataset>_pc1_variance_210m.png
    (a)-(f) variance panels, (g)-(i) Pearson r on the top-16/64/128 dims
    (before -> after PC1-3), (j) PC1 loading density.
<dataset>_pc1_weight_density_210m.png
    Standalone PC1 loading density.

Inputs
------
--csv        per-dimension table with dim, pc1_w, <lang>_{mean,var}_{before,after3},
             all_{mean,var}_{before,after3}
--loadings   .npy vector of all PC1 loadings
--cross-json optional: langs, pearson_top{16,64,128}_{before,after}
--byk-csv    optional: topk, <lang>_{before,after3}, all_{before,after3}

Example
-------
python pc1_dimension_plots.py --dataset flores \
  --csv flores_pc1_top16_per_language_mean_var.csv \
  --loadings flores_pc1_loadings.npy \
  --cross-json flores_pc1_cross_correlation.json \
  --byk-csv flores_pc1_mean_variance_by_topk.csv \
  --outdir pc1_210 --rows-total 8024 --rows-per-lang 2006
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {"en": "#1f77b4", "it": "#ff7f0e", "de": "#2ca02c",
          "fr": "#d62728", "es": "#9467bd", "ja": "#8c564b",
          "zh": "#e377c2"}
TH_COLORS = {16: "#c44e52", 64: "#dd8452", 128: "#8172b2"}


def read_rows(path, topk):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[:topk] if topk and topk < len(rows) else rows


def languages(rows):
    cols = list(rows[0].keys())
    langs = [c[:-len("_var_before")] for c in cols
             if c.endswith("_var_before") and c != "all_var_before"]
    order = {c: i for i, c in enumerate(cols)}
    return sorted(langs, key=lambda x: order.get(x + "_var_before", 999))


def _add_density(ax, loadings, langs_rows=16):
    w = np.asarray(loadings, dtype=float)
    w = w[np.isfinite(w)]
    ax.hist(w, bins=60, color="#4c72b0", alpha=.65, edgecolor="white")
    for k in (16, 64, 128):
        thr = np.sort(np.abs(w))[-k]
        for sd in (-1, 1):
            ax.axvline(sd * thr, color=TH_COLORS[k], ls="--", lw=1.1,
                       alpha=.85, label=("top-%d |w|" % k) if sd == 1 else None)
    ax.set_xlabel("PC1 loading $w_j$")
    ax.set_ylabel("counts")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=.25)


def _add_pearson(ax, B, Af, langs, vmax=None):
    D = Af - B
    if vmax is None:
        vmax = max(float(np.abs(D).max()), 1e-6)
    mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.array(D, mask=mask), cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    for i in range(len(langs)):
        for j in range(len(langs)):
            if i >= j:
                ax.text(j, i, "%.2f\n%.2f" % (B[i, j], Af[i, j]),
                        ha="center", va="center", fontsize=7, color="black")
    return im


def _add_stats(ax, rows, langs, cross):
    def a(k):
        return np.array([float(r[k]) for r in rows])
    lines = ["MEAN STATS (top-%d dims)" % len(rows), ""]
    lines.append("all-rows var     %.3f -> %.3f" % (a("all_var_before").mean(), a("all_var_after3").mean()))
    wb = np.mean([a(lg + "_var_before").mean() for lg in langs])
    wa = np.mean([a(lg + "_var_after3").mean() for lg in langs])
    lines.append("within-lang var  %.3f -> %.3f" % (wb, wa))
    lines.append("")
    for lg in langs:
        lines.append("%-3s var          %.3f -> %.3f" % (lg, a(lg + "_var_before").mean(), a(lg + "_var_after3").mean()))
    if cross is not None:
        lines.append("")
        for k in (16, 64, 128):
            kb = "pearson_top%d_before" % k
            if kb in cross:
                B = np.array(cross[kb], dtype=float)
                Af = np.array(cross["pearson_top%d_after" % k], dtype=float)
                off = ~np.eye(len(B), dtype=bool)
                lines.append("mean r top-%-3d  %.3f -> %.3f" % (k, B[off].mean(), Af[off].mean()))
    ax.axis("off")
    ax.set_title("Mean statistics", fontsize=10, loc="left")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)


def plot_variance(rows, langs, dataset, outpath, title, rows_total,
                  per_lang_rows, cross=None, byk=None, loadings=None):
    n = len(rows)
    dim = np.array([int(r["dim"]) for r in rows])
    aw = np.abs(np.array([float(r["pc1_w"]) for r in rows]))

    def arr(k):
        return np.array([float(r[k]) for r in rows])

    allb, alla = arr("all_var_before"), arr("all_var_after3")
    mean_all_b, mean_all_a = float(allb.mean()), float(alla.mean())
    mean_w_b = float(np.mean([arr(lg + "_var_before").mean() for lg in langs]))
    mean_w_a = float(np.mean([arr(lg + "_var_after3").mean() for lg in langs]))
    has_cross = cross is not None and "pearson_top16_before" in cross
    total_rows = 3 + (1 if has_cross else 0) + (1 if loadings is not None else 0)
    fig = plt.figure(figsize=(13, 23 if loadings is not None else
                              (19 if has_cross else 15.5)))
    gs = fig.add_gridspec(total_rows, 2, hspace=0.38, wspace=0.28)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]

    # (a) all-rows variance scatter
    a = ax[0][0]
    sc = a.scatter(allb, alla, c=aw, cmap="viridis", s=45)
    m = max(allb.max(), alla.max())
    a.plot([0, m], [0, m], "k--", lw=1)
    a.set_xlabel("variance of all %s rows, before" % format(rows_total, ","))
    a.set_ylabel("after PC1-3")
    a.set_title("(a) all-rows variance, top-%d PC1 dims" % n)
    a.text(0.03, 0.97, "mean: %.3f -> %.3f" % (mean_all_b, mean_all_a),
           transform=a.transAxes, va="top", fontsize=9,
           bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    a.grid(alpha=.25)
    fig.colorbar(sc, ax=a, label="|PC1 loading|")

    # (b) per-language variance
    a = ax[0][1]
    for lg in langs:
        a.scatter(arr(lg + "_var_before"), arr(lg + "_var_after3"),
                  s=35, alpha=.75, color=COLORS.get(lg), label=lg)
    m = max(max(arr(lg + "_var_before").max(), arr(lg + "_var_after3").max())
            for lg in langs)
    a.plot([0, m], [0, m], "k--", lw=1)
    a.set_xlabel("variance within language, before")
    a.set_ylabel("after PC1-3")
    a.set_title("(b) per-language variance, %s rows each" % format(per_lang_rows, ","))
    a.text(0.03, 0.97, "mean within-lang: %.3f -> %.3f" % (mean_w_b, mean_w_a),
           transform=a.transAxes, va="top", fontsize=9,
           bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    a.legend(fontsize=8)
    a.grid(alpha=.25)

    # (c) bars
    a = ax[1][0]
    x = np.arange(n)
    a.bar(x - .2, allb, .4, label="before", color="#4c72b0")
    a.bar(x + .2, alla, .4, label="after PC1-3", color="#c44e52")
    a.set_xticks(x)
    a.set_xticklabels(dim, rotation=90, fontsize=7)
    a.set_ylabel("variance of all rows")
    a.set_title("(c) top-%d dims: all-rows variance" % n)
    a.text(0.03, 0.97, "mean: %.3f -> %.3f" % (mean_all_b, mean_all_a),
           transform=a.transAxes, va="top", fontsize=9,
           bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    a.legend(fontsize=8)
    a.grid(axis="y", alpha=.25)

    # (d) retention ratios
    a = ax[1][1]
    for lg in langs:
        a.plot(np.arange(1, n + 1), arr(lg + "_var_after3") / arr(lg + "_var_before"),
               color=COLORS.get(lg), label=lg)
    a.plot(np.arange(1, n + 1), alla / allb, "k-", lw=2, label="all rows")
    a.axhline(1, color="gray", ls=":", lw=1)
    a.set_xlabel("dimension rank by |PC1 loading|")
    a.set_ylabel("variance after / before")
    a.set_title("(d) variance retention after PC1-3")
    a.legend(fontsize=8)
    a.grid(alpha=.25)

    # (e) retention heatmap
    a = ax[2][0]
    H = np.stack([arr(lg + "_var_after3") / arr(lg + "_var_before") for lg in langs], axis=1)
    im = a.imshow(H, aspect="auto", cmap="magma", vmin=.5, vmax=1)
    a.set_xticks(range(len(langs)))
    a.set_xticklabels(langs)
    a.set_ylabel("top-%d dim rank" % n)
    a.set_title("(e) per-language variance retention")
    fig.colorbar(im, ax=a, label="ratio")

    # (f) mean variance vs top-k
    a = ax[2][1]
    if byk:
        ks = [int(r["topk"]) for r in byk]
        for lg in langs:
            a.plot(ks, [float(r[lg + "_before"]) for r in byk], "-o",
                   color=COLORS.get(lg), label=lg)
            a.plot(ks, [float(r[lg + "_after3"]) for r in byk], "--o",
                   color=COLORS.get(lg), alpha=.7)
        a.plot(ks, [float(r["all_before"]) for r in byk], "-ko", lw=2, label="all")
        a.plot(ks, [float(r["all_after3"]) for r in byk], "--ko", lw=2, alpha=.7)
        for xv in (64, 128):
            a.axvline(xv, color="gray", ls="--", lw=1, alpha=.5)
        a.set_xticks(ks)
        a.set_xticklabels([str(k) for k in ks])
        a.set_xlabel("number of top PC1 dimensions")
        a.set_ylabel("mean variance")
        a.set_title("(f) mean variance vs top-k dims")
        handles = [Line2D([0], [0], color=COLORS.get(lg), marker="o", lw=1.6, label=lg)
                   for lg in langs]
        handles += [Line2D([0], [0], color="black", linestyle="-", marker="o", lw=1.8, label="before"),
                    Line2D([0], [0], color="black", linestyle="--", marker="o", lw=1.8, label="after PC1-3")]
        a.legend(handles=handles, fontsize=7, ncol=2, loc="best")
        a.grid(alpha=.25)
    else:
        mb = [arr(lg + "_var_before").mean() for lg in langs]
        ma = [arr(lg + "_var_after3").mean() for lg in langs]
        x = np.arange(len(langs))
        a.bar(x - .2, mb, .4, label="before", color="#4c72b0")
        a.bar(x + .2, ma, .4, label="after PC1-3", color="#c44e52")
        a.set_xticks(x)
        a.set_xticklabels(langs)
        a.set_ylabel("mean variance over top-%d dims" % n)
        a.set_title("(f) mean per-language variance")
        a.legend(fontsize=8)
        a.grid(axis="y", alpha=.25)

    # (g)-(i) Pearson on top-16/64/128
    next_row = 3
    if has_cross:
        cl = cross["langs"]
        order = [cl.index(lg) for lg in langs]
        ks = [(k, lab) for k, lab in ((16, "(g)"), (64, "(h)"), (128, "(i)"))
              if "pearson_top%d_before" % k in cross]
        subgs = gs[next_row, :].subgridspec(1, len(ks) + 1,
                                            width_ratios=[1] * len(ks) + [0.05],
                                            wspace=0.35)
        vmax = 0.10 if dataset.lower() == "flores" else 0.02
        ims, paxes = [], []
        for kk, (k, lab) in enumerate(ks):
            B = np.array(cross["pearson_top%d_before" % k], dtype=float)[np.ix_(order, order)]
            Af = np.array(cross["pearson_top%d_after" % k], dtype=float)[np.ix_(order, order)]
            a = fig.add_subplot(subgs[0, kk])
            im = _add_pearson(a, B, Af, langs, vmax=vmax)
            ims.append(im); paxes.append(a)
            off = ~np.eye(len(langs), dtype=bool)
            mb, ma = float(B[off].mean()), float(Af[off].mean())
            a.set_title("%s Pearson r, top-%d (mean %.3f -> %.3f)" % (lab, k, mb, ma), fontsize=9)
        cax = fig.add_subplot(subgs[0, len(ks)])
        fig.colorbar(ims[0], cax=cax, label="after - before (scale +/-%.2f)" % vmax)
        next_row += 1

    # (j) density
    dens_letter = chr(ord("g") + len(ks)) if has_cross else "g"
    if loadings is not None:
        subgs2 = gs[next_row, :].subgridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.12)
        a = fig.add_subplot(subgs2[0, 0])
        _add_density(a, loadings, n)
        a.set_title("(%s) PC1 loading histogram (counts)" % dens_letter, fontsize=9)
        _add_stats(fig.add_subplot(subgs2[0, 1]), rows, langs, cross)

    legend = [
        "Panel key:",
        "(a) Each dot = one dimension; x = variance over all %s rows before; y = after PC1-3; below the diagonal = variance lost; color = |PC1 loading|." % format(rows_total, ","),
        "(b) Each dot = one dimension in one language; x = variance of the %s sentences in that language before; y = after; points near the diagonal keep their within-language variance." % format(per_lang_rows, ","),
        "(c) Bars = all-rows variance (%s rows) for each of the top-%d dimensions, before vs after PC1-3." % (format(rows_total, ","), n),
        "(d) Ratio after/before per dimension; colored = one language, black = all rows; 1.0 = no change, lower = more variance lost.",
        "(e) Heatmap: rows = top-%d dimensions, columns = languages, color = variance after/before; bright = retained, dark = lost." % n,
        "(f) Mean per-language variance for the top-16/64/128/256 dimensions; solid = before, dashed = after PC1-3; vertical dashed lines mark top-64 and top-128.",
    ]
    if has_cross:
        legend += [
            "(g) Pearson r on the top-16 PC1 dims: each cell shows before (top) and after PC1-3 (bottom); color = after - before, fixed scale per dataset across (g)-(i).",
            "(h) Pearson r on the top-64 PC1 dims: same encoding.",
            "(i) Pearson r on the top-128 PC1 dims: same encoding.",
        ]
    if loadings is not None:
        legend.append("(%s) Histogram of PC1 loadings across all dimensions (counts); dashed lines mark the top-16/64/128 |w| thresholds." % dens_letter)

    fig.text(0.012, 0.006, "\n".join(legend), fontsize=7.4, va="bottom",
             ha="left", linespacing=1.35)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(top=0.95, bottom=0.14, left=0.07, right=0.88)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_density(loadings, topk, dataset, outpath, title):
    fig, a = plt.subplots(figsize=(8.5, 5.0))
    _add_density(a, loadings, topk)
    a.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--loadings", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--rows-total", type=int, default=8024)
    ap.add_argument("--rows-per-lang", type=int, default=2006)
    ap.add_argument("--cross-json", default=None)
    ap.add_argument("--byk-csv", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = read_rows(args.csv, args.topk)
    langs = languages(rows)
    title = args.title or ("%s PC1 dimensions: variance before vs after PC1-3 removal"
                           % args.dataset.upper())
    cross = json.load(open(args.cross_json, encoding="utf-8")) if args.cross_json else None
    byk = list(csv.DictReader(open(args.byk_csv, encoding="utf-8"))) if args.byk_csv else None
    loadings = np.load(args.loadings)
    plot_variance(rows, langs, args.dataset,
                  os.path.join(args.outdir, "%s_pc1_variance_means_210m.png" % args.dataset),
                  title, args.rows_total, args.rows_per_lang, cross, byk, loadings)
    plot_density(loadings, args.topk, args.dataset,
                 os.path.join(args.outdir, "%s_pc1_weight_hist_210m.png" % args.dataset),
                 "%s PC1 loading histogram (counts)" % args.dataset.upper())


if __name__ == "__main__":
    main()
