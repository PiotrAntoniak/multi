"""Reusable plots for the leading-PC variance analysis.

Two outputs:
  <dataset>_pc1_variance_210m.png       six-panel variance summary
  <dataset>_pc1_weight_density_210m.png PC1 loading density

The variance panels use the first --topk rows of a per-dimension CSV.
Required CSV columns:
  dim, pc1_w
  <lang>_mean_before, <lang>_var_before, <lang>_mean_after3, <lang>_var_after3
  all_mean_before, all_var_before, all_mean_after3, all_var_after3

The density plot uses a .npy vector of all PC1 loadings (one per original dim).

Examples
--------
python pc1_dimension_plots.py --dataset flores \
  --csv flores_pc1_top16_per_language_mean_var.csv \
  --loadings flores_pc1_loadings.npy --outdir pc1_210

python pc1_dimension_plots.py --dataset marc \
  --csv marc_pc1_top16_per_language_mean_var.csv \
  --loadings marc_pc1_loadings.npy --outdir pc1_marc
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path, topk):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if topk and topk < len(rows):
        rows = rows[:topk]
    return rows


def languages(rows):
    cols = rows[0].keys()
    langs = [c[:-len("_var_before")] for c in cols
             if c.endswith("_var_before") and c != "all_var_before"]
    order = {c: i for i, c in enumerate(rows[0].keys())}
    return sorted(langs, key=lambda x: order.get(x + "_var_before", 999))


def plot_variance(rows, langs, dataset, outpath, title, rows_total, per_lang_rows, cross=None, byk=None):
    dim = np.array([int(r["dim"]) for r in rows])
    aw = np.abs(np.array([float(r["pc1_w"]) for r in rows]))
    n = len(rows)

    def arr(k):
        return np.array([float(r[k]) for r in rows])

    allb, alla = arr("all_var_before"), arr("all_var_after3")
    colors = {"en": "#1f77b4", "it": "#ff7f0e", "de": "#2ca02c",
              "fr": "#d62728", "es": "#9467bd", "ja": "#8c564b",
              "zh": "#e377c2"}

    nrows = 4 if cross is not None else 3
    fig, ax = plt.subplots(nrows, 2, figsize=(13, 19.5 if cross is not None else 15.5))

    a = ax[0, 0]
    sc = a.scatter(allb, alla, c=aw, cmap="viridis", s=45)
    m = max(allb.max(), alla.max())
    a.plot([0, m], [0, m], "k--", lw=1)
    a.set_xlabel("variance of all %s rows, before" % format(rows_total, ","))
    a.set_ylabel("after PC1-3")
    a.set_title("(a) all-rows variance, top-%d PC1 dims" % n)
    a.grid(alpha=.25)
    fig.colorbar(sc, ax=a, label="|PC1 loading|")

    a = ax[0, 1]
    for lg in langs:
        a.scatter(arr(lg + "_var_before"), arr(lg + "_var_after3"),
                  s=35, alpha=.75, color=colors.get(lg), label=lg)
    m = max(max(arr(lg + "_var_before").max(), arr(lg + "_var_after3").max())
            for lg in langs)
    a.plot([0, m], [0, m], "k--", lw=1)
    a.set_xlabel("variance within language, before")
    a.set_ylabel("after PC1-3")
    a.set_title("(b) per-language variance, %s rows each" % format(per_lang_rows, ","))
    a.legend(fontsize=8)
    a.grid(alpha=.25)

    a = ax[1, 0]
    x = np.arange(n)
    a.bar(x - .2, allb, .4, label="before", color="#4c72b0")
    a.bar(x + .2, alla, .4, label="after PC1-3", color="#c44e52")
    a.set_xticks(x)
    a.set_xticklabels(dim, rotation=90, fontsize=7)
    a.set_ylabel("variance of all rows")
    a.set_title("(c) top-%d dims: all-rows variance" % n)
    a.legend(fontsize=8)
    a.grid(axis="y", alpha=.25)

    a = ax[1, 1]
    for lg in langs:
        a.plot(np.arange(1, n + 1), arr(lg + "_var_after3") / arr(lg + "_var_before"),
               color=colors.get(lg), label=lg)
    a.plot(np.arange(1, n + 1), alla / allb, "k-", lw=2, label="all rows")
    a.axhline(1, color="gray", ls=":", lw=1)
    a.set_xlabel("dimension rank by |PC1 loading|")
    a.set_ylabel("variance after / before")
    a.set_title("(d) variance retention after PC1-3")
    a.legend(fontsize=8)
    a.grid(alpha=.25)

    a = ax[2, 0]
    H = np.stack([arr(lg + "_var_after3") / arr(lg + "_var_before") for lg in langs], axis=1)
    im = a.imshow(H, aspect="auto", cmap="magma", vmin=.5, vmax=1)
    a.set_xticks(range(len(langs)))
    a.set_xticklabels(langs)
    a.set_ylabel("top-%d dim rank" % n)
    a.set_title("(e) per-language variance retention")
    fig.colorbar(im, ax=a, label="ratio")

    a = ax[2, 1]
    if byk:
        ks = [int(r["topk"]) for r in byk]
        for lg in langs:
            a.plot(ks, [float(r[lg + "_before"]) for r in byk], "-o", color=colors.get(lg), label=lg)
            a.plot(ks, [float(r[lg + "_after3"]) for r in byk], "--o", color=colors.get(lg), alpha=.7)
        a.plot(ks, [float(r["all_before"]) for r in byk], "-ko", lw=2, label="all")
        a.plot(ks, [float(r["all_after3"]) for r in byk], "--ko", lw=2, alpha=.7)
        a.set_xticks(ks)
        a.set_xticklabels([str(k) for k in ks])
        a.set_xlabel("number of top PC1 dimensions")
        a.set_ylabel("mean variance")
        a.set_title("(f) mean variance vs top-k dims")
        a.legend(fontsize=7, ncol=2)
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


    if cross is not None:
        cl = cross["langs"]
        order = [cl.index(l) for l in langs]
        mats = [(np.array(cross["pearson_dim"], dtype=float), "Pearson r", "(g) cross-correlation: per-dimension Pearson r", "r")]
        if "cosine_sentence" in cross:
            mats.append((np.array(cross["cosine_sentence"], dtype=float), "cosine", "(h) cross-similarity: sentence-aligned cosine", "cosine"))
        for k, (M, lbl, tt, cbl) in enumerate(mats):
            M = M[np.ix_(order, order)]
            a = ax[3, k]
            im = a.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
            a.set_xticks(range(len(langs))); a.set_xticklabels(langs)
            a.set_yticks(range(len(langs))); a.set_yticklabels(langs)
            for i in range(len(langs)):
                for j in range(len(langs)):
                    a.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                           color="white" if M[i, j] < 0.7 else "black", fontsize=8)
            a.set_title(tt)
            fig.colorbar(im, ax=a, label=cbl)
    legend = "\n".join([
        "Panel key:",
        "(a) Each dot = one dimension; x = variance over all %s rows before; y = after PC1-3; below the diagonal = variance lost; color = |PC1 loading|." % format(rows_total, ","),
        "(b) Each dot = one dimension in one language; x = variance of the %s sentences in that language before; y = after; points near the diagonal keep their within-language variance." % format(per_lang_rows, ","),
        "(c) Bars = all-rows variance (%s rows) for each of the top-%d dimensions, before vs after PC1-3." % (format(rows_total, ","), n),
        "(d) Ratio after/before per dimension; colored = one language, black = all rows; 1.0 = no change, lower = more variance lost.",
        "(e) Heatmap: rows = top-%d dimensions, columns = languages, color = variance after/before; bright = retained, dark = lost." % n,
        "(f) Mean per-language variance for the top-16/64/128/256 dimensions; solid = before, dashed = after PC1-3.",
        "(g) Cross-correlation between languages: mean per-dimension Pearson r over the %s rows per language; 1.0 = identical variation." % format(per_lang_rows, ","),
        "(h) Cross-similarity: mean cosine between the two embeddings of the same sentence; 1.0 = identical direction." if cross is not None and "cosine_sentence" in cross else None,
    ])
    legend = "\n".join([x for x in legend.split("\n") if x != "None"])
    fig.text(0.012, 0.006, legend, fontsize=7.6, va="bottom", ha="left", linespacing=1.35)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.13, 1, .96])
    fig.subplots_adjust(hspace=0.30)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_density(loadings, topk, dataset, outpath, title):
    w = np.asarray(loadings, dtype=float)
    w = w[np.isfinite(w)]
    fig, a = plt.subplots(figsize=(8.5, 5.0))
    a.hist(w, bins=60, density=True, color="#4c72b0", alpha=.65, edgecolor="white")
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(w.min(), w.max(), 400)
        a.plot(xs, gaussian_kde(w)(xs), color="#c44e52", lw=2, label="density")
    except Exception:
        pass
    thr = np.sort(np.abs(w))[-topk]
    for s in (-1, 1):
        a.axvline(s * thr, color="#c44e52", ls="--", lw=1.2,
                  label=("top-%d |w| threshold" % topk) if s == 1 else None)
    a.set_xlabel("PC1 loading $w_j$")
    a.set_ylabel("density")
    a.set_title(title)
    a.legend(fontsize=8)
    a.grid(alpha=.25)
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
    title = args.title or ("%s PC1 dimensions: variance before vs after PC1-3 removal" % args.dataset.upper())
    cross = json.load(open(args.cross_json, encoding="utf-8")) if args.cross_json else None
    byk = list(csv.DictReader(open(args.byk_csv, encoding="utf-8"))) if args.byk_csv else None
    plot_variance(rows, langs, args.dataset,
                  os.path.join(args.outdir, "%s_pc1_variance_210m.png" % args.dataset), title,
                  args.rows_total, args.rows_per_lang, cross, byk)
    plot_density(np.load(args.loadings), args.topk, args.dataset,
                 os.path.join(args.outdir, "%s_pc1_weight_density_210m.png" % args.dataset),
                 "%s PC1 loading density (top-%d marked)" % (args.dataset.upper(), args.topk))


if __name__ == "__main__":
    main()
