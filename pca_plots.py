"""PCA plots of the FLORES-200 EuroBERT embeddings (local analysis; script + outputs are
gitignored). Per pooling mode: all 4 languages on one figure, raw (PCA fit on all languages
pooled) vs +D (PCA fit on the English embeddings only), plus a PC1-removed variant (PC2 vs PC3).
For `mean` additionally: colored by primary lenient topic."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import xgboost_topic_optuna as X   # lenient targets for topic colors (read-only import)

BASE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "it", "de", "fr"]
MODES = ["mean", "bos", "eos", "lead"]
OUT = os.path.join(BASE, "pca_plots")
os.makedirs(OUT, exist_ok=True)
LANG_COLORS = {"en": "#1f77b4", "it": "#2ca02c", "de": "#d62728", "fr": "#9467bd"}


def load(mode):
    return {l: np.load(os.path.join(BASE, "embeddings", mode, f"emb_{l}.npy")) for l in LANGS}


def scatter(ax, pts, labels, colors, title, legend=True):
    for lab in sorted(set(labels)):
        m = np.array([x == lab for x in labels])
        ax.scatter(pts[m, 0], pts[m, 1], s=4, alpha=0.5, color=colors[lab], label=lab)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        ax.legend(markerscale=3, fontsize=6, loc="best")


def pca23(Xfit, Xplot):
    """Project onto PC2/PC3 (PC1 removed) of a PCA fitted on Xfit."""
    return PCA(3, random_state=0).fit(Xfit).transform(Xplot)[:, 1:3]


def main():
    # 1) per-mode: all languages; raw = PCA fit on all languages, +D = PCA fit on English only
    for mode in MODES:
        E = load(mode)
        labels = sum([[l] * len(E[l]) for l in LANGS], [])
        shifts = {l: (E["en"] - E[l]).mean(0) for l in LANGS}
        Xraw = np.vstack([E[l] for l in LANGS])
        Xshift = np.vstack([E[l] + shifts[l] for l in LANGS])
        Praw = PCA(2, random_state=0).fit_transform(Xraw)      # fit on all languages pooled
        pca_en = PCA(2, random_state=0).fit(E["en"])           # fit on English embeddings only
        Pshift = pca_en.transform(Xshift)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
        scatter(axes[0], Praw, labels, LANG_COLORS, "raw — PCA fit on all languages")
        scatter(axes[1], Pshift, labels, LANG_COLORS, "+D — PCA fit on English only")
        fig.suptitle(f"FLORES {mode} — all languages", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"pca_lang_{mode}.png"), dpi=150)
        plt.close(fig)
        Praw23 = pca23(Xraw, Xraw)
        Pshift23 = pca23(E["en"], Xshift)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
        scatter(axes[0], Praw23, labels, LANG_COLORS, "raw — no PC1 (fit on all languages)")
        scatter(axes[1], Pshift23, labels, LANG_COLORS, "+D — no PC1 (fit on en)")
        fig.suptitle(f"FLORES {mode} — all languages, PC1 removed (PC2 vs PC3)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"pca_lang_{mode}_noPC1.png"), dpi=150)
        plt.close(fig)

    # 2) mean mode: language vs primary topic
    E = load("mean")
    Xall = np.vstack([E[l] for l in LANGS])
    P = PCA(2, random_state=0).fit_transform(Xall)
    _, Y, _, _ = X.load_data("lenient")
    prim = []
    for row in Y:
        idx = np.where(row == 1)[0]
        prim.append(X.TAGS[idx[0]] if len(idx) else "none")
    prim4 = prim * len(LANGS)          # same rows for every language
    cmap = plt.get_cmap("tab10")
    tag_colors = {t: cmap(i % 10) for i, t in enumerate(X.TAGS)}
    tag_colors["none"] = "#cccccc"
    labels_l = sum([[l] * len(E[l]) for l in LANGS], [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    scatter(axes[0], P, labels_l, LANG_COLORS, "by language")
    scatter(axes[1], P, prim4, tag_colors, "by primary topic (lenient)")
    fig.suptitle("FLORES mean — PCA (all 4 languages pooled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pca_mean_lang_topic.png"), dpi=150)
    plt.close(fig)
    P23 = pca23(Xall, Xall)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    scatter(axes[0], P23, labels_l, LANG_COLORS, "by language — no PC1")
    scatter(axes[1], P23, prim4, tag_colors, "by primary topic (lenient) — no PC1")
    fig.suptitle("FLORES mean — PC1 removed (PC2 vs PC3), all 4 languages pooled", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pca_mean_lang_topic_noPC1.png"), dpi=150)
    plt.close(fig)

    print("wrote plots to", OUT)
    for f in sorted(os.listdir(OUT)):
        print(" ", f, os.path.getsize(os.path.join(OUT, f)))


if __name__ == "__main__":
    main()
