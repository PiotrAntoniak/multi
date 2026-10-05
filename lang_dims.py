"""Which embedding dimensions carry the language? Per-dimension one-way ANOVA (eta^2) across
en/it/de/fr on the FLORES embeddings, plus the cumulative curve: how many top-ranked dimensions
are needed to capture 50/80/90/95% of the total language-separability mass.
Writes pca_plots/lang_dims_cumulative.png."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lang_metrics import eta2

BASE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "it", "de", "fr"]
OUT = os.path.join(BASE, "pca_plots")
os.makedirs(OUT, exist_ok=True)


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ("mean", "bos"):
        E = {l: np.load(os.path.join(BASE, "embeddings", mode, f"emb_{l}.npy")) for l in LANGS}
        X = np.vstack([E[l] for l in LANGS])
        y = np.repeat(LANGS, len(E["en"]))
        e2 = eta2(X, y)
        order = np.argsort(-e2)
        cum = np.cumsum(e2[order]) / e2.sum()
        ks = {f: int(np.searchsorted(cum, f) + 1) for f in (0.5, 0.8, 0.9, 0.95)}
        print(f"{mode}: dims for 50/80/90/95% = {ks[0.5]}/{ks[0.8]}/{ks[0.9]}/{ks[0.95]} "
              f"| total eta2 mass = {e2.sum():.1f} | max eta2 = {e2.max():.3f}")
        ax.plot(np.arange(1, len(cum) + 1), cum,
                label=f"{mode}  (50%: {ks[0.5]} dims, 90%: {ks[0.9]})")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("number of top dimensions (ranked by per-dim eta^2)")
    ax.set_ylabel("cumulative fraction of language-separability mass")
    ax.set_title("How many dimensions carry the language? (FLORES, en/it/de/fr)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "lang_dims_cumulative.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
