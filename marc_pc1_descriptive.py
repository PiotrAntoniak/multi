"""MARC 6 descriptive PC1 analysis (no classifier).

Reads the cached MARC test embeddings, fits one global PCA on all rows, removes the
top three components, inverse-transforms, and writes the per-dimension measures used
by pc1_dimension_plots.py.

MARC is not a parallel corpus: the languages are independent review collections.
The per-dimension Pearson matrix is therefore a distributional correlation over
independent rows, not a sentence-aligned correlation.
"""
import csv
import json
import os

import numpy as np
from sklearn.decomposition import PCA

LANGS = ["en", "de", "es", "fr", "ja", "zh"]
EMB = "marc_emb"
OUT = "marc_pc1"


def load_split(split="test"):
    X = {lg: np.load(os.path.join(EMB, "%s_%s.npy" % (lg, split))) for lg in LANGS}
    n = min(v.shape[0] for v in X.values())
    X = {lg: v[:n].astype(np.float64) for lg, v in X.items()}
    return X, n


def moments(M, y, n_lang):
    means = np.stack([M[y == i].mean(0) for i in range(n_lang)])
    vars_ = np.stack([M[y == i].var(0) for i in range(n_lang)])
    return means, vars_, M.mean(0), M.var(0)


def main():
    X, n = load_split("test")
    A = np.vstack([X[lg] for lg in LANGS])
    y = np.repeat(np.arange(len(LANGS)), n)
    mu = A.mean(0)
    Ac = A - mu
    pca = PCA(svd_solver="full").fit(Ac)
    V = pca.components_[:3]
    Ak = mu + Ac - (Ac @ V.T) @ V

    mb, vb, ob, tb = moments(A, y, len(LANGS))
    ma, va, oa, ta = moments(Ak, y, len(LANGS))
    w = pca.components_[0]
    order = np.argsort(-np.abs(w))

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "marc_pc1_loadings.npy"), w)

    def write_csv(path, k):
        idx = order[:k]
        hdr = ["dim", "pc1_w"]
        for lg in LANGS:
            hdr += [lg + "_mean_before", lg + "_var_before"]
        hdr += ["all_mean_before", "all_var_before"]
        for lg in LANGS:
            hdr += [lg + "_mean_after3", lg + "_var_after3"]
        hdr += ["all_mean_after3", "all_var_after3"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(hdr)
            for d in idx:
                row = [int(d), float(w[d])]
                for li, lg in enumerate(LANGS):
                    row += [float(mb[li, d]), float(vb[li, d])]
                row += [float(ob[d]), float(tb[d])]
                for li, lg in enumerate(LANGS):
                    row += [float(ma[li, d]), float(va[li, d])]
                row += [float(oa[d]), float(ta[d])]
                wr.writerow(row)

    write_csv(os.path.join(OUT, "marc_pc1_top50_per_language_mean_var.csv"), 50)
    write_csv("marc_pc1_top16_per_language_mean_var.csv", 16)
    write_csv("marc_pc1_top50_per_language_mean_var.csv", 50)

    with open("marc_pc1_mean_variance_by_topk.csv", "w", newline="", encoding="utf-8") as f:
        hdr = ["topk"] + [lg + "_before" for lg in LANGS] + [lg + "_after3" for lg in LANGS] + ["all_before", "all_after3"]
        wr = csv.DictWriter(f, fieldnames=hdr)
        wr.writeheader()
        for k in [16, 64, 128, 256]:
            idx = order[:k]
            row = {"topk": k}
            for li, lg in enumerate(LANGS):
                row[lg + "_before"] = float(vb[li, idx].mean())
                row[lg + "_after3"] = float(va[li, idx].mean())
            row["all_before"] = float(tb[idx].mean())
            row["all_after3"] = float(ta[idx].mean())
            wr.writerow(row)

    pear = np.eye(len(LANGS))
    cent = np.eye(len(LANGS))
    C = {lg: A[y == i].mean(0) for i, lg in enumerate(LANGS)}
    for i in range(len(LANGS)):
        for j in range(i + 1, len(LANGS)):
            x, z = A[y == i], A[y == j]
            rs = [np.corrcoef(x[:, d], z[:, d])[0, 1]
                  for d in range(x.shape[1])
                  if x[:, d].std() > 1e-12 and z[:, d].std() > 1e-12]
            pear[i, j] = pear[j, i] = float(np.mean(rs))
            cent[i, j] = cent[j, i] = float(
                C[LANGS[i]] @ C[LANGS[j]] /
                (np.linalg.norm(C[LANGS[i]]) * np.linalg.norm(C[LANGS[j]])))
    def pmat(M, dims):
        P = np.eye(len(LANGS))
        for i in range(len(LANGS)):
            for j in range(i + 1, len(LANGS)):
                x = M[y == i][:, dims]; z = M[y == j][:, dims]
                P[i, j] = P[j, i] = float(np.mean(
                    [np.corrcoef(x[:, d], z[:, d])[0, 1] for d in range(len(dims))]))
        return P
    cross = {"langs": LANGS, "rows_per_lang": n,
             "pearson_dim": pear.tolist(), "centroid_cosine": cent.tolist(),
             "note": "independent MARC reviews; Pearson is distributional, not sentence-aligned"}
    for k in (16, 64, 128):
        dims = order[:k]
        cross["pearson_top%d_before" % k] = pmat(A, dims).tolist()
        cross["pearson_top%d_after" % k] = pmat(Ak, dims).tolist()
    with open("marc_pc1_cross_correlation.json", "w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2)

    print("rows/language", n, "dims", A.shape[1])
    print("PC1 EVR", pca.explained_variance_ratio_[0])
    print("pearson", np.round(pear, 3))
    print("centroid cosine", np.round(cent, 3))
    print("wrote marc_pc1_* tables")


if __name__ == "__main__":
    main()
