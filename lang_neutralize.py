"""Shared language-neutralization primitives for the cross-lingual experiments.

Pure numpy (float64 internally). Operates on a dict ``X_by_lang = {lang: array(n_lang, d)}``:
eta^2 per dim, zero-dims, language centroids/subspace projection, per-language
centering, and All-but-the-Top (ABTT, Mu & Viswanath ICLR 2018, arXiv:1702.01417).
"""
import numpy as np


def eta2_per_dim(X_by_lang, langs=None):
    """Per-dimension eta^2 of a one-way ANOVA over the languages."""
    langs = list(X_by_lang.keys()) if langs is None else list(langs)
    mats = {l: np.asarray(X_by_lang[l], dtype=np.float64) for l in langs}
    X = np.vstack([mats[l] for l in langs])
    overall = X.mean(0)
    ss_tot = ((X - overall) ** 2).sum(0)
    ss_bet = np.zeros(X.shape[1], dtype=np.float64)
    for l in langs:
        ss_bet += len(mats[l]) * (mats[l].mean(0) - overall) ** 2
    return np.divide(ss_bet, ss_tot, out=np.zeros_like(ss_bet), where=ss_tot > 0)


def top_dims(e2, frac):
    """Indices of the top dims covering `frac` of the total eta^2 mass (desc)."""
    order = np.argsort(-np.asarray(e2, dtype=np.float64))
    cum = np.cumsum(np.asarray(e2)[order]) / np.asarray(e2).sum()
    k = int(np.searchsorted(cum, frac) + 1)
    return order[:k], k


def zero_dims(X, dims):
    """Copy of X with columns `dims` set to 0."""
    X = np.array(X, dtype=np.float64, copy=True)
    X[:, np.asarray(dims, dtype=int)] = 0.0
    return X


def centroids(X_by_lang, langs=None):
    langs = list(X_by_lang.keys()) if langs is None else list(langs)
    return {l: np.asarray(X_by_lang[l], dtype=np.float64).mean(0) for l in langs}


def language_subspace(X_by_lang, demean=True, tol=1e-9, max_rank=None):
    """Orthonormal basis (r, d) spanning the language centroids. Returns (V, r)."""
    C = np.stack([np.asarray(X_by_lang[l], dtype=np.float64).mean(0) for l in X_by_lang])
    if demean:
        C = C - C.mean(0)
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    r = int(np.sum(s > tol))
    if max_rank is not None:
        r = min(r, int(max_rank))
    return Vt[:r].copy(), r


def project_out(X, V):
    """Remove the row span of V (r, d) from every row of X."""
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] == 0:
        return X.copy()
    return X - (X @ V.T) @ V


def center_by_lang(X, centroid):
    return np.asarray(X, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)


def abtt_fit(mat, n_components=None):
    """Fit ABTT on `mat`: return (mu, U) with U rows = top-D principal dirs."""
    mat = np.asarray(mat, dtype=np.float64)
    mu = mat.mean(0)
    centered = mat - mu
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    if n_components is None:
        n_components = max(1, int(round(mat.shape[1] / 100.0)))
    n_components = min(int(n_components), Vt.shape[0])
    return mu, Vt[:n_components].copy()


def abtt_apply(arr, mu, U):
    """Paper-faithful ABTT projection: (v - mu) - (v @ U.T) @ U."""
    arr = np.asarray(arr, dtype=np.float64)
    return (arr - mu) - (arr @ U.T) @ U


def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n
