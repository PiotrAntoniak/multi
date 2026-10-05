"""Shared language-separability helpers for the FLORES analysis scripts.

Per-dimension eta^2 (one-way ANOVA over languages) and the top-k dimensions
covering a given fraction of the language-separability mass.  This was
previously copied verbatim into lang_dims.py, shap_flores_lang.py,
shap_flores_dropped.py, xgboost_flores_lang.py and xgboost_flores_matrix.py.
"""
import numpy as np


def eta2(X, y):
    """Per-dimension eta^2 of a one-way ANOVA over the groups in `y`."""
    overall = X.mean(0)
    ss_tot = ((X - overall) ** 2).sum(0)
    ss_bet = np.zeros(X.shape[1])
    for l in set(y):
        m = y == l
        ss_bet += m.sum() * (X[m].mean(0) - overall) ** 2
    return ss_bet / ss_tot


def eta2_lang(mode):
    """Per-dimension eta^2 over the four FLORES languages for one pooling mode."""
    import xgboost_topic_optuna as X   # lazy: keeps `eta2` usable without the xgboost stack
    E = {l: X.load_one_embedding(mode, l) for l in X.LANGS}
    Xall = np.vstack([E[l] for l in X.LANGS])
    y = np.repeat(X.LANGS, len(E["en"]))
    return eta2(Xall, y)


def top_dims(mode, frac):
    """Dims covering `frac` of the language eta^2 mass, ranked by separability."""
    e2 = eta2_lang(mode)
    order = np.argsort(-e2)
    cum = np.cumsum(e2[order]) / e2.sum()
    k = int(np.searchsorted(cum, frac) + 1)
    return order[:k], k
