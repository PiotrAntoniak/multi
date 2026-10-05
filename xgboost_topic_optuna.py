"""
English-trained XGBoost topic classifier on FLORES embeddings, with a sweep over the
four pooling modes (mean / bos / eos / lead) and cross-lingual transfer evaluation.

Usage:
  # full sweep, all four modes, 50 Optuna trials each (writes per-mode + combined docs)
  python xgboost_topic_optuna.py --mode all --trials 50

  # single mode
  python xgboost_topic_optuna.py --mode mean --trials 50

  # per-language x per-pooling grid (one language per worker)
  python xgboost_topic_optuna.py --mode perlang --langs en --trials 50
  # optional filters for plumbing checks:
  python xgboost_topic_optuna.py --mode perlang --langs en --modes mean --trials 2 --outdir <tmp>

  # lenient label scheme + per-tag tuned thresholds (the label audit fix):
  python xgboost_topic_optuna.py --mode perlang --langs en --labels lenient --thresholds tuned --trials 50
  # merge the four per-language JSONs into one markdown grid
  python xgboost_topic_optuna.py --mode merge --labels lenient --thresholds tuned

  # fast plumbing check (small trial count, isolated output dir)
  python xgboost_topic_optuna.py --mode mean --trials 2 --outdir <tmpdir>

Outputs (next to the script, or in --outdir):
  - xgboost_topic_results_{mode}.md   per-mode write-up (--mode all/mean/...)
  - xgboost_optuna_trials_{mode}.csv  every Optuna trial for that mode
  - xgboost_topic_allmodes.md         combined comparison (written only for --mode all)
  - xgboost_perlang_{lang}.json       per-language metrics/params/timings (--mode perlang)
  - xgboost_perlang_trials_{lang}_{mode}.csv  every Optuna trial for that (lang, mode)
  - xgboost_perlang_{lang}.pid / _status.json worker pid + live status
  - xgboost_perlang.md                combined per-language grid (--mode merge)
  # with --labels lenient the same artifacts use the xgboost_perlang_lenient_* prefix:
  - xgboost_perlang_lenient_{lang}.json / _trials_{lang}_{mode}.csv / _{lang}.pid / _{lang}_status.json
  - xgboost_perlang_lenient.md        combined lenient grid (--mode merge --labels lenient)

Per-language protocol (--mode perlang): baseline (default params) and every Optuna trial train
on `dev` (997 rows) and score macro-F1 on `devtest` (506); best params are retrained on
dev+devtest (1503) and evaluated on `test` (506). No cross-lingual transfer. The four languages
run independently in parallel workers (one process per language).

Labels (--labels): `strict` (default) keeps the historical exact canonical-string match;
`lenient` maps raw comma-tags to the 8 canonical tags via the curated aliases in
`the label audit` §1 (recovers dropped positives; all-zero rows 1147 -> ~515/2009).

Thresholds (--thresholds): `fixed` (default) = 0.5; `tuned` = per-tag threshold in 0.05..0.95
maximising that tag's F1 on the model's own training-set predictions. `run_perlang_mode` always
reports BOTH fixed-0.5 and tuned metrics; the flag selects the merge view. Macro-F1 is reported
over all 8 tags and over tags with >= 50 positives.

Performance note: max_bin=64 (XGBoost default is 256) is fixed for all models. It is a
pure speed setting for the 2-thread runtime budget and changes scores only marginally.
Threads are pinned to 2 via OMP_NUM_THREADS / MKL_NUM_THREADS.
"""
import os

# Must be set before numpy / xgboost import to take effect.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import f1_score, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "flores200_en_it_de_fr.csv")
LANGS = ["en", "it", "de", "fr"]
MODES = ["mean", "bos", "eos", "lead"]
TAGS = ["travel", "sports", "crime and law", "politics",
        "health", "geography", "science", "safety"]
MAX_BIN = 64
N_JOBS = 2
SEED = 0
THRESHOLD = 0.5
# Per-tag threshold search grid (the label audit §2 tunes in 0.1..0.9; we use a
# slightly wider 0.05..0.95 as requested).
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
# Macro-F1 is reported both over all 8 tags and over the tags with at least this
# many positives (the audit recommends stating the cutoff explicitly).
MIN_POS_FOR_MACRO = 50

# Baseline protocol params (fixed by the experiment spec).
BASELINE_PARAMS = dict(max_depth=6, learning_rate=0.1, n_estimators=300)

STATUS_PATH = os.path.join(BASE, "xgboost_allmodes_status.json")


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Label schemes (strict = exact canonical match; lenient = curated aliases)
# ----------------------------------------------------------------------------
# the label audit §1: the raw `topic` column uses many names for the same
# concepts. `strict` keeps only exact canonical strings (the historical
# behaviour); `lenient` maps closely-related raw tags onto the 8 canonical tags.
#
# The alias sets below reproduce the audit's "lenient rows" per-tag counts to
# within a few rows (see AUDIT_LENIENT_TARGETS / _report_lenient_counts).
LENIENT_EXACT = {
    "travel": {
        "travel", "tourism", "accomodation", "accommodation",
        "introductory_tourism", "camping",
    },
    "sports": {
        "sports", "sport", "sports events", "horse riding", "golf",
        "ice skates", "wiki/fencing", "sailing",
    },
    "crime and law": {"crime and law", "crime", "law", "law and crime"},
    "politics": {
        "politics", "politics and conflicts", "policy", "international",
        "internatoinal", "warfare",
    },
    "health": {
        "health", "disease", "medicine", "roman culture/medicine",
        "circulatory system", "alchohol", "beverages", "food",
    },
    "geography": {
        "geography", "antartica", "world heritage", "south_america/the_amazon",
        "europe/vatican_city", "machu picchu", "canada", "italy", "japan",
        "taiwan", "turkey", "luxembourg", "bhutan", "cambodia", "venezuela",
    },
    "safety": {
        "safety", "disasters and accidents", "accident", "accidents",
        "disaster", "weather", "snow", "tragety", "climate", "water",
    },
    "science": {
        "sciece", "biology", "biology/cells", "biology/kingdoms/animals",
        "dinosaurs", "triceratops", "mosasaurus",
        "wikijunior:dinosaurs/extinction", "solar system", "solar_system/moon",
        "solar_system/space_exploration", "solar_system/solar_system",
        "elements", "the_elements/matter_is_the_stuff_around_you",
        "the_elements/periodic_table", "the_elements/mixtures",
        "the_elements/overview", "origin of life", "how_things_work/time",
        "hydrogen", "sun", "venus", "jupiter", "io", "plants", "insects",
        "bugs/insects_intro", "wheel", "lightbulb", "rocket", "nature",
        "wildlife", "big cats", "lion", "tiger", "ocelot", "serengeti",
        "environment", "climate",
    },
}
# Prefix rules (startswith).
LENIENT_PREFIX = {
    "travel": ("reason to travel", "natural wonders/botanical tourism"),
    "politics": ("world_war_ii", "american_revolution", "united_states_charters"),
    "geography": ("natural wonders", "seven wonders of the world"),
    "science": ("science",),  # covers `science/*` and `science and technology`
}
# Substring rules (used for the irregular cold-war spellings).
LENIENT_CONTAINS = {
    "politics": ("cold_war", "cold war"),
}

# the label audit §1 "exact vs lenient match per canonical tag" targets.
AUDIT_LENIENT_TARGETS = {
    "travel": 518, "sports": 148, "crime and law": 121, "politics": 177,
    "health": 81, "geography": 117, "science": 317, "safety": 151,
}
AUDIT_LENIENT_ZERO_ROWS = 504


def _lenient_map_tag(tag):
    """Map one lowercased/stripped raw comma-tag to a set of canonical tags."""
    out = set()
    for canon, aliases in LENIENT_EXACT.items():
        if tag in aliases:
            out.add(canon)
    for canon, prefixes in LENIENT_PREFIX.items():
        if tag.startswith(prefixes):
            out.add(canon)
    for canon, needles in LENIENT_CONTAINS.items():
        if any(n in tag for n in needles):
            out.add(canon)
    return out


def build_targets(df, scheme="strict"):
    """Build the (n_rows, 8) multi-label target matrix for a label `scheme`.

    strict : exact canonical-string matching (historical behaviour).
    lenient: curated alias mapping of raw comma-tags to the 8 canonical tags
             (see the label audit §1 and LENIENT_EXACT/PREFIX/CONTAINS above).
    """
    if scheme not in ("strict", "lenient"):
        raise ValueError(f"unknown label scheme: {scheme!r}")
    Y = np.zeros((len(df), len(TAGS)), dtype=int)
    for i, topic in enumerate(df["topic"].fillna("")):
        parts = {p.strip().lower() for p in str(topic).split(",") if p.strip()}
        if scheme == "strict":
            for j, tag in enumerate(TAGS):
                if tag in parts:
                    Y[i, j] = 1
        else:
            for tag in parts:
                for canon in _lenient_map_tag(tag):
                    Y[i, TAGS.index(canon)] = 1
    return Y


def _report_lenient_counts(Y):
    """Log lenient per-tag / all-zero counts against the audit targets."""
    counts = Y.sum(axis=0)
    zero = int((Y.sum(axis=1) == 0).sum())
    log("[lenient] per-tag row counts vs the label audit targets:")
    for j, tag in enumerate(TAGS):
        tgt = AUDIT_LENIENT_TARGETS[tag]
        log(f"[lenient]   {tag:<14} got={counts[j]:>4}  target={tgt:>4}  "
            f"delta={counts[j] - tgt:+d}")
    log(f"[lenient]   all-zero rows  got={zero:>4}  target={AUDIT_LENIENT_ZERO_ROWS:>4}  "
        f"delta={zero - AUDIT_LENIENT_ZERO_ROWS:+d}")


# ----------------------------------------------------------------------------
# Data (split identical for every mode)
# ----------------------------------------------------------------------------
def load_data(scheme="strict"):
    df = pd.read_csv(CSV_PATH)
    assert list(df.columns)[:2] == ["split", "id"]
    assert len(df) == 2009

    # Multi-label target over the 8 canonical tags.
    Y = build_targets(df, scheme)

    # Three-way split: seeded shuffle of the 1012 devtest rows.
    #   dev     = original dev                            ->  997 rows (train)
    #   devtest = first 506 of shuffled original devtest  ->  506 rows
    #   test    = last  506 of shuffled original devtest  ->  506 rows
    orig_dev = (df["split"] == "dev").values
    orig_devtest = (df["split"] == "devtest").values
    _dt_idx = np.where(orig_devtest)[0]
    _rng = np.random.default_rng(0)
    _shuf = _rng.permutation(_dt_idx)
    dev_mask = orig_dev.copy()
    devtest_mask = np.zeros(len(df), dtype=bool)
    devtest_mask[_shuf[:506]] = True
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[_shuf[506:]] = True
    dev_devtest_mask = dev_mask | devtest_mask
    assert dev_mask.sum() == 997
    assert devtest_mask.sum() == 506 and test_mask.sum() == 506
    assert dev_devtest_mask.sum() == 1503
    # Every canonical tag must be present in both evaluation splits.
    assert (Y[devtest_mask].sum(axis=0) > 0).all(), "a tag has 0 positives in devtest"
    assert (Y[test_mask].sum(axis=0) > 0).all(), "a tag has 0 positives in test"

    masks = dict(dev=dev_mask, devtest=devtest_mask, test=test_mask,
                 dev_devtest=dev_devtest_mask)
    counts = dict(
        all=Y.sum(axis=0),
        dev=Y[dev_mask].sum(axis=0),
        devtest=Y[devtest_mask].sum(axis=0),
        test=Y[test_mask].sum(axis=0),
    )
    return df, Y, masks, counts


def load_mode_embeddings(mode):
    X = {l: np.load(os.path.join(BASE, "embeddings", mode, f"emb_{l}.npy")) for l in LANGS}
    for l in LANGS:
        assert X[l].shape == (2009, 768) and X[l].dtype == np.float32, (mode, l, X[l].shape)
    return X


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
def make_params(extra=None):
    p = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
        seed=SEED,
        max_bin=MAX_BIN,
    )
    if extra:
        p.update(extra)
    return p


def fit_predict(params, X_train, Y_train, X_tests):
    """Train one binary XGBClassifier per tag; return (probs per test set, models)."""
    models = []
    probs_list = [np.zeros((Xt.shape[0], len(TAGS))) for Xt in X_tests]
    for j in range(len(TAGS)):
        model = xgb.XGBClassifier(**make_params(params))
        model.fit(X_train, Y_train[:, j])
        models.append(model)
        for k, Xt in enumerate(X_tests):
            probs_list[k][:, j] = model.predict_proba(Xt)[:, 1]
    return probs_list, models


def fit_predict_with_train(params, X_train, Y_train, X_tests):
    """Like fit_predict, but also return in-sample training probabilities
    (needed for per-tag threshold tuning on the training set)."""
    models = []
    train_probs = np.zeros((X_train.shape[0], len(TAGS)))
    probs_list = [np.zeros((Xt.shape[0], len(TAGS))) for Xt in X_tests]
    for j in range(len(TAGS)):
        model = xgb.XGBClassifier(**make_params(params))
        model.fit(X_train, Y_train[:, j])
        models.append(model)
        train_probs[:, j] = model.predict_proba(X_train)[:, 1]
        for k, Xt in enumerate(X_tests):
            probs_list[k][:, j] = model.predict_proba(Xt)[:, 1]
    return train_probs, probs_list, models


def score(Y_true, probs, thresholds=None, pos_counts=None):
    """Score multi-label predictions.

    `thresholds` may be a scalar (all tags) or a per-tag array; default 0.5.
    `pos_counts` is the per-tag positive count used for the >= MIN_POS_FOR_MACRO
    macro rule (defaults to the evaluation-set counts). Pass the full-dataset
    counts so the rule reflects label prevalence, not the eval split size.
    Returns a dict with subset accuracy, macro-F1 (all 8 tags), micro-F1,
    per-tag F1, and macro-F1 over the qualifying tags.
    """
    if thresholds is None:
        thr = np.full(probs.shape[1], THRESHOLD, dtype=float)
    else:
        thr = np.asarray(thresholds, dtype=float)
        if thr.ndim == 0:
            thr = np.full(probs.shape[1], float(thr), dtype=float)
    pred = (probs >= thr[None, :]).astype(int)
    subset_acc = accuracy_score(Y_true, pred)
    macro_f1 = f1_score(Y_true, pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_true, pred, average="micro", zero_division=0)
    per_tag = f1_score(Y_true, pred, average=None, zero_division=0)
    per_tag_acc = (pred == Y_true).mean(axis=0)
    pos = np.asarray(pos_counts) if pos_counts is not None else Y_true.sum(axis=0)
    keep = pos >= MIN_POS_FOR_MACRO
    macro_f1_ge50 = float(per_tag[keep].mean()) if keep.any() else 0.0
    return {
        "subset_acc": float(subset_acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_tag": per_tag.tolist(),
        "per_tag_acc": per_tag_acc.tolist(),
        "macro_f1_ge50": macro_f1_ge50,
        "n_tags_ge50": int(keep.sum()),
    }


def tune_thresholds(Y_train, probs_train):
    """Per-tag threshold in THRESHOLD_GRID maximising that tag's F1 on the
    model's own training-set predictions (same in-sample approach as the audit).
    """
    n_tags = Y_train.shape[1]
    best = np.full(n_tags, THRESHOLD, dtype=float)
    for j in range(n_tags):
        y = Y_train[:, j]
        best_f1, best_t = -1.0, THRESHOLD
        for t in THRESHOLD_GRID:
            f1 = f1_score(y, (probs_train[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        best[j] = best_t
    return best


def fmt_row(vals):
    return " | ".join(f"{v:.3f}" for v in vals)


# ----------------------------------------------------------------------------
# One full mode run
# ----------------------------------------------------------------------------
def run_mode(mode, n_trials, outdir, Y, masks, counts, df, labels="strict",
             thresholds="fixed"):
    """One pooling-mode run (English dev+devtest -> test, plus cross-lingual transfer).

    Reports BOTH fixed-0.5 and per-tag tuned-threshold metrics. Thresholds are
    tuned in-sample on each model's own training rows: dev+devtest for the
    English-test models, and all 2009 English rows for the transfer models.
    """
    t_start = time.time()
    dev_devtest_mask = masks["dev_devtest"]
    test_mask = masks["test"]
    all_mask = np.ones(len(Y), dtype=bool)
    X = load_mode_embeddings(mode)
    suffix = "_lenient" if labels == "lenient" else ""
    pos = counts["all"]

    def sc(Yte, probs, ths=None):
        return score(Yte, probs, ths, pos_counts=pos)

    log(f"\n========== MODE: {mode} (labels={labels}, thresholds={thresholds}) ==========")

    # 1. BASELINE (default params, train dev+devtest -> test)
    log(f"[{mode}] [1/5] Baseline (max_depth=6, lr=0.1, n_estimators=300) ...")
    t = time.time()
    tp_base, (probs_base,), _ = fit_predict_with_train(
        BASELINE_PARAMS,
        X["en"][dev_devtest_mask], Y[dev_devtest_mask],
        [X["en"][test_mask]],
    )
    base_fixed = sc(Y[test_mask], probs_base)
    base_th = tune_thresholds(Y[dev_devtest_mask], tp_base)
    base_tuned = sc(Y[test_mask], probs_base, base_th)
    base_subset, base_macro = base_fixed["subset_acc"], base_fixed["macro_f1"]
    base_per_tag = np.asarray(base_fixed["per_tag"])
    log(f"[{mode}]   baseline test: fixed subset={base_fixed['subset_acc']:.4f} "
        f"macro={base_fixed['macro_f1']:.4f} micro={base_fixed['micro_f1']:.4f} | "
        f"tuned subset={base_tuned['subset_acc']:.4f} macro={base_tuned['macro_f1']:.4f} "
        f"micro={base_tuned['micro_f1']:.4f} ({time.time() - t:.1f}s)")

    # 2. OPTUNA (TPE seed 0, objective = test macro-F1 at fixed 0.5)
    log(f"[{mode}] [2/5] Optuna TPE ({n_trials} trials, objective=test macro-F1, seed=0) ...")

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )
        (probs,), _ = fit_predict(
            params,
            X["en"][dev_devtest_mask], Y[dev_devtest_mask],
            [X["en"][test_mask]],
        )
        macro = sc(Y[test_mask], probs)["macro_f1"]
        trial.set_user_attr("macro_f1", macro)
        return macro

    trials_csv = os.path.join(outdir, f"xgboost_optuna_trials{suffix}_{mode}.csv")

    def save_trials(study, _trial=None):
        rows = []
        for tr in study.trials:
            if tr.value is None:
                continue
            row = {"trial": tr.number, "value_macro_f1": tr.value, "state": tr.state.name}
            row.update(tr.params)
            rows.append(row)
        pd.DataFrame(rows).sort_values("trial").to_csv(trials_csv, index=False)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    t = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[save_trials])
    log(f"[{mode}]   optuna done in {time.time() - t:.1f}s")
    log(f"[{mode}]   BEST trial #{study.best_trial.number}: macro_f1={study.best_value:.4f}")
    log(f"[{mode}]   BEST params: {study.best_params}")
    best_params = dict(study.best_params)

    # 3. RETRAIN / EVALUATE
    log(f"[{mode}] [3/5] Retraining with best params ...")
    t = time.time()
    # (a) dev+devtest-trained models -> test, thresholds tuned on dev+devtest.
    tp_test, (probs_test,), _ = fit_predict_with_train(
        best_params,
        X["en"][dev_devtest_mask], Y[dev_devtest_mask],
        [X["en"][test_mask]],
    )
    tuned_fixed = sc(Y[test_mask], probs_test)
    tuned_th = tune_thresholds(Y[dev_devtest_mask], tp_test)
    tuned_tuned = sc(Y[test_mask], probs_test, tuned_th)
    tuned_subset, tuned_macro = tuned_fixed["subset_acc"], tuned_fixed["macro_f1"]
    tuned_per_tag = np.asarray(tuned_fixed["per_tag"])
    # (b) final models on ALL 2009 English rows for transfer; thresholds tuned on all 2009.
    models_full = []
    train_probs_full = np.zeros((len(Y), len(TAGS)))
    for j in range(len(TAGS)):
        model = xgb.XGBClassifier(**make_params(best_params))
        model.fit(X["en"][all_mask], Y[all_mask][:, j])
        models_full.append(model)
        train_probs_full[:, j] = model.predict_proba(X["en"][all_mask])[:, 1]
    transfer_th = tune_thresholds(Y[all_mask], train_probs_full)
    log(f"[{mode}]   retrain + transfer-model fit done ({time.time() - t:.1f}s)")
    log(f"[{mode}]   English-test tuned thresholds: {dict(zip(TAGS, tuned_th.tolist()))}")
    log(f"[{mode}]   transfer (all-2009) tuned thresholds: {dict(zip(TAGS, transfer_th.tolist()))}")

    # 4. CROSS-LINGUAL EVAL
    log(f"[{mode}] [4/5] Cross-lingual transfer evaluation ...")
    shifts = {l: (X["en"] - X[l]).mean(axis=0) for l in LANGS}  # D_L = mean(emb_en - emb_L)
    assert np.allclose(shifts["en"], 0.0)

    transfer_rows = []        # fixed-0.5 legacy tuples (name, subset, macro, per_tag)
    transfer_fixed_rows = []  # fixed-0.5 full score dicts
    transfer_tuned_rows = []  # tuned-threshold full score dicts

    def eval_full(mask_label, Xtest):
        probs = np.zeros((Xtest.shape[0], len(TAGS)))
        for j, model in enumerate(models_full):
            probs[:, j] = model.predict_proba(Xtest)[:, 1]
        sf = sc(Y, probs)
        st = sc(Y, probs, transfer_th)
        transfer_rows.append((mask_label, sf["subset_acc"], sf["macro_f1"],
                              np.asarray(sf["per_tag"])))
        transfer_fixed_rows.append((mask_label, sf))
        transfer_tuned_rows.append((mask_label, st))

    eval_full("en (in-sample)", X["en"])
    for l in ["it", "de", "fr"]:
        eval_full(f"{l} raw", X[l])
        eval_full(f"{l}+D", X[l] + shifts[l])

    for (name, subset, macro, _pt), (_, st) in zip(transfer_rows, transfer_tuned_rows):
        log(f"[{mode}]   {name:<14} fixed subset={subset:.4f} macro={macro:.4f} | "
            f"tuned subset={st['subset_acc']:.4f} macro={st['macro_f1']:.4f} "
            f"micro={st['micro_f1']:.4f}")

    # 5. WRITE PER-MODE RESULTS
    log(f"[{mode}] [5/5] Writing per-mode results ...")
    runtime = time.time() - t_start
    res = dict(
        mode=mode, n_trials=n_trials, labels=labels, thresholds_mode=thresholds,
        suffix=suffix,
        base_subset=base_subset, base_macro=base_macro, base_per_tag=base_per_tag.tolist(),
        tuned_subset=tuned_subset, tuned_macro=tuned_macro, tuned_per_tag=tuned_per_tag.tolist(),
        best_trial=study.best_trial.number, best_value=study.best_value,
        best_params=best_params,
        base_fixed=base_fixed, base_tuned=base_tuned,
        base_thresholds=base_th.tolist(),
        tuned_fixed=tuned_fixed, tuned_tuned=tuned_tuned,
        tuned_thresholds=tuned_th.tolist(),
        transfer_thresholds=transfer_th.tolist(),
        transfer=[(n, s, m, p.tolist()) for (n, s, m, p) in transfer_rows],
        transfer_fixed=[(n, s["subset_acc"], s["macro_f1"], s["micro_f1"], s["per_tag"])
                        for n, s in transfer_fixed_rows],
        transfer_tuned=[(n, s["subset_acc"], s["macro_f1"], s["micro_f1"], s["per_tag"])
                        for n, s in transfer_tuned_rows],
        runtime=runtime,
    )
    if suffix:
        write_mode_md_lenient(res, outdir, Y, masks, counts, df)
    else:
        write_mode_md(res, outdir, Y, masks, counts, df)
    log(f"[{mode}] done in {runtime:.1f}s -> "
        f"{os.path.join(outdir, f'xgboost_topic_results{suffix}_{mode}.md')}")
    return res


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def write_mode_md(res, outdir, Y, masks, counts, df):
    mode = res["mode"]
    dev_mask, devtest_mask, test_mask = masks["dev"], masks["devtest"], masks["test"]
    counts_all, counts_dev, counts_devtest, counts_test = (
        counts["all"], counts["dev"], counts["devtest"], counts["test"])

    lines = []
    lines.append(f"# XGBoost topic classifier on FLORES ({mode}) embeddings\n")
    lines.append(f"Generated by `xgboost_topic_optuna.py --mode {mode}` — runtime "
                 f"{res['runtime']:.1f}s, xgboost {xgb.__version__}, optuna {optuna.__version__}, "
                 f"`tree_method=hist`, `max_bin={MAX_BIN}`, threads=2.\n")
    lines.append("Task: multi-label classification over the 8 canonical tags "
                 "{travel, sports, crime and law, politics, health, geography, science, safety}. "
                 "`topic` is split on commas, lowercased and stripped; only exact canonical-tag "
                 "matches count (the raw label space contains ~250 finer tags, e.g. `sport`, "
                 "`crime`, `law`, `science and technology`, most of which do not map to the 8).\n")

    lines.append(f"Splits: `dev` = {int(dev_mask.sum())} rows (original dev), `devtest` = "
                 f"{int(devtest_mask.sum())} rows and `test` = {int(test_mask.sum())} rows from a "
                 "seeded shuffle (`np.random.default_rng(0)`) of the original 1012 devtest rows "
                 "(first 506 / last 506); every canonical tag is present in both evaluation splits. "
                 "Baseline and every Optuna trial train on dev+devtest (1503 rows) and score macro-F1 "
                 "on `test` (506 rows); the best params are then retrained on all 2009 English rows "
                 "for the cross-lingual transfer.\n")
    lines.append("## Label positive counts\n")
    lines.append(f"| tag | all 2009 | dev {int(dev_mask.sum())} | devtest {int(devtest_mask.sum())} "
                 f"| test {int(test_mask.sum())} |")
    lines.append("|---|---:|---:|---:|---:|")
    for j, tag in enumerate(TAGS):
        lines.append(f"| {tag} | {counts_all[j]} | {counts_dev[j]} | {counts_devtest[j]} "
                     f"| {counts_test[j]} |")
    lines.append(f"\nRows with missing `topic`: {int(df['topic'].isna().sum())} "
                 f"(treated as all-zero target).\n")

    lines.append(f"## English test (train dev+devtest=1503, n={int(test_mask.sum())})\n")
    lines.append("| metric | baseline | tuned (Optuna) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| subset accuracy | {res['base_subset']:.4f} | {res['tuned_subset']:.4f} |")
    lines.append(f"| macro-F1 | {res['base_macro']:.4f} | {res['tuned_macro']:.4f} |")
    lines.append(f"\nPer-tag F1 (baseline / tuned):\n")
    lines.append("| tag | baseline | tuned |")
    lines.append("|---|---:|---:|")
    for j, tag in enumerate(TAGS):
        lines.append(f"| {tag} | {res['base_per_tag'][j]:.3f} | {res['tuned_per_tag'][j]:.3f} |")
    lines.append("")

    lines.append("\n### Best Optuna parameters (shared across all 8 labels)\n")
    lines.append("```")
    lines.append(f"best trial #: {res['best_trial']}")
    lines.append(f"best value (test macro-F1): {res['best_value']:.4f}")
    for k, v in res["best_params"].items():
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append(f"\nPer-trial log: `xgboost_optuna_trials_{mode}.csv` ({res['n_trials']} trials). "
                 "Hyperparameters were selected by maximizing macro-F1 on the English test set "
                 "per protocol (no cross-validation).\n")

    lines.append("## Cross-lingual transfer (train on all 2009 English rows)\n")
    lines.append("`D_L = mean_over_2009(emb_en - emb_L)`; shifted features are `emb_L + D_L` "
                 "(`D_en = 0`, so `en` shifted == `en`). All 2009 rows of each language are "
                 "evaluated with the same tags.\n")
    lines.append("| test set | subset acc | macro-F1 | " + " | ".join(TAGS) + " |")
    lines.append("|---|---:|---:|" + "---:|" * len(TAGS))
    for name, subset, macro, per_tag in res["transfer"]:
        lines.append(f"| {name} | {subset:.4f} | {macro:.4f} | {fmt_row(per_tag)} |")
    lines.append("")

    tr = {r[0]: (r[1], r[2], r[3]) for r in res["transfer"]}
    raw = {l: tr[f"{l} raw"] for l in ["it", "de", "fr"]}
    sh = {l: tr[f"{l}+D"] for l in ["it", "de", "fr"]}
    best_tag_idx = max(range(len(TAGS)),
                       key=lambda j: raw["it"][2][j] + raw["de"][2][j] + raw["fr"][2][j])
    worst_tag_idx = min(range(len(TAGS)),
                        key=lambda j: raw["it"][2][j] + raw["de"][2][j] + raw["fr"][2][j])
    avg_raw = np.mean([raw[l][1] for l in raw])
    avg_sh = np.mean([sh[l][1] for l in sh])
    delta = {l: sh[l][1] - raw[l][1] for l in raw}

    lines.append("## Notes\n")
    lines.append(f"- **Tag transfer:** across the three target languages the strongest raw "
                 f"macro-F1 tag is `{TAGS[best_tag_idx]}`; the weakest is `{TAGS[worst_tag_idx]}`. "
                 "Rare tags (e.g. geography, safety, science) have very few positives and near-zero "
                 "F1, dominating the macro average.")
    lines.append(f"- **Raw vs shifted (macro-F1):** mean raw = {avg_raw:.4f} vs mean shifted = "
                 f"{avg_sh:.4f}. Per language the shift changes macro-F1 by "
                 + ", ".join(f"{l}: {delta[l]:+.4f}" for l in ["it", "de", "fr"]) +
                 ". A per-row mean shift can only translate the cloud, not fix task-relevant "
                 "geometry, so its effect is small and not consistently positive.")
    lines.append(f"- **Per-language:** raw macro-F1 is " +
                 ", ".join(f"{l} {raw[l][1]:.4f}" for l in ["it", "de", "fr"]) +
                 "; subset accuracy is " +
                 ", ".join(f"{l} {raw[l][0]:.4f}" for l in ["it", "de", "fr"]) + ".")
    lines.append(f"- **In-sample English** scores ({tr['en (in-sample)'][0]:.4f} subset / "
                 f"{tr['en (in-sample)'][1]:.4f} macro-F1) are much higher than every cross-lingual "
                 "number because the models were trained on those exact rows; they are not a fair "
                 "transfer reference.")
    lines.append("- **Selection on test:** the seeded shuffle places every canonical tag in both "
                 "evaluation splits (all per-tag counts > 0; see table), so macro-F1 is not dominated "
                 "by absent classes. Per protocol, however, the Optuna objective and the reported "
                 "baseline/tuned scores are the same `test` set, so the tuned number is a "
                 "selection-set score (there is no separate held-out English estimate).")
    lines.append("- **Rare tags:** several canonical tags have <45 positives overall "
                 "(health 44, safety 40, science 39, geography 34), so per-tag F1 is noisy and "
                 "macro-F1 is unstable; exact-match subset accuracy is dominated by the all-zero "
                 "rows, which is why it is low.")

    with open(os.path.join(outdir, f"xgboost_topic_results_{mode}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_mode_md_lenient(res, outdir, Y, masks, counts, df):
    """Per-mode write-up for the lenient-label / tuned-threshold run."""
    mode = res["mode"]
    suffix = res.get("suffix", "_lenient")
    dev_mask, devtest_mask, test_mask = masks["dev"], masks["devtest"], masks["test"]
    counts_all, counts_dev, counts_devtest, counts_test = (
        counts["all"], counts["dev"], counts["devtest"], counts["test"])

    def mrow(label, s):
        return (f"| {label} | {s['subset_acc']:.4f} | {s['macro_f1']:.4f} | "
                f"{s['micro_f1']:.4f} | {s['macro_f1_ge50']:.4f} |")

    lines = []
    lines.append(f"# XGBoost topic classifier on FLORES ({mode}) embeddings — lenient labels\n")
    lines.append(f"Generated by `xgboost_topic_optuna.py --mode {mode} --labels lenient "
                 f"--thresholds {res.get('thresholds_mode', 'tuned')}` — runtime "
                 f"{res['runtime']:.1f}s, xgboost {xgb.__version__}, optuna {optuna.__version__}, "
                 f"`tree_method=hist`, `max_bin={MAX_BIN}`, threads=2.\n")
    lines.append("**Label scheme: `lenient`** — raw comma-tags mapped onto the 8 canonical tags via "
                 "the curated aliases in `the label audit` §1 (see `LENIENT_EXACT/PREFIX/CONTAINS`); "
                 "all-zero rows drop from 1147/2009 (strict) to ~504/2009.\n")
    lines.append("**Thresholds:** per-tag, tuned in-sample on each model's own training rows "
                 "(dev+devtest for the English-test models; all 2009 English rows for the transfer "
                 "models), grid 0.05..0.95 step 0.05. Both fixed-0.5 and tuned metrics are reported. "
                 "The Optuna search is unchanged (objective = test macro-F1 at fixed 0.5).\n")
    lines.append(f"Splits: `dev`={int(dev_mask.sum())}, `devtest`={int(devtest_mask.sum())}, "
                 f"`test`={int(test_mask.sum())}. Macro-F1 is over all 8 tags; the "
                 f">= {MIN_POS_FOR_MACRO}-positive macro is also shown.\n")

    lines.append("## Label positive counts (lenient)\n")
    lines.append(f"| tag | all 2009 | dev | devtest | test |")
    lines.append("|---|---:|---:|---:|---:|")
    for j, tag in enumerate(TAGS):
        lines.append(f"| {tag} | {counts_all[j]} | {counts_dev[j]} | {counts_devtest[j]} "
                     f"| {counts_test[j]} |")
    lines.append(f"\nAll-zero rows: {int((Y.sum(axis=1) == 0).sum())}/2009.\n")

    lines.append(f"## English test (train dev+devtest=1503, n={int(test_mask.sum())})\n")
    lines.append("| model | subset acc | macro-F1 | micro-F1 | macro-F1 (>=50) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(mrow("baseline @0.5", res["base_fixed"]))
    lines.append(mrow("baseline tuned", res["base_tuned"]))
    lines.append(mrow("best params @0.5", res["tuned_fixed"]))
    lines.append(mrow("best params tuned", res["tuned_tuned"]))
    lines.append("")

    lines.append("### Per-tag F1 (English test)\n")
    lines.append("| tag | baseline @0.5 | baseline tuned | best @0.5 | best tuned |")
    lines.append("|---|---:|---:|---:|---:|")
    for j, tag in enumerate(TAGS):
        lines.append(f"| {tag} | {res['base_fixed']['per_tag'][j]:.3f} | "
                     f"{res['base_tuned']['per_tag'][j]:.3f} | "
                     f"{res['tuned_fixed']['per_tag'][j]:.3f} | "
                     f"{res['tuned_tuned']['per_tag'][j]:.3f} |")
    lines.append("")

    lines.append("### Per-tag tuned thresholds\n")
    lines.append("| tag | dev+devtest (English test) | all-2009 English (transfer) |")
    lines.append("|---|---:|---:|")
    for j, tag in enumerate(TAGS):
        lines.append(f"| {tag} | {res['tuned_thresholds'][j]:.2f} | "
                     f"{res['transfer_thresholds'][j]:.2f} |")
    lines.append("")

    lines.append("\n### Best Optuna parameters (shared across all 8 labels)\n")
    lines.append("```")
    lines.append(f"best trial #: {res['best_trial']}")
    lines.append(f"best value (test macro-F1 @0.5): {res['best_value']:.4f}")
    for k, v in res["best_params"].items():
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append(f"\nPer-trial log: `xgboost_optuna_trials{suffix}_{mode}.csv` "
                 f"({res['n_trials']} trials).\n")

    lines.append("## Cross-lingual transfer (train on all 2009 English rows)\n")
    lines.append("`D_L = mean_over_2009(emb_en - emb_L)`; shifted features are `emb_L + D_L`. "
                 "Thresholds for this section are tuned in-sample on the all-2009 English rows.\n")
    lines.append("### Fixed 0.5\n")
    lines.append("| test set | subset acc | macro-F1 | micro-F1 | " + " | ".join(TAGS) + " |")
    lines.append("|---|---:|---:|---:|" + "---:|" * len(TAGS))
    for name, subset, macro, micro, per_tag in res["transfer_fixed"]:
        lines.append(f"| {name} | {subset:.4f} | {macro:.4f} | {micro:.4f} | {fmt_row(per_tag)} |")
    lines.append("")
    lines.append("### Tuned thresholds\n")
    lines.append("| test set | subset acc | macro-F1 | micro-F1 | " + " | ".join(TAGS) + " |")
    lines.append("|---|---:|---:|---:|" + "---:|" * len(TAGS))
    for name, subset, macro, micro, per_tag in res["transfer_tuned"]:
        lines.append(f"| {name} | {subset:.4f} | {macro:.4f} | {micro:.4f} | {fmt_row(per_tag)} |")
    lines.append("")

    tr = {r[0]: r for r in res["transfer_tuned"]}
    raw = {l: tr[f"{l} raw"] for l in ["it", "de", "fr"]}
    sh = {l: tr[f"{l}+D"] for l in ["it", "de", "fr"]}
    lines.append("## Notes\n")
    lines.append(f"- **Tuned vs fixed (English test, best params):** macro-F1 "
                 f"{res['tuned_fixed']['macro_f1']:.4f} -> {res['tuned_tuned']['macro_f1']:.4f}, "
                 f"micro-F1 {res['tuned_fixed']['micro_f1']:.4f} -> "
                 f"{res['tuned_tuned']['micro_f1']:.4f}, subset acc "
                 f"{res['tuned_fixed']['subset_acc']:.4f} -> {res['tuned_tuned']['subset_acc']:.4f}. "
                 "Per-tag thresholds trade subset accuracy for rare-tag recall that macro-F1 rewards.")
    lines.append(f"- **Cross-lingual (tuned):** mean raw macro-F1 over it/de/fr = "
                 f"{np.mean([raw[l][2] for l in raw]):.4f}, mean shifted = "
                 f"{np.mean([sh[l][2] for l in sh]):.4f}.")
    lines.append(f"- **In-sample English transfer** (trained on the same 2009 rows) scores "
                 f"{tr['en (in-sample)'][1]:.4f} subset / {tr['en (in-sample)'][2]:.4f} macro-F1; "
                 "it is an upper bound, not a fair transfer reference.")
    lines.append(f"- **Selection:** Optuna still selects on `test` at fixed 0.5, so the English tuned "
                 "numbers are selection-set scores; the per-tag thresholds are tuned in-sample on the "
                 "training split (no held-out threshold selection).")
    lines.append(f"- **Reproducibility:** `xgboost_topic_results{suffix}_{mode}.md`, "
                 f"`xgboost_optuna_trials{suffix}_{mode}.csv`.")

    path = os.path.join(outdir, f"xgboost_topic_results{suffix}_{mode}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_allmodes_md_lenient(results, outdir, elapsed, suffix="_lenient"):
    """Combined pooling-mode comparison for the lenient/tuned transfer run."""
    modes = [m for m in MODES if m in results]
    lines = []
    lines.append("# XGBoost topic classifier — pooling-mode comparison (lenient labels)\n")
    lines.append(f"Generated by `xgboost_topic_optuna.py --mode all --labels lenient` — total runtime "
                 f"{elapsed:.1f}s, xgboost {xgb.__version__}, optuna {optuna.__version__}, "
                 f"`tree_method=hist`, `max_bin={MAX_BIN}`, threads=2.\n")
    lines.append("Lenient labels (`the label audit` §1) + per-tag tuned thresholds (in-sample on "
                 "each model's training rows). Optuna search unchanged (test macro-F1 at fixed 0.5). "
                 "Both fixed-0.5 and tuned metrics are shown.\n")

    lines.append("## English test (train dev+devtest=1503, n=506)\n")
    lines.append("| mode | base macro@0.5 | base macro tuned | best macro@0.5 | best macro tuned "
                 "| best micro tuned | best subset tuned | best params |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for m in modes:
        r = results[m]
        lines.append(f"| {m} | {r['base_fixed']['macro_f1']:.4f} | "
                     f"{r['base_tuned']['macro_f1']:.4f} | {r['tuned_fixed']['macro_f1']:.4f} | "
                     f"{r['tuned_tuned']['macro_f1']:.4f} | {r['tuned_tuned']['micro_f1']:.4f} | "
                     f"{r['tuned_tuned']['subset_acc']:.4f} | {params_summary(r['best_params'])} |")
    lines.append("")

    order = ["en (in-sample)", "it raw", "it+D", "de raw", "de+D", "fr raw", "fr+D"]
    for title, key, midx in [("Transfer (tuned) — subset accuracy", "transfer_tuned", 1),
                             ("Transfer (tuned) — macro-F1", "transfer_tuned", 2),
                             ("Transfer (tuned) — micro-F1", "transfer_tuned", 3)]:
        lines.append(f"## {title}\n")
        lines.append("| mode | en | it raw | it+D | de raw | de+D | fr raw | fr+D |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in modes:
            tr = {r[0]: r for r in results[m][key]}
            lines.append(f"| {m} | " + " | ".join(f"{tr[k][midx]:.4f}" for k in order) + " |")
        lines.append("")

    lines.append("## Notes\n")
    best_eng = max(modes, key=lambda m: results[m]["tuned_tuned"]["macro_f1"])
    best_micro = max(modes, key=lambda m: results[m]["tuned_tuned"]["micro_f1"])
    best_tr = max(modes, key=lambda m: np.mean(
        [r[2] for r in results[m]["transfer_tuned"] if r[0] in ("it raw", "de raw", "fr raw")]))
    lines.append(f"- **English tuned macro-F1:** best mode `{best_eng}` "
                 f"({results[best_eng]['tuned_tuned']['macro_f1']:.4f}); ranking " +
                 ", ".join(f"{m} {results[m]['tuned_tuned']['macro_f1']:.4f}" for m in modes) + ".")
    lines.append(f"- **English tuned micro-F1:** best mode `{best_micro}` "
                 f"({results[best_micro]['tuned_tuned']['micro_f1']:.4f}).")
    lines.append(f"- **Cross-lingual raw transfer (mean it/de/fr tuned macro-F1):** best mode "
                 f"`{best_tr}`.")
    lines.append(f"- **Reproducibility:** `xgboost_topic_allmodes{suffix}.md`, per-mode "
                 + ", ".join(f"`xgboost_topic_results{suffix}_{m}.md`" for m in modes) + ".")

    path = os.path.join(outdir, f"xgboost_topic_allmodes{suffix}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def params_summary(p):
    return (f"d{p['max_depth']} lr{p['learning_rate']:.3f} n{p['n_estimators']} "
            f"cs{p['colsample_bytree']:.3f} mcw{p['min_child_weight']} "
            f"l2={p['reg_lambda']:.3f} g{p['gamma']:.2f}")


def write_allmodes_md(results, outdir, elapsed):
    modes = [m for m in MODES if m in results]
    lines = []
    lines.append("# XGBoost topic classifier — pooling-mode comparison\n")
    lines.append(f"Generated by `xgboost_topic_optuna.py --mode all` — total runtime "
                 f"{elapsed:.1f}s, xgboost {xgb.__version__}, optuna {optuna.__version__}, "
                 f"`tree_method=hist`, `max_bin={MAX_BIN}`, threads=2.\n")
    lines.append("Identical protocol for every pooling mode: baseline (default params) and 50 "
                 "Optuna trials (TPE seed 0, same search space) train on dev+devtest (1503 English "
                 "rows) and score macro-F1 on `test` (506 English rows); best params are retrained "
                 "on all 2009 English rows for the cross-lingual transfer. Splits are the seed-0 "
                 "shuffled halves of the original devtest (shared across modes).\n")

    lines.append("## English test (train dev+devtest=1503, n=506)\n")
    lines.append("| mode | base subset | base macro-F1 | tuned subset | tuned macro-F1 | best params |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for m in modes:
        r = results[m]
        lines.append(f"| {m} | {r['base_subset']:.4f} | {r['base_macro']:.4f} | "
                     f"{r['tuned_subset']:.4f} | {r['tuned_macro']:.4f} | {params_summary(r['best_params'])} |")
    lines.append("")

    def transfer_matrix(metric_idx, header):
        out = [f"## {header}\n"]
        out.append("| mode | en | it raw | it+D | de raw | de+D | fr raw | fr+D |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in modes:
            tr = {r[0]: r for r in results[m]["transfer"]}
            order = ["en (in-sample)", "it raw", "it+D", "de raw", "de+D", "fr raw", "fr+D"]
            vals = " | ".join(f"{tr[k][metric_idx]:.4f}" for k in order)
            out.append(f"| {m} | {vals} |")
        out.append("")
        return out

    lines.extend(transfer_matrix(1, "Transfer — subset accuracy"))
    lines.extend(transfer_matrix(2, "Transfer — macro-F1"))

    # ---- bullets ----
    def raw_macro(m):
        tr = {r[0]: r for r in results[m]["transfer"]}
        return {l: tr[f"{l} raw"][2] for l in ["it", "de", "fr"]}

    def shifted_macro(m):
        tr = {r[0]: r for r in results[m]["transfer"]}
        return {l: tr[f"{l}+D"][2] for l in ["it", "de", "fr"]}

    best_eng = max(modes, key=lambda m: results[m]["tuned_macro"])
    best_transfer = max(modes, key=lambda m: np.mean(list(raw_macro(m).values())))
    shift_delta = {m: np.mean([shifted_macro(m)[l] - raw_macro(m)[l] for l in ["it", "de", "fr"]])
                   for m in modes}
    best_shift = max(modes, key=lambda m: shift_delta[m])

    lines.append("## Notes\n")
    lines.append(f"- **English (tuned test macro-F1):** best mode is `{best_eng}` "
                 f"({results[best_eng]['tuned_macro']:.4f}); full ranking " +
                 ", ".join(f"{m} {results[m]['tuned_macro']:.4f}" for m in modes) + ".")
    lines.append(f"- **Cross-lingual raw transfer (mean of it/de/fr raw macro-F1):** best mode is "
                 f"`{best_transfer}` (" +
                 ", ".join(f"{m} {np.mean(list(raw_macro(m).values())):.4f}" for m in modes) + ").")
    lines.append(f"- **Shift effect (mean delta macro-F1, +D minus raw over it/de/fr):** " +
                 ", ".join(f"{m} {shift_delta[m]:+.4f}" for m in modes) +
                 f"; most positive is `{best_shift}`. The shift helps some modes/languages but is "
                 "not uniformly positive, consistent with a mean vector only translating the cloud.")
    lines.append("- **Raw f1 is dominated by `travel`** (and to a lesser degree `sports` and "
                 "`politics` in French); rare tags (geography, science, safety, health) are near "
                 "zero across all modes, so macro-F1 differences between modes are driven by the "
                 "high-count tags.")
    lines.append("- **Caveats:** Optuna selects on the same `test` set it reports (selection-set "
                 "scores), and the rare tags have very few positives, so per-mode macro-F1 is noisy.")
    lines.append(f"- **Reproducibility:** each mode uses the identical split and search space; "
                 "only the pooling features differ. Per-mode details: "
                 + ", ".join(f"`xgboost_topic_results_{m}.md`" for m in modes) + ".")

    with open(os.path.join(outdir, "xgboost_topic_allmodes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Per-language grid (--mode perlang)
# ----------------------------------------------------------------------------
def load_one_embedding(mode, lang):
    X = np.load(os.path.join(BASE, "embeddings", mode, f"emb_{lang}.npy"))
    assert X.shape == (2009, 768) and X.dtype == np.float32, (mode, lang, X.shape)
    return X


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_perlang_status(status, lang, outdir, prefix="xgboost_perlang"):
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(os.path.join(outdir, f"{prefix}_{lang}_status.json"), status)


def _optuna_search_space(trial):
    """Identical search space to the existing --mode runs (kept in one place)."""
    return dict(
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )


def run_perlang_mode(lang, mode, n_trials, outdir, Y, masks, labels="strict",
                     thresholds="fixed", prefix="xgboost_perlang", pos_counts=None):
    """One (language, pooling) cell of the per-language grid.

    Baseline (default params) and best-params fits report BOTH fixed-0.5 and
    per-tag tuned-threshold metrics on devtest and test. The Optuna search is
    unchanged (objective = devtest macro-F1 at fixed 0.5).
    """
    dev_mask = masks["dev"]
    devtest_mask = masks["devtest"]
    test_mask = masks["test"]
    dev_devtest_mask = masks["dev_devtest"]
    X = load_one_embedding(mode, lang)
    if pos_counts is None:
        pos_counts = Y.sum(axis=0)

    def sc(Yte, probs, ths=None):
        return score(Yte, probs, ths, pos_counts=pos_counts)

    timings = {}
    log(f"\n===== PERLANG lang={lang} mode={mode} labels={labels} thresholds={thresholds} =====")

    # 1. BASELINE (default params) -----------------------------------------
    # (a) train dev (997) -> eval devtest (506)
    log(f"[{lang}/{mode}] baseline: train dev -> eval devtest ...")
    t = time.time()
    tp_dt, (probs_dt,), _ = fit_predict_with_train(
        BASELINE_PARAMS, X[dev_mask], Y[dev_mask], [X[devtest_mask]])
    b_dt_fixed = sc(Y[devtest_mask], probs_dt)
    b_dt_th = tune_thresholds(Y[dev_mask], tp_dt)
    b_dt_tuned = sc(Y[devtest_mask], probs_dt, b_dt_th)
    timings["baseline_devtest_s"] = time.time() - t
    # (b) train dev+devtest (1503) -> eval test (506)
    log(f"[{lang}/{mode}] baseline: train dev+devtest -> eval test ...")
    t = time.time()
    tp_test, (probs_test,), _ = fit_predict_with_train(
        BASELINE_PARAMS, X[dev_devtest_mask], Y[dev_devtest_mask], [X[test_mask]])
    b_test_fixed = sc(Y[test_mask], probs_test)
    b_test_th = tune_thresholds(Y[dev_devtest_mask], tp_test)
    b_test_tuned = sc(Y[test_mask], probs_test, b_test_th)
    timings["baseline_test_s"] = time.time() - t
    log(f"[{lang}/{mode}]   baseline devtest macro fixed={b_dt_fixed['macro_f1']:.4f} "
        f"tuned={b_dt_tuned['macro_f1']:.4f} micro fixed={b_dt_fixed['micro_f1']:.4f} "
        f"tuned={b_dt_tuned['micro_f1']:.4f} | test macro fixed={b_test_fixed['macro_f1']:.4f} "
        f"tuned={b_test_tuned['macro_f1']:.4f}")

    # 2. OPTUNA (train dev, objective = devtest macro-F1 at fixed 0.5) -----
    log(f"[{lang}/{mode}] optuna TPE {n_trials} trials (train dev, objective=devtest macro-F1, seed=0) ...")

    def objective(trial):
        params = _optuna_search_space(trial)
        (probs,), _ = fit_predict(params, X[dev_mask], Y[dev_mask], [X[devtest_mask]])
        macro = sc(Y[devtest_mask], probs)["macro_f1"]
        trial.set_user_attr("macro_f1", macro)
        return macro

    trials_csv = os.path.join(outdir, f"{prefix}_trials_{lang}_{mode}.csv")

    def save_trials(study, _trial=None):
        rows = []
        for tr in study.trials:
            if tr.value is None:
                continue
            row = {"trial": tr.number, "value_macro_f1": tr.value, "state": tr.state.name}
            row.update(tr.params)
            rows.append(row)
        pd.DataFrame(rows).sort_values("trial").to_csv(trials_csv, index=False)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    t = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[save_trials])
    timings["optuna_s"] = time.time() - t
    best_params = dict(study.best_params)
    log(f"[{lang}/{mode}]   optuna done in {timings['optuna_s']:.1f}s, best #{study.best_trial.number} "
        f"devtest macro={study.best_value:.4f}")
    log(f"[{lang}/{mode}]   best params: {best_params}")

    # 3. BEST PARAMS: retrain dev -> devtest and dev+devtest -> test -------
    log(f"[{lang}/{mode}] retrain best: dev -> devtest and dev+devtest -> test ...")
    t = time.time()
    tp_bdt, (probs_bdt,), _ = fit_predict_with_train(
        best_params, X[dev_mask], Y[dev_mask], [X[devtest_mask]])
    t_dt_fixed = sc(Y[devtest_mask], probs_bdt)
    t_dt_th = tune_thresholds(Y[dev_mask], tp_bdt)
    t_dt_tuned = sc(Y[devtest_mask], probs_bdt, t_dt_th)
    tp_btest, (probs_btest,), _ = fit_predict_with_train(
        best_params, X[dev_devtest_mask], Y[dev_devtest_mask], [X[test_mask]])
    t_test_fixed = sc(Y[test_mask], probs_btest)
    t_test_th = tune_thresholds(Y[dev_devtest_mask], tp_btest)
    t_test_tuned = sc(Y[test_mask], probs_btest, t_test_th)
    timings["retrain_s"] = time.time() - t
    t_total = sum(timings.values())
    timings["total_s"] = t_total
    log(f"[{lang}/{mode}]   tuned-params devtest macro fixed={t_dt_fixed['macro_f1']:.4f} "
        f"tuned={t_dt_tuned['macro_f1']:.4f} | test macro fixed={t_test_fixed['macro_f1']:.4f} "
        f"tuned={t_test_tuned['macro_f1']:.4f} ({t_total:.1f}s total)")
    log(f"[{lang}/{mode}]   devtest tuned thresholds: {dict(zip(TAGS, t_dt_th.tolist()))}")

    return {
        "lang": lang, "mode": mode, "n_trials": n_trials,
        "labels": labels, "thresholds_mode": thresholds,
        "baseline": {
            # legacy fixed-0.5 keys (strict merge / backwards compat)
            "devtest_subset": b_dt_fixed["subset_acc"], "devtest_macro": b_dt_fixed["macro_f1"],
            "devtest_per_tag": b_dt_fixed["per_tag"],
            "test_subset": b_test_fixed["subset_acc"], "test_macro": b_test_fixed["macro_f1"],
            "test_per_tag": b_test_fixed["per_tag"],
            # new: both fixed-0.5 and tuned-threshold metrics
            "fixed": {"devtest": b_dt_fixed, "test": b_test_fixed},
            "tuned": {"devtest": b_dt_tuned, "test": b_test_tuned,
                      "thresholds": {"devtest": b_dt_th.tolist(), "test": b_test_th.tolist()}},
        },
        "tuned": {
            "devtest_subset": t_dt_fixed["subset_acc"], "devtest_macro": t_dt_fixed["macro_f1"],
            "devtest_per_tag": t_dt_fixed["per_tag"],
            "test_subset": t_test_fixed["subset_acc"], "test_macro": t_test_fixed["macro_f1"],
            "test_per_tag": t_test_fixed["per_tag"],
            "best_trial": int(study.best_trial.number), "best_value": float(study.best_value),
            "best_params": best_params,
            "fixed": {"devtest": t_dt_fixed, "test": t_test_fixed},
            "tuned": {"devtest": t_dt_tuned, "test": t_test_tuned,
                      "thresholds": {"devtest": t_dt_th.tolist(), "test": t_test_th.tolist()}},
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }


def run_perlang(langs, modes, n_trials, outdir, labels="strict", thresholds="fixed"):
    prefix = "xgboost_perlang_lenient" if labels == "lenient" else "xgboost_perlang"
    # Write PIDs immediately so a cancelled worker's process can be killed.
    for lang in langs:
        with open(os.path.join(outdir, f"{prefix}_{lang}.pid"), "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    df, Y, masks, counts = load_data(labels)
    if labels == "lenient":
        _report_lenient_counts(Y)

    log("Label positive counts (all / dev / devtest / test):")
    for j, tag in enumerate(TAGS):
        log(f"  {tag:<14} {counts['all'][j]:>4} / {counts['dev'][j]:>4} / "
            f"{counts['devtest'][j]:>4} / {counts['test'][j]:>4}")

    for lang in langs:
        t_lang = time.time()
        status = {
            "pid": os.getpid(), "lang": lang, "n_trials": n_trials, "modes": list(modes),
            "labels": labels, "thresholds": thresholds,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modes_status": {m: {"state": "pending"} for m in modes},
        }
        write_perlang_status(status, lang, outdir, prefix)

        res_lang = {
            "lang": lang, "n_trials": n_trials, "labels": labels,
            "thresholds_mode": thresholds,
            "xgb_version": xgb.__version__, "optuna_version": optuna.__version__,
            "baseline_params": BASELINE_PARAMS, "modes": {},
        }
        for mode in modes:
            status["modes_status"][mode] = {"state": "running",
                                            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            write_perlang_status(status, lang, outdir, prefix)
            try:
                res_mode = run_perlang_mode(lang, mode, n_trials, outdir, Y, masks,
                                            labels=labels, thresholds=thresholds, prefix=prefix,
                                            pos_counts=counts["all"])
            except Exception as exc:
                status["modes_status"][mode] = {"state": "failed", "error": repr(exc)}
                write_perlang_status(status, lang, outdir, prefix)
                log(f"[perlang {lang}/{mode}] FAILED: {exc!r}")
                raise
            res_lang["modes"][mode] = res_mode
            status["modes_status"][mode] = {
                "state": "done",
                "base_devtest_macro": round(res_mode["baseline"]["devtest_macro"], 4),
                "tuned_devtest_macro": round(res_mode["tuned"]["devtest_macro"], 4),
                "tuned_test_macro": round(res_mode["tuned"]["test_macro"], 4),
                "tuned_threshold_devtest_macro": round(
                    res_mode["tuned"]["tuned"]["devtest"]["macro_f1"], 4),
                "tuned_threshold_test_macro": round(
                    res_mode["tuned"]["tuned"]["test"]["macro_f1"], 4),
                "devtest_thresholds": res_mode["tuned"]["tuned"]["thresholds"]["devtest"],
                "runtime_s": round(res_mode["timings"]["total_s"], 1),
            }
            write_perlang_status(status, lang, outdir, prefix)

        res_lang["runtime_s"] = round(time.time() - t_lang, 1)
        res_lang["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_json(os.path.join(outdir, f"{prefix}_{lang}.json"), res_lang)
        status["finished"] = res_lang["finished"]
        status["elapsed_s"] = res_lang["runtime_s"]
        write_perlang_status(status, lang, outdir, prefix)
        log(f"[perlang {lang}] wrote {prefix}_{lang}.json ({res_lang['runtime_s']}s)")

    return langs


# ----------------------------------------------------------------------------
# Merge per-language grid (--mode merge)
# ----------------------------------------------------------------------------
def write_perlang_md(data, outdir):
    langs = [l for l in LANGS if l in data]
    modes = [m for m in MODES if any(m in data[l]["modes"] for l in langs)]

    def cell_tuned(l, m):
        t = data[l]["modes"][m]["tuned"]
        return f"{t['devtest_macro']:.3f} → {t['test_macro']:.3f}"

    def cell_tuned_acc(l, m):
        t = data[l]["modes"][m]["tuned"]
        return f"{t['devtest_subset']:.3f} → {t['test_subset']:.3f}"

    def cell_base(l, m):
        b = data[l]["modes"][m]["baseline"]
        return f"{b['devtest_subset']:.3f} → {b['test_subset']:.3f}"

    lines = []
    lines.append("# XGBoost topic classifier — per-language × pooling grid\n")
    lines.append(f"Generated by `xgboost_topic_optuna.py --mode merge` — xgboost {xgb.__version__}, "
                 f"optuna {optuna.__version__}, `tree_method=hist`, `max_bin={MAX_BIN}`, 2 threads per "
                 "worker. No cross-lingual transfer (each language trains and tests on itself).\n")
    lines.append("Protocol: for each (language, pooling) the baseline (default params) and every one "
                 "of the Optuna trials (TPE seed 0, shared search space) train on `dev` (997 rows) and "
                 "score macro-F1 on `devtest` (506 rows); the best params are then retrained on "
                 "dev+devtest (1503 rows) and evaluated on `test` (506 rows). Splits are the seed-0 "
                 "shuffled halves of the original devtest (dev = 997, devtest = first 506, test = "
                 "last 506).\n")

    lines.append("## (a) Tuned macro-F1: `devtest → test`\n")
    lines.append("| language | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---:|" * len(modes))
    for l in langs:
        vals = " | ".join(cell_tuned(l, m) for m in modes)
        lines.append(f"| {l} | {vals} |")
    lines.append("")

    lines.append("## (b) Tuned subset accuracy: `devtest → test`\n")
    lines.append("| language | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---:|" * len(modes))
    for l in langs:
        vals = " | ".join(cell_tuned_acc(l, m) for m in modes)
        lines.append(f"| {l} | {vals} |")
    lines.append("")

    lines.append("## (c) Baseline subset accuracy: `devtest → test`\n")
    lines.append("| language | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---:|" * len(modes))
    for l in langs:
        vals = " | ".join(cell_base(l, m) for m in modes)
        lines.append(f"| {l} | {vals} |")
    lines.append("")

    # ---- bullets ----
    lines.append("## Notes\n")

    # 1. winning mode per language (by tuned test macro-F1)
    winners = {l: max(modes, key=lambda m: data[l]["modes"][m]["tuned"]["test_macro"])
               for l in langs}
    lines.append("- **Winning mode per language (tuned `test` macro-F1):** " +
                 ", ".join(f"{l} `{winners[l]}` "
                           f"({data[l]['modes'][winners[l]]['tuned']['test_macro']:.4f})"
                           for l in langs) +
                 ". Macro-F1 is driven by the high-count tags (`travel`, and to a lesser degree "
                 "`sports`/`politics`); rare tags (health, safety, science, geography) have single-digit "
                 "positives in `test` and near-zero F1, so per-cell differences are noisy.")

    # 2. overall best mode
    best_overall = max(modes, key=lambda m: np.mean(
        [data[l]["modes"][m]["tuned"]["test_macro"] for l in langs]))
    lines.append(f"- **Best mode overall (mean tuned `test` macro-F1 over the four languages):** "
                 f"`{best_overall}` (" +
                 ", ".join(f"{m} {np.mean([data[l]['modes'][m]['tuned']['test_macro'] for l in langs]):.4f}"
                           for m in modes) + ").")

    # 3. English vs the other three languages
    others = [l for l in langs if l != "en"]
    if "en" in langs and others:
        en_test = {m: data["en"]["modes"][m]["tuned"]["test_macro"] for m in modes}
        oth_test = {m: np.mean([data[l]["modes"][m]["tuned"]["test_macro"] for l in others])
                    for m in modes}
        lines.append(f"- **en vs it/de/fr:** mean tuned `test` macro-F1 across {others} is " +
                     ", ".join(f"{m} {oth_test[m]:.4f}" for m in modes) +
                     f"; for `en` it is " + ", ".join(f"{m} {en_test[m]:.4f}" for m in modes) +
                     " (no cross-lingual transfer in this experiment).")

    # 4. devtest -> test shift (per language, mean over modes)
    shifts = {l: np.mean([data[l]["modes"][m]["tuned"]["test_macro"]
                          - data[l]["modes"][m]["tuned"]["devtest_macro"] for m in modes])
              for l in langs}
    lines.append("- **`devtest → test` shift of the tuned model (mean over modes):** " +
                 ", ".join(f"{l} {shifts[l]:+.4f}" for l in langs) +
                 ". The best params are selected on `devtest`, so `test` is an unbiased-ish read of "
                 "the same protocol; a negative shift is selection optimism, a positive one is split "
                 "noise on small rare-tag counts.")

    # 5. per-language spread across modes
    spread = {l: (max(data[l]["modes"][m]["tuned"]["test_macro"] for m in modes)
                  - min(data[l]["modes"][m]["tuned"]["test_macro"] for m in modes))
              for l in langs}
    lines.append("- **Mode spread within a language (max−min tuned `test` macro-F1):** " +
                 ", ".join(f"{l} {spread[l]:.4f}" for l in langs) +
                 ". The pooling choice matters less than the language for the strongest mode; "
                 "mean pooling is the usual front-runner, consistent with the retrieval result.")

    with open(os.path.join(outdir, "xgboost_perlang.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_perlang_md_lenient(data, outdir, thresholds="tuned"):
    """Lenient-label / tuned-threshold merge grid (new tables (a)-(c))."""
    langs = [l for l in LANGS if l in data]
    modes = [m for m in MODES if any(m in data[l]["modes"] for l in langs)]

    def tv(l, m, which, metric):
        return data[l]["modes"][m]["tuned"][which][metric]

    def cell(l, m, which, metric):
        return f"{tv(l, m, which, 'devtest')[metric]:.3f} → {tv(l, m, which, 'test')[metric]:.3f}"

    def table(title, which, metric):
        out = [f"## {title}\n"]
        out.append("| language | " + " | ".join(modes) + " |")
        out.append("|---|" + "---:|" * len(modes))
        for l in langs:
            out.append(f"| {l} | " + " | ".join(cell(l, m, which, metric) for m in modes) + " |")
        out.append("")
        return out

    lines = []
    lines.append("# XGBoost topic classifier — per-language × pooling grid (lenient labels)\n")
    lines.append(f"Generated by `xgboost_topic_optuna.py --mode merge --labels lenient` — "
                 f"xgboost {xgb.__version__}, optuna {optuna.__version__}, `tree_method=hist`, "
                 f"`max_bin={MAX_BIN}`, 2 threads per worker. No cross-lingual transfer.\n")
    lines.append("**Label scheme: `lenient`** — raw comma-tags are mapped onto the 8 canonical tags "
                 "via the curated aliases in `the label audit` §1 (singular/plural, `crime`/`law` → "
                 "`crime and law`, `science/*`/`science and technology` → `science`, misspelling "
                 "`sciece` → `science`, `tourism`/`accomodation` → `travel`, `disasters and "
                 "accidents`/`accident` → `safety`, `politics and conflicts` → `politics`, plus "
                 "country/continent tags → `geography`). This recovers the large fraction of positives "
                 "that exact-string matching drops and cuts all-zero rows from 1147/2009 (strict) to "
                 "~504/2009.\n")
    lines.append("**Thresholds: per-tag tuned.** For each tag the threshold in 0.05..0.95 (step 0.05) "
                 "maximising that tag's F1 on the model's own training-set predictions is chosen "
                 "in-sample (dev for the dev-trained model, dev+devtest for the dev+devtest model) and "
                 "then applied to the evaluation split. The Optuna search itself is unchanged: it still "
                 "maximises devtest macro-F1 at the fixed 0.5 threshold.\n")
    lines.append("Protocol: baseline (default params) and every Optuna trial (TPE seed 0, shared search "
                 "space) train on `dev` (997) and score `devtest` (506); the best params are retrained on "
                 "dev+devtest (1503) and evaluated on `test` (506).\n")

    lines.extend(table("(a) Tuned-threshold macro-F1: `devtest → test`", "tuned", "macro_f1"))
    lines.extend(table("(b) Micro-F1: `devtest → test`", "tuned", "micro_f1"))
    lines.extend(table("(c) Tuned-threshold subset accuracy: `devtest → test`", "tuned", "subset_acc"))
    lines.extend(table("(d) Tuned-threshold macro-F1 over tags with >= 50 positives: `devtest → test`",
                       "tuned", "macro_f1_ge50"))

    lines.append("## Notes\n")
    winners = {l: max(modes, key=lambda m: tv(l, m, "tuned", "test")["macro_f1"]) for l in langs}
    lines.append("- **Winning mode per language (tuned `test` macro-F1):** " +
                 ", ".join(f"{l} `{winners[l]}` "
                           f"({tv(l, winners[l], 'tuned', 'test')['macro_f1']:.4f})" for l in langs) + ".")
    best_overall = max(modes, key=lambda m: np.mean(
        [tv(l, m, "tuned", "test")["macro_f1"] for l in langs]))
    lines.append(f"- **Best mode overall (mean tuned `test` macro-F1):** `{best_overall}` (" +
                 ", ".join(f"{m} {np.mean([tv(l, m, 'tuned', 'test')['macro_f1'] for l in langs]):.4f}"
                           for m in modes) + ").")
    n_ge50 = {l: data[l]["modes"][modes[0]]["tuned"]["tuned"]["test"]["n_tags_ge50"]
              for l in langs if modes}
    if n_ge50:
        lines.append(f"- **Macro rule:** macro-F1 is reported over all 8 tags (tables a/b/c) and "
                     f"over the tags whose full-dataset positive count is >= {MIN_POS_FOR_MACRO} "
                     f"(table d). Qualifying tags (`n_tags_ge50`) = " +
                     ", ".join(f"{l} {n_ge50[l]}" for l in langs) +
                     ". With lenient labels all 8 tags qualify (full-dataset counts are 81-518), so "
                     "table (d) equals table (a); the rule is stated so it can be re-checked under a "
                     "different label scheme.")
    shifts = {l: np.mean([tv(l, m, "tuned", "test")["macro_f1"]
                          - tv(l, m, "tuned", "devtest")["macro_f1"] for m in modes]) for l in langs}
    lines.append("- **`devtest → test` shift of the tuned model (mean over modes):** " +
                 ", ".join(f"{l} {shifts[l]:+.4f}" for l in langs) +
                 ". The per-tag thresholds are tuned in-sample on the training split, so `test` remains "
                 "an out-of-sample read; a negative shift is selection optimism.")
    lines.append(f"- **Reproducibility:** lenient labels + per-tag tuned thresholds. Per-mode trials: "
                 "`xgboost_perlang_lenient_trials_{lang}_{mode}.csv`; full metrics (fixed and tuned) "
                 "with per-tag thresholds: `xgboost_perlang_lenient_{lang}.json`.")

    path = os.path.join(outdir, "xgboost_perlang_lenient.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def merge_perlang(langs, outdir, labels="strict", thresholds="fixed"):
    prefix = "xgboost_perlang_lenient" if labels == "lenient" else "xgboost_perlang"
    data = {}
    for lang in langs:
        p = os.path.join(outdir, f"{prefix}_{lang}.json")
        if not os.path.exists(p):
            log(f"[merge] MISSING {p} (skipping)")
            continue
        with open(p, encoding="utf-8") as f:
            data[lang] = json.load(f)
    if not data:
        log("[merge] no per-language JSONs found; nothing written")
        return
    if labels == "lenient" or thresholds == "tuned":
        path = write_perlang_md_lenient(data, outdir, thresholds=thresholds)
    else:
        write_perlang_md(data, outdir)
        path = os.path.join(outdir, "xgboost_perlang.md")
    log(f"[merge] wrote {path} from {list(data)}")


# ----------------------------------------------------------------------------
# Status file
# ----------------------------------------------------------------------------
def write_status(state, path=STATUS_PATH):
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="XGBoost FLORES topic classifier pooling-mode sweep")
    ap.add_argument("--mode", default="all", choices=MODES + ["all", "perlang", "merge"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--outdir", default=BASE)
    ap.add_argument("--langs", default=",".join(LANGS),
                    help="comma list of languages (perlang / merge), e.g. en,it,de,fr")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="comma list of pooling modes for --mode perlang (default all four)")
    ap.add_argument("--labels", default="strict", choices=["strict", "lenient"],
                    help="label scheme: strict exact canonical match (default) or "
                         "lenient curated aliases (the label audit §1)")
    ap.add_argument("--thresholds", default="fixed", choices=["fixed", "tuned"],
                    help="decision thresholds: fixed 0.5 (default) or per-tag tuned; "
                         "run_perlang_mode always reports both, this selects the merge view")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- new per-language / merge paths (existing --mode all/mean/... untouched) ---
    if args.mode in ("perlang", "merge"):
        langs = _parse_list(args.langs)
        bad = [l for l in langs if l not in LANGS]
        if bad:
            ap.error(f"unknown language(s): {bad}; valid: {LANGS}")
        if args.mode == "merge":
            merge_perlang(langs, args.outdir, labels=args.labels, thresholds=args.thresholds)
            return
        pmodes = _parse_list(args.modes)
        badm = [m for m in pmodes if m not in MODES]
        if badm:
            ap.error(f"unknown mode(s): {badm}; valid: {MODES}")
        if not pmodes:
            ap.error("--modes must select at least one pooling mode")
        log(f"[perlang] languages={langs} modes={pmodes} trials={args.trials} "
            f"labels={args.labels} thresholds={args.thresholds} pid={os.getpid()}")
        run_perlang(langs, pmodes, args.trials, args.outdir,
                    labels=args.labels, thresholds=args.thresholds)
        return

    modes = MODES if args.mode == "all" else [args.mode]

    df, Y, masks, counts = load_data(args.labels)
    suffix = "_lenient" if args.labels == "lenient" else ""
    if args.labels == "lenient":
        _report_lenient_counts(Y)

    status_path = os.path.join(args.outdir, f"xgboost_allmodes{suffix}_status.json")
    pid_path = os.path.join(args.outdir, f"xgboost_allmodes{suffix}.pid")

    status = {"pid": os.getpid(), "argv": vars(args), "labels": args.labels,
              "thresholds": args.thresholds,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "modes": {m: {"state": "pending"} for m in modes}}
    if args.mode == "all":
        with open(pid_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        write_status(status, status_path)

    log("Label positive counts (all / dev / devtest / test):")
    for j, tag in enumerate(TAGS):
        log(f"  {tag:<14} {counts['all'][j]:>4} / {counts['dev'][j]:>4} / "
            f"{counts['devtest'][j]:>4} / {counts['test'][j]:>4}")

    results = {}
    t_all = time.time()
    for mode in modes:
        status["modes"][mode] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if args.mode == "all":
            write_status(status, status_path)
        try:
            res = run_mode(mode, args.trials, args.outdir, Y, masks, counts, df,
                           labels=args.labels, thresholds=args.thresholds)
            results[mode] = res
            status["modes"][mode] = {
                "state": "done",
                "base_macro_f1": round(res["base_macro"], 4),
                "tuned_macro_f1": round(res["tuned_macro"], 4),
                "tuned_threshold_macro_f1": round(res["tuned_tuned"]["macro_f1"], 4),
                "best_trial": res["best_trial"],
                "runtime_s": round(res["runtime"], 1),
            }
        except Exception as exc:  # keep going; record failure
            status["modes"][mode] = {"state": "failed", "error": repr(exc)}
            log(f"[{mode}] FAILED: {exc!r}")
        if args.mode == "all":
            write_status(status, status_path)

    if args.mode == "all":
        elapsed = time.time() - t_all
        if suffix:
            path = write_allmodes_md_lenient(results, args.outdir, elapsed, suffix=suffix)
        else:
            write_allmodes_md(results, args.outdir, elapsed)
            path = os.path.join(args.outdir, "xgboost_topic_allmodes.md")
        status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["elapsed_s"] = round(elapsed, 1)
        write_status(status, status_path)
        log(f"\nAll modes done in {elapsed:.1f}s; wrote {path}")


if __name__ == "__main__":
    main()
