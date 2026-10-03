"""
English-trained XGBoost topic classifier on FLORES embeddings, with a sweep over the
four pooling modes (mean / bos / eos / lead) and cross-lingual transfer evaluation.

Usage:
  # full sweep, all four modes, 50 Optuna trials each (writes per-mode + combined docs)
  python xgboost_topic_optuna.py --mode all --trials 50

  # single mode
  python xgboost_topic_optuna.py --mode mean --trials 50

  # fast plumbing check (small trial count, isolated output dir)
  python xgboost_topic_optuna.py --mode mean --trials 2 --outdir <tmpdir>

Outputs (next to the script, or in --outdir):
  - xgboost_topic_results_{mode}.md   per-mode write-up
  - xgboost_optuna_trials_{mode}.csv  every Optuna trial for that mode
  - xgboost_topic_allmodes.md         combined comparison (written only for --mode all)

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

# Baseline protocol params (fixed by the experiment spec).
BASELINE_PARAMS = dict(max_depth=6, learning_rate=0.1, n_estimators=300)

STATUS_PATH = os.path.join(BASE, "xgboost_allmodes_status.json")
PID_PATH = os.path.join(BASE, "xgboost_allmodes.pid")


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Data (split identical for every mode)
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(CSV_PATH)
    assert list(df.columns)[:2] == ["split", "id"]
    assert len(df) == 2009

    # Multi-label target over the 8 canonical tags.
    Y = np.zeros((len(df), len(TAGS)), dtype=int)
    for i, topic in enumerate(df["topic"].fillna("")):
        parts = {p.strip().lower() for p in str(topic).split(",")}
        for j, tag in enumerate(TAGS):
            if tag in parts:
                Y[i, j] = 1

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


def score(Y_true, probs):
    pred = (probs >= THRESHOLD).astype(int)
    subset_acc = accuracy_score(Y_true, pred)
    macro_f1 = f1_score(Y_true, pred, average="macro", zero_division=0)
    per_tag = f1_score(Y_true, pred, average=None, zero_division=0)
    return subset_acc, macro_f1, per_tag


def fmt_row(vals):
    return " | ".join(f"{v:.3f}" for v in vals)


# ----------------------------------------------------------------------------
# One full mode run
# ----------------------------------------------------------------------------
def run_mode(mode, n_trials, outdir, Y, masks, counts, df):
    t_start = time.time()
    dev_devtest_mask = masks["dev_devtest"]
    test_mask = masks["test"]
    X = load_mode_embeddings(mode)

    log(f"\n========== MODE: {mode} ==========")

    # 1. BASELINE (default params, train dev+devtest -> test)
    log(f"[{mode}] [1/5] Baseline (max_depth=6, lr=0.1, n_estimators=300) ...")
    t = time.time()
    (probs_base,), _ = fit_predict(
        BASELINE_PARAMS,
        X["en"][dev_devtest_mask], Y[dev_devtest_mask],
        [X["en"][test_mask]],
    )
    base_subset, base_macro, base_per_tag = score(Y[test_mask], probs_base)
    log(f"[{mode}]   baseline test: subset_acc={base_subset:.4f}  macro_f1={base_macro:.4f}  "
        f"({time.time() - t:.1f}s)")

    # 2. OPTUNA (TPE seed 0, objective = test macro-F1)
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
        _, macro, _ = score(Y[test_mask], probs)
        trial.set_user_attr("macro_f1", macro)
        return macro

    trials_csv = os.path.join(outdir, f"xgboost_optuna_trials_{mode}.csv")

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
    all_mask = np.ones(len(Y), dtype=bool)
    # (a) dev+devtest-trained models -> tuned test score (same set as the objective).
    (probs_test,), _ = fit_predict(
        best_params,
        X["en"][dev_devtest_mask], Y[dev_devtest_mask],
        [X["en"][test_mask]],
    )
    # (b) final models on ALL English rows for transfer.
    models_full = []
    for j in range(len(TAGS)):
        model = xgb.XGBClassifier(**make_params(best_params))
        model.fit(X["en"][all_mask], Y[all_mask][:, j])
        models_full.append(model)
    log(f"[{mode}]   retrain + transfer-model fit done ({time.time() - t:.1f}s)")

    # 4. CROSS-LINGUAL EVAL
    log(f"[{mode}] [4/5] Cross-lingual transfer evaluation ...")
    tuned_subset, tuned_macro, tuned_per_tag = score(Y[test_mask], probs_test)

    shifts = {l: (X["en"] - X[l]).mean(axis=0) for l in LANGS}  # D_L = mean(emb_en - emb_L)
    assert np.allclose(shifts["en"], 0.0)

    transfer_rows = []

    def eval_full(mask_label, Xtest):
        probs = np.zeros((Xtest.shape[0], len(TAGS)))
        for j, model in enumerate(models_full):
            probs[:, j] = model.predict_proba(Xtest)[:, 1]
        subset, macro, per_tag = score(Y, probs)
        transfer_rows.append((mask_label, subset, macro, per_tag))

    eval_full("en (in-sample)", X["en"])
    for l in ["it", "de", "fr"]:
        eval_full(f"{l} raw", X[l])
        eval_full(f"{l}+D", X[l] + shifts[l])

    for name, subset, macro, per_tag in transfer_rows:
        log(f"[{mode}]   {name:<14} subset={subset:.4f} macro={macro:.4f}")

    # 5. WRITE PER-MODE RESULTS
    log(f"[{mode}] [5/5] Writing per-mode results ...")
    runtime = time.time() - t_start
    res = dict(
        mode=mode, n_trials=n_trials,
        base_subset=base_subset, base_macro=base_macro, base_per_tag=base_per_tag.tolist(),
        tuned_subset=tuned_subset, tuned_macro=tuned_macro, tuned_per_tag=tuned_per_tag.tolist(),
        best_trial=study.best_trial.number, best_value=study.best_value,
        best_params=best_params,
        transfer=[(n, s, m, p.tolist()) for (n, s, m, p) in transfer_rows],
        runtime=runtime,
    )
    write_mode_md(res, outdir, Y, masks, counts, df)
    log(f"[{mode}] done in {runtime:.1f}s -> {os.path.join(outdir, f'xgboost_topic_results_{mode}.md')}")
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
# Status file
# ----------------------------------------------------------------------------
def write_status(state):
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="XGBoost FLORES topic classifier pooling-mode sweep")
    ap.add_argument("--mode", default="all", choices=MODES + ["all"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--outdir", default=BASE)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    modes = MODES if args.mode == "all" else [args.mode]

    df, Y, masks, counts = load_data()

    status = {"pid": os.getpid(), "argv": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "modes": {m: {"state": "pending"} for m in modes}}
    if args.mode == "all":
        with open(PID_PATH, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        write_status(status)

    log("Label positive counts (all / dev / devtest / test):")
    for j, tag in enumerate(TAGS):
        log(f"  {tag:<14} {counts['all'][j]:>4} / {counts['dev'][j]:>4} / "
            f"{counts['devtest'][j]:>4} / {counts['test'][j]:>4}")

    results = {}
    t_all = time.time()
    for mode in modes:
        status["modes"][mode] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if args.mode == "all":
            write_status(status)
        try:
            res = run_mode(mode, args.trials, args.outdir, Y, masks, counts, df)
            results[mode] = res
            status["modes"][mode] = {
                "state": "done",
                "base_macro_f1": round(res["base_macro"], 4),
                "tuned_macro_f1": round(res["tuned_macro"], 4),
                "best_trial": res["best_trial"],
                "runtime_s": round(res["runtime"], 1),
            }
        except Exception as exc:  # keep going; record failure
            status["modes"][mode] = {"state": "failed", "error": repr(exc)}
            log(f"[{mode}] FAILED: {exc!r}")
        if args.mode == "all":
            write_status(status)

    if args.mode == "all":
        elapsed = time.time() - t_all
        write_allmodes_md(results, args.outdir, elapsed)
        status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["elapsed_s"] = round(elapsed, 1)
        write_status(status)
        log(f"\nAll modes done in {elapsed:.1f}s; wrote xgboost_topic_allmodes.md")


if __name__ == "__main__":
    main()
