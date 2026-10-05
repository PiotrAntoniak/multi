"""Driver: FLORES XGBoost topic grid with the top-50%-language-mass dims ZEROED.

Runs 4 (language, pooling) cells sequentially -- en/mean, en/bos, it/mean, it/bos -- reusing the
exact per-language protocol of xgboost_topic_optuna.py --mode perlang (lenient labels, per-tag
tuned thresholds):

  * Optuna TPE seed 0, 50 trials; each trial trains on dev (997) and scores macro-F1 on devtest
    (506) at fixed 0.5 (objective = devtest macro-F1).
  * Best params retrained on dev+devtest (1503) and evaluated on test (506); per-tag thresholds
    tuned in-sample on the dev+devtest training predictions.
  * Baseline (default params) reported for reference.

Difference vs the README grid: for each pooling mode we rank the 768 dims by the 4-language
per-dim eta^2 (one-way ANOVA over en/it/de/fr, via the shared lang_metrics.top_dims helper), take
the top-k dims covering 50% of the eta^2 mass (k = 237 for mean, 211 for bos), and zero exactly
those dims in every array used for training / scoring / evaluation of the cell.

Launched detached by launch_hidden.vbs; writes xgboost_flores_drop_optuna.log / .pid /
_status.json (all git-ignored). The status JSON is rewritten after every trial.
"""
import os

# Must be set before numpy / xgboost import to take effect (2 threads per the budget).
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import json
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(REPO, "xgboost_flores_drop_optuna.log")
PID_PATH = os.path.join(REPO, "xgboost_flores_drop_optuna.pid")
STATUS_PATH = os.path.join(REPO, "xgboost_flores_drop_optuna_status.json")

N_TRIALS = 50
FRACTION = 0.50
CELLS = [("en", "mean"), ("en", "bos"), ("it", "mean"), ("it", "bos")]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_pid(pid):
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(pid))


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, STATUS_PATH)


# Write the PID immediately (the python pid is the process to kill on abort).
write_pid(os.getpid())

# Import the repo module (reuses its masks, split, label scheme, functions, protocol).
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import optuna  # noqa: E402

import model_store  # noqa: E402
import xgboost_topic_optuna as X  # noqa: E402
from lang_metrics import top_dims  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_cell(lang, mode, zero_dims, k, Y, masks, pos, state):
    dev, devtest, test, dev_devtest = (masks["dev"], masks["devtest"],
                                        masks["test"], masks["dev_devtest"])
    cell = f"{lang}/{mode}"
    log(f"===== CELL {cell}: zeroing top-{k} {mode} eta^2 dims =====")

    # Zero exactly the top-k language dims in EVERY array used for this cell.
    Xl = X.load_one_embedding(mode, lang).copy()
    Xl[:, zero_dims] = 0.0

    # 1. BASELINE (default params), same two fits as run_perlang_mode.
    log(f"[{cell}] baseline: dev->devtest and dev+devtest->test ...")
    tp_bdt, (probs_bdt,), _ = X.fit_predict_with_train(
        X.BASELINE_PARAMS, Xl[dev], Y[dev], [Xl[devtest]])
    b_dt_fixed = X.score(Y[devtest], probs_bdt, pos_counts=pos)
    b_dt_th = X.tune_thresholds(Y[dev], tp_bdt)
    b_dt_tuned = X.score(Y[devtest], probs_bdt, b_dt_th, pos_counts=pos)
    tp_btest, (probs_btest,), _ = X.fit_predict_with_train(
        X.BASELINE_PARAMS, Xl[dev_devtest], Y[dev_devtest], [Xl[test]])
    b_test_fixed = X.score(Y[test], probs_btest, pos_counts=pos)
    b_test_th = X.tune_thresholds(Y[dev_devtest], tp_btest)
    b_test_tuned = X.score(Y[test], probs_btest, b_test_th, pos_counts=pos)
    log(f"[{cell}] baseline test: fixed macro={b_test_fixed['macro_f1']:.4f} "
        f"tuned macro={b_test_tuned['macro_f1']:.4f} | devtest fixed="
        f"{b_dt_fixed['macro_f1']:.4f} tuned={b_dt_tuned['macro_f1']:.4f}")

    # 2. OPTUNA (TPE seed 0; train dev, objective = devtest macro-F1 @0.5).
    state["cell"] = cell
    state["trial"] = -1
    state["best_so_far"] = None
    state["cells"][cell] = {"state": "running", "trial": -1, "best_so_far": None,
                            "k": k, "mode": mode, "lang": lang}
    write_status(state)

    def objective(trial):
        params = X._optuna_search_space(trial)
        (probs,), _ = X.fit_predict(params, Xl[dev], Y[dev], [Xl[devtest]])
        macro = X.score(Y[devtest], probs, pos_counts=pos)["macro_f1"]
        trial.set_user_attr("macro_f1", macro)
        return macro

    def callback(study, trial):
        state["trial"] = int(trial.number)
        state["best_so_far"] = float(study.best_value)
        state["last_value"] = (float(trial.value) if trial.value is not None else None)
        state["cells"][cell] = {"state": "running", "trial": int(trial.number),
                                "best_so_far": float(study.best_value), "k": k,
                                "last_value": state["last_value"]}
        write_status(state)
        log(f"[{cell}] trial {trial.number:>2}/{N_TRIALS} value={state['last_value']} "
            f"best={state['best_so_far']:.4f}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=X.SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    best_params = dict(study.best_params)
    log(f"[{cell}] optuna best #{study.best_trial.number} devtest macro={study.best_value:.4f} "
        f"params={best_params}")

    # 3. RETRAIN best params on dev+devtest -> test (fixed + in-sample tuned thresholds).
    tp_btest, (probs_btest,), models = X.fit_predict_with_train(
        best_params, Xl[dev_devtest], Y[dev_devtest], [Xl[test]])
    t_test_fixed = X.score(Y[test], probs_btest, pos_counts=pos)
    t_test_th = X.tune_thresholds(Y[dev_devtest], tp_btest)
    t_test_tuned = X.score(Y[test], probs_btest, t_test_th, pos_counts=pos)
    log(f"[{cell}] BEST test: fixed macro={t_test_fixed['macro_f1']:.4f} "
        f"micro={t_test_fixed['micro_f1']:.4f} subset={t_test_fixed['subset_acc']:.4f} | "
        f"tuned macro={t_test_tuned['macro_f1']:.4f} micro={t_test_tuned['micro_f1']:.4f} "
        f"subset={t_test_tuned['subset_acc']:.4f}")

    # 3b. Persist the retrained tag models so evaluations can load instead of retrain.
    bundle_name = f"flores_{lang}_{mode}_drop{int(FRACTION * 100)}_optuna"
    bundle_meta = {
        "lang": lang, "mode": mode, "fraction": FRACTION, "k": k,
        "zeroed_dims_note": "zeroed dims recomputable via "
                            "lang_metrics.top_dims(mode, FRACTION)",
        "best_params": best_params,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "n_trials": N_TRIALS,
        "thresholds": t_test_th.tolist(),
        "train_rows": "dev+devtest",
        "test_fixed": t_test_fixed,
        "test_tuned": t_test_tuned,
    }
    model_store.save_bundle(bundle_name, models, bundle_meta,
                            root=os.path.join(REPO, "models"))
    log(f"[{cell}] saved bundle '{bundle_name}' ({len(models)} tag models)")

    res = {
        "lang": lang, "mode": mode, "k": k, "n_trials": N_TRIALS,
        "best_trial": int(study.best_trial.number), "best_value": float(study.best_value),
        "best_params": best_params,
        "baseline": {"test_fixed": b_test_fixed, "test_tuned": b_test_tuned,
                     "devtest_fixed": b_dt_fixed, "devtest_tuned": b_dt_tuned},
        "best": {"test_fixed": t_test_fixed, "test_tuned": t_test_tuned,
                 "thresholds": t_test_th.tolist()},
    }
    state["cells"][cell] = {"state": "done", "trial": N_TRIALS,
                            "best_so_far": float(study.best_value), "k": k,
                            "best_trial": res["best_trial"],
                            "test_macro_fixed": t_test_fixed["macro_f1"],
                            "test_macro_tuned": t_test_tuned["macro_f1"],
                            "test_micro_tuned": t_test_tuned["micro_f1"],
                            "test_subset_tuned": t_test_tuned["subset_acc"]}
    state.setdefault("results", {})[cell] = res
    write_status(state)
    return res


def main():
    t0 = time.time()
    log("FLORES drop-dim Optuna driver starting")
    log(f"python pid={os.getpid()} | trials={N_TRIALS} | cells={CELLS}")
    log(f"xgboost={__import__('xgboost').__version__} optuna={optuna.__version__}")

    df, Y, masks, counts = X.load_data("lenient")
    pos = counts["all"]
    log(f"loaded lenient labels: dev={int(masks['dev'].sum())} devtest={int(masks['devtest'].sum())} "
        f"test={int(masks['test'].sum())} dev+devtest={int(masks['dev_devtest'].sum())}")

    # Top-k dims per pooling mode (computed once, shared by both languages of that mode).
    zdims = {}
    for mode in ("mean", "bos"):
        dims, k = top_dims(mode, FRACTION)
        zdims[mode] = (dims, k)
        log(f"eta^2 {mode}: k(50%)={k} dims")

    state = {
        "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": N_TRIALS, "fraction": FRACTION, "cells_order": [f"{l}/{m}" for l, m in CELLS],
        "cell": None, "trial": None, "best_so_far": None, "cells": {},
    }
    write_status(state)

    for lang, mode in CELLS:
        zero_dims, k = zdims[mode]
        run_cell(lang, mode, zero_dims, k, Y, masks, pos, state)

    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["elapsed_s"] = round(time.time() - t0, 1)
    write_status(state)
    log(f"ALL DONE in {state['elapsed_s']}s")
    summary = {c: {"k": r["k"],
                   "test_macro_fixed": r["best"]["test_fixed"]["macro_f1"],
                   "test_macro_tuned": r["best"]["test_tuned"]["macro_f1"],
                   "test_micro_tuned": r["best"]["test_tuned"]["micro_f1"],
                   "test_subset_tuned": r["best"]["test_tuned"]["subset_acc"]}
               for c, r in state.get("results", {}).items()}
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # record failure in the status file, then surface it in the log
        try:
            write_status({"pid": os.getpid(), "state": "failed", "error": repr(exc),
                          "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        finally:
            log(f"FATAL: {exc!r}")
        raise
