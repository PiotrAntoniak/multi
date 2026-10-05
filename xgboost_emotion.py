"""XGBoost on EuroBERT-210m sentence embeddings for `dair-ai/emotion` (6-class).

Two pooling modes of the LAST hidden state of `EuroBERT/EuroBERT-210m`
(native `AutoModel` + `AutoTokenizer`; remote code is broken on transformers 5.x):

  * mean : masked mean over all non-pad tokens of the PLAIN forward (the
           tokenizer appends the end token, which is included).
  * bos  : position 0 of the last hidden state of a forward over the PLAIN ids
           with special id 128000 manually PREPENDED (+ a leading 1 mask).

Dataset `dair-ai/emotion`, config `split`: train 16,000 / validation 2,000 / test
2,000 rows (text field `text`, label field `label`; classes sadness, joy, love,
anger, fear, surprise). `--limit-train N` keeps the first N train rows (smoke only).
Embeddings are cached to `<outdir>/emotion_emb/{mode}_{train,val,test}.npy` (float32)
and reused if present.

XGBoost multiclass (`objective=multi:softprob`, `num_class=6`, `eval_metric=mlogloss`,
`tree_method=hist`, `max_bin=64`) with a baseline (max_depth=6, learning_rate=0.1,
n_estimators=300) and an Optuna TPE (seed 0) search identical to
`xgboost_topic_optuna.py::_optuna_search_space`; objective = validation AUC (OVR macro),
trained on the train split. Reports validation + test accuracy, macro-F1, AUC, and a
per-emotion breakdown (F1 / recall / one-vs-rest AUC per class).

Usage: python xgboost_emotion.py --modes mean,bos --trials 50 --limit-train 0 --outdir . --threads 4
"""
import os
import sys

# Thread cap must be set before numpy / torch imports to take effect. The --threads
# flag is pre-scanned from argv so the smoke run's lower cap is honored; default 4.
_THREADS = "4"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _THREADS = sys.argv[_i + 1]
    elif _a.startswith("--threads="):
        _THREADS = _a.split("=", 1)[1]
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import xgboost as xgb
import optuna
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "EuroBERT/EuroBERT-210m"
BOS_ID = 128000
BATCH, MAX_LEN = 256, 512
MAX_BIN = 64
SEED = 0
NUM_CLASS = 6
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
# Baseline protocol params (fixed by the experiment spec); merged with make_params defaults.
BASELINE_PARAMS = dict(max_depth=6, learning_rate=0.1, n_estimators=300)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def make_params(threads, extra=None):
    p = dict(objective="multi:softprob", num_class=NUM_CLASS, eval_metric="mlogloss",
             tree_method="hist", max_bin=MAX_BIN, n_jobs=threads, seed=SEED)
    if extra:
        p.update(extra)
    return p


def _optuna_search_space(trial):
    """Identical search space to `xgboost_topic_optuna.py::_optuna_search_space`."""
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


# ---------------------------------------------------------------------------
# Data + embeddings
# ---------------------------------------------------------------------------
def load_emotion(limit_train):
    local = os.path.join(BASE, "emotion.csv")
    if os.path.exists(local):
        import pandas as pd
        df = pd.read_csv(local)
        train = df[df["split"] == "train"]
        val = df[df["split"] == "validation"]
        test = df[df["split"] == "test"]
        if limit_train and limit_train > 0:
            train = train.head(limit_train)
        log(f"loaded {local}: train={len(train)} val={len(val)} test={len(test)}")
        return (list(train["text"]), np.asarray(train["label"], dtype=np.int64),
                list(val["text"]), np.asarray(val["label"], dtype=np.int64),
                list(test["text"]), np.asarray(test["label"], dtype=np.int64))
    ds = load_dataset("dair-ai/emotion", "split")
    train, val = ds["train"], ds["validation"]
    if limit_train and limit_train > 0:
        train = train.select(range(min(limit_train, len(train))))
    return (list(train["text"]), np.asarray(train["label"], dtype=np.int64),
            list(val["text"]), np.asarray(val["label"], dtype=np.int64), None, None)


def embed_split(mode, texts, tokenizer, model, device, out_path, split, status_path):
    """Return float32 (n, hidden) embeddings for one split, caching to `out_path`."""
    if os.path.exists(out_path):
        arr = np.load(out_path)
        log(f"  [embed/{mode}/{split}] cache hit {arr.shape} {arr.dtype} -> {out_path}")
        return arr
    n = len(texts)
    n_batches = (n + BATCH - 1) // BATCH
    acc = []
    with torch.no_grad():
        for b in range(n_batches):
            chunk = texts[b * BATCH:(b + 1) * BATCH]
            max_len = MAX_LEN - 1 if mode == "bos" else MAX_LEN
            tok = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)
            if mode == "mean":
                h = model(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).to(h.dtype)
                pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
            elif mode == "bos":
                lead = torch.full((ids.shape[0], 1), BOS_ID, dtype=ids.dtype, device=device)
                p_ids = torch.cat([lead, ids], dim=1)
                p_mask = torch.cat([torch.ones_like(lead), mask], dim=1)
                h = model(input_ids=p_ids, attention_mask=p_mask).last_hidden_state
                pooled = h[:, 0, :].float().cpu().numpy()
            else:
                raise ValueError(f"unknown mode {mode!r}")
            acc.append(pooled.astype(np.float32))
            write_json(status_path, dict(stage="embedding", mode=mode, split=split, batch=b + 1,
                                         total_batches=n_batches, rows_done=min(n, (b + 1) * BATCH),
                                         rows_total=n, updated=datetime.now().isoformat(timespec="seconds")))
            if (b + 1) % 10 == 0 or b + 1 == n_batches:
                log(f"  [embed/{mode}/{split}] batch {b + 1}/{n_batches} "
                    f"({min(n, (b + 1) * BATCH)}/{n} rows)")
    arr = np.concatenate(acc, 0).astype(np.float32)
    np.save(out_path, arr)
    log(f"  [embed/{mode}/{split}] saved {arr.shape} {arr.dtype} -> {out_path}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite embeddings for {mode}/{split}")
    return arr


def metrics(y, probs):
    """Multiclass metrics: predict by argmax; overall + per-emotion (per-class) breakdown."""
    pred = probs.argmax(1)
    out = dict(accuracy=float(accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
               per_class_f1=f1_score(y, pred, average=None, zero_division=0).tolist(),
               per_class_recall=recall_score(y, pred, average=None, zero_division=0).tolist())
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        out["per_class_auc_ovr"] = roc_auc_score(y, probs, multi_class="ovr",
                                                 average=None).tolist()
    except Exception:  # noqa: BLE001
        out["auc_ovr_macro"] = float("nan")
        out["per_class_auc_ovr"] = [float("nan")] * NUM_CLASS
    return out


# ---------------------------------------------------------------------------
# One mode
# ---------------------------------------------------------------------------
def run_mode(mode, X_train, y_train, X_val, y_val, X_test, y_test, trials, threads, outdir,
             status_path):
    log(f"===== MODE {mode}: baseline + Optuna TPE {trials} trials =====")

    t0 = time.time()
    base = xgb.XGBClassifier(**make_params(threads, BASELINE_PARAMS))
    base.fit(X_train, y_train)
    base_m = metrics(y_val, base.predict_proba(X_val))
    base_tm = metrics(y_test, base.predict_proba(X_test))
    log(f"[{mode}] baseline val acc={base_m['accuracy']:.4f} macro_f1={base_m['macro_f1']:.4f} "
        f"auc={base_m['auc_ovr_macro']:.4f} | test acc={base_tm['accuracy']:.4f} "
        f"macro_f1={base_tm['macro_f1']:.4f} auc={base_tm['auc_ovr_macro']:.4f} "
        f"({time.time() - t0:.1f}s)")

    trials_csv = os.path.join(outdir, f"xgboost_emotion_trials_{mode}.csv")

    def objective(trial):
        t = time.time()
        params = _optuna_search_space(trial)
        clf = xgb.XGBClassifier(**make_params(threads, params))
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)
        m = metrics(y_val, probs)
        trial.set_user_attr("accuracy", m["accuracy"])
        trial.set_user_attr("macro_f1", m["macro_f1"])
        log(f"[{mode}] trial {trial.number + 1}/{trials} auc={m['auc_ovr_macro']:.4f} "
            f"acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"({time.time() - t:.1f}s) {params}")
        done = [t.value for t in study.trials if t.value is not None]
        write_json(status_path, dict(stage="optuna", mode=mode, trial=trial.number + 1,
                                     trials=trials, value_auc=m["auc_ovr_macro"],
                                     value_acc=m["accuracy"],
                                     best=max(done) if done else None,
                                     updated=datetime.now().isoformat(timespec="seconds")))
        return m["auc_ovr_macro"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))

    def save_trials(_study, _trial=None):
        rows = []
        for tr in _study.trials:
            if tr.value is None:
                continue
            row = {"trial": tr.number, "value_auc": tr.value,
                   "value_accuracy": tr.user_attrs.get("accuracy"), "state": tr.state.name}
            row.update(tr.params)
            rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).sort_values("trial").to_csv(trials_csv, index=False)

    study.optimize(objective, n_trials=trials, callbacks=[save_trials])
    best_params = dict(study.best_params)
    log(f"[{mode}] BEST trial #{study.best_trial.number}: auc={study.best_value:.4f} "
        f"(objective = val AUC OVR macro)")
    log(f"[{mode}] BEST params: {best_params}")

    tuned = xgb.XGBClassifier(**make_params(threads, best_params))
    tuned.fit(X_train, y_train)
    tuned_m = metrics(y_val, tuned.predict_proba(X_val))
    tuned_tm = metrics(y_test, tuned.predict_proba(X_test))
    log(f"[{mode}] tuned val acc={tuned_m['accuracy']:.4f} macro_f1={tuned_m['macro_f1']:.4f} "
        f"auc={tuned_m['auc_ovr_macro']:.4f} | test acc={tuned_tm['accuracy']:.4f} "
        f"macro_f1={tuned_tm['macro_f1']:.4f} auc={tuned_tm['auc_ovr_macro']:.4f}")
    return dict(mode=mode, baseline=base_m, baseline_test=base_tm, tuned=tuned_m,
                tuned_test=tuned_tm, best_params=best_params,
                best_trial=study.best_trial.number, best_value=float(study.best_value),
                n_trials=len([t for t in study.trials if t.value is not None]),
                trials_csv=os.path.basename(trials_csv))


def write_md(results, cfg, path):
    lines = ["# XGBoost on EuroBERT-210m `dair-ai/emotion` embeddings\n"]
    lines.append(f"Generated by `xgboost_emotion.py` — xgboost {xgb.__version__}, "
                 f"optuna {optuna.__version__}, transformers {__import__('transformers').__version__}, "
                 f"`objective=multi:softprob`, `eval_metric=mlogloss`, `tree_method=hist`, "
                 f"`max_bin={MAX_BIN}`, threads={cfg['threads']}, "
                 f"train={cfg['n_train']}, val={cfg['n_val']}, test={cfg['n_test']}.\n")
    lines.append("Embeddings: EuroBERT-210m native fp32, batch 256, GPU if available. "
                 "`mean` = masked mean of the plain forward (end token included); "
                 "`bos` = position 0 after prepending special id 128000. "
                 "6 classes: sadness, joy, love, anger, fear, surprise. "
                 "Optuna TPE seed 0, objective = validation AUC (OVR macro); accuracy and "
                 "per-emotion metrics are reported. "
                 "Baseline = max_depth=6, learning_rate=0.1, n_estimators=300.\n")
    lines.append(f"## Validation ({cfg['n_val']} rows) and test ({cfg['n_test']} rows)\n")
    lines.append("| mode | model | val acc | val macro-F1 | val AUC | test acc | test macro-F1 "
                 "| test AUC | trial count |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    def cells(mm):
        return (f"{mm['accuracy']:.4f} | {mm['macro_f1']:.4f} | {mm['auc_ovr_macro']:.4f}"
                if mm else "- | - | -")

    for res in results:
        m = res["mode"]
        lines.append(f"| {m} | baseline | {cells(res['baseline'])} | "
                     f"{cells(res['baseline_test'])} | - |")
        lines.append(f"| {m} | tuned | {cells(res['tuned'])} | "
                     f"{cells(res['tuned_test'])} | {res['n_trials']} |")
    lines.append("")
    lines.append("### Per-emotion performance (baseline vs tuned)\n")
    for res in results:
        lines.append(f"**{res['mode']}**\n")
        for split_key, split_name in (("", "validation"), ("_test", "test")):
            lines.append(f"*{split_name}*")
            lines.append("")
            lines.append("| emotion | baseline F1 | baseline recall | baseline AUC | tuned F1 "
                         "| tuned recall | tuned AUC |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            b = res[f"baseline{split_key}"]
            t = res[f"tuned{split_key}"]
            for j, name in enumerate(EMOTIONS):
                lines.append(
                    f"| {name} | {b['per_class_f1'][j]:.4f} | {b['per_class_recall'][j]:.4f} | "
                    f"{b['per_class_auc_ovr'][j]:.4f} | {t['per_class_f1'][j]:.4f} | "
                    f"{t['per_class_recall'][j]:.4f} | {t['per_class_auc_ovr'][j]:.4f} |")
            lines.append("")
    lines.append("")
    for res in results:
        lines.append(f"### Best Optuna parameters — {res['mode']}\n")
        lines.append("```")
        lines.append(f"best trial #: {res['best_trial']}")
        lines.append(f"best value (val AUC OVR macro): {res['best_value']:.4f}")
        for k, v in res["best_params"].items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append(f"\nPer-trial log: `{res['trials_csv']}`.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="mean,bos")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    emb_dir = os.path.join(outdir, "emotion_emb")
    os.makedirs(emb_dir, exist_ok=True)
    tag = args.modes.replace(",", "+")
    status_path = os.path.join(outdir, f"xgboost_emotion_status_{tag}.json")
    pid_path = os.path.join(outdir, f"xgboost_emotion_{tag}.pid")
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))
    log(f"pid={os.getpid()} modes={args.modes} trials={args.trials} limit_train={args.limit_train} "
        f"outdir={outdir} threads={args.threads}")
    write_json(status_path, dict(stage="start", pid=os.getpid(),
                                 updated=datetime.now().isoformat(timespec="seconds")))
    torch.set_num_threads(args.threads)

    t_data = time.time()
    train_txt, y_train, val_txt, y_val, test_txt, y_test = load_emotion(args.limit_train)
    if y_test is None:
        raise RuntimeError("emotion test split not found in emotion.csv")
    log(f"emotion train={len(train_txt)} val={len(val_txt)} test={len(test_txt)} "
        f"({time.time() - t_data:.1f}s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(torch.float32).to(device).eval()
    log(f"{MODEL_ID} on {device}, fp32, hidden={model.config.hidden_size}, batch={BATCH}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    for mode in modes:
        train_path = os.path.join(emb_dir, f"{mode}_train.npy")
        val_path = os.path.join(emb_dir, f"{mode}_val.npy")
        X_train = embed_split(mode, train_txt, tokenizer, model, device, train_path,
                              "train", status_path)
        X_val = embed_split(mode, val_txt, tokenizer, model, device, val_path,
                            "val", status_path)
        test_path = os.path.join(emb_dir, f"{mode}_test.npy")
        X_test = embed_split(mode, test_txt, tokenizer, model, device, test_path,
                             "test", status_path)
        if (X_train.shape[0] != len(y_train) or X_val.shape[0] != len(y_val)
                or X_test.shape[0] != len(y_test)):
            raise RuntimeError(f"embedding/row mismatch for {mode}: "
                               f"{X_train.shape} / {X_val.shape} / {X_test.shape}")
        write_json(status_path, dict(stage="embedding_done", mode=mode,
                                     updated=datetime.now().isoformat(timespec="seconds")))
        results.append(run_mode(mode, X_train, y_train, X_val, y_val, X_test, y_test,
                                args.trials, args.threads, outdir, status_path))
        write_json(os.path.join(outdir, f"xgboost_emotion_results_{tag}.json"),
                   dict(config=dict(modes=modes, trials=args.trials, limit_train=args.limit_train,
                                    n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
                                    threads=args.threads, model=MODEL_ID, num_class=NUM_CLASS,
                                    max_bin=MAX_BIN),
                        results=results))
        write_md(results, dict(threads=args.threads, n_train=len(y_train), n_val=len(y_val),
                               n_test=len(y_test)),
                 os.path.join(outdir, f"xgboost_emotion_results_{tag}.md"))
    write_json(status_path, dict(stage="done", updated=datetime.now().isoformat(timespec="seconds")))
    log("DONE")


if __name__ == "__main__":
    main()
