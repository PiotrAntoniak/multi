"""XGBoost on EuroBERT-210m sentence embeddings for SST-2 (GLUE).

Two pooling modes of the LAST hidden state of `EuroBERT/EuroBERT-210m`
(native `AutoModel` + `AutoTokenizer`; remote code is broken on transformers 5.x):

  * mean : masked mean over all non-pad tokens of the PLAIN forward (the
           tokenizer appends the end token, which is included).
  * bos  : position 0 of the last hidden state of a forward over the PLAIN ids
           with special id 128000 manually PREPENDED (+ a leading 1 mask).

SST-2: train 67,349 rows (`sentence`, `label`), validation 872 rows. `--limit-train N`
keeps the first N train rows (smoke only). Embeddings are cached to
`<outdir>/sst2_emb/{mode}_train.npy` + `_val.npy` (float32) and reused if present.

XGBoost (`tree_method=hist`, `max_bin=64`) with a baseline (max_depth=6,
learning_rate=0.1, n_estimators=300) and an Optuna TPE (seed 0) search identical to
`xgboost_topic_optuna.py::_optuna_search_space`; objective = validation ACCURACY at
threshold 0.5, trained on the train split. Reports validation accuracy, macro-F1 and AUC.

Usage: python xgboost_sst2.py --modes mean,bos --trials 50 --limit-train 0 --outdir . --threads 4
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "EuroBERT/EuroBERT-210m"
BOS_ID = 128000
BATCH, MAX_LEN = 256, 512
MAX_BIN = 64
SEED = 0
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
    p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
             max_bin=MAX_BIN, n_jobs=threads, seed=SEED)
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
def load_sst2(limit_train):
    local = os.path.join(BASE, "sst2.csv")
    if os.path.exists(local):
        import pandas as pd
        df = pd.read_csv(local)
        train, val = df[df["split"] == "train"], df[df["split"] == "validation"]
        if limit_train and limit_train > 0:
            train = train.head(limit_train)
        log(f"loaded {local}: train={len(train)} val={len(val)}")
        return (list(train["sentence"]), np.asarray(train["label"], dtype=np.int64),
                list(val["sentence"]), np.asarray(val["label"], dtype=np.int64))
    # fallback: Hub (datasets 4.x dropped script datasets; `glue` alias 404s -> nyu-mll/glue)
    try:
        ds = load_dataset("glue", "sst2")
    except Exception as e:  # noqa: BLE001
        log(f"load_dataset('glue','sst2') failed ({type(e).__name__}); using 'nyu-mll/glue'")
        ds = load_dataset("nyu-mll/glue", "sst2")
    train, val = ds["train"], ds["validation"]
    if limit_train and limit_train > 0:
        train = train.select(range(min(limit_train, len(train))))
    return (list(train["sentence"]), np.asarray(train["label"], dtype=np.int64),
            list(val["sentence"]), np.asarray(val["label"], dtype=np.int64))


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
    pred = (probs >= 0.5).astype(int)
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                auc=float(roc_auc_score(y, probs)))


# ---------------------------------------------------------------------------
# One mode
# ---------------------------------------------------------------------------
def run_mode(mode, X_train, y_train, X_val, y_val, trials, threads, outdir, status_path):
    log(f"===== MODE {mode}: baseline + Optuna TPE {trials} trials =====")

    t0 = time.time()
    base = xgb.XGBClassifier(**make_params(threads, BASELINE_PARAMS))
    base.fit(X_train, y_train)
    base_m = metrics(y_val, base.predict_proba(X_val)[:, 1])
    log(f"[{mode}] baseline acc={base_m['accuracy']:.4f} macro_f1={base_m['macro_f1']:.4f} "
        f"auc={base_m['auc']:.4f} ({time.time() - t0:.1f}s)")

    trials_csv = os.path.join(outdir, f"xgboost_sst2_trials_{mode}.csv")

    def objective(trial):
        t = time.time()
        params = _optuna_search_space(trial)
        clf = xgb.XGBClassifier(**make_params(threads, params))
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        acc = float(accuracy_score(y_val, (probs >= 0.5).astype(int)))
        trial.set_user_attr("macro_f1", float(f1_score(y_val, (probs >= 0.5).astype(int),
                                                       average="macro", zero_division=0)))
        log(f"[{mode}] trial {trial.number + 1}/{trials} acc={acc:.4f} "
            f"macro_f1={trial.user_attrs['macro_f1']:.4f} ({time.time() - t:.1f}s) {params}")
        done = [t.value for t in study.trials if t.value is not None]
        write_json(status_path, dict(stage="optuna", mode=mode, trial=trial.number + 1,
                                     trials=trials, value_acc=acc,
                                     best=max(done) if done else None,
                                     updated=datetime.now().isoformat(timespec="seconds")))
        return acc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))

    def save_trials(_study, _trial=None):
        rows = []
        for tr in _study.trials:
            if tr.value is None:
                continue
            row = {"trial": tr.number, "value_accuracy": tr.value, "state": tr.state.name}
            row.update(tr.params)
            rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).sort_values("trial").to_csv(trials_csv, index=False)

    study.optimize(objective, n_trials=trials, callbacks=[save_trials])
    best_params = dict(study.best_params)
    log(f"[{mode}] BEST trial #{study.best_trial.number}: acc={study.best_value:.4f}")
    log(f"[{mode}] BEST params: {best_params}")

    tuned = xgb.XGBClassifier(**make_params(threads, best_params))
    tuned.fit(X_train, y_train)
    tuned_m = metrics(y_val, tuned.predict_proba(X_val)[:, 1])
    log(f"[{mode}] tuned acc={tuned_m['accuracy']:.4f} macro_f1={tuned_m['macro_f1']:.4f} "
        f"auc={tuned_m['auc']:.4f}")
    return dict(mode=mode, baseline=base_m, tuned=tuned_m, best_params=best_params,
                best_trial=study.best_trial.number, best_value=float(study.best_value),
                n_trials=len([t for t in study.trials if t.value is not None]),
                trials_csv=os.path.basename(trials_csv))


def write_md(results, cfg, path):
    lines = ["# XGBoost on EuroBERT-210m SST-2 embeddings\n"]
    lines.append(f"Generated by `xgboost_sst2.py` — xgboost {xgb.__version__}, "
                 f"optuna {optuna.__version__}, transformers {__import__('transformers').__version__}, "
                 f"`tree_method=hist`, `max_bin={MAX_BIN}`, threads={cfg['threads']}, "
                 f"train={cfg['n_train']}, val={cfg['n_val']}.\n")
    lines.append("Embeddings: EuroBERT-210m native fp32, batch 256, GPU if available. "
                 "`mean` = masked mean of the plain forward (end token included); "
                 "`bos` = position 0 after prepending special id 128000. "
                 "Optuna TPE seed 0, objective = validation accuracy at threshold 0.5. "
                 f"Baseline = max_depth=6, learning_rate=0.1, n_estimators=300.\n")
    lines.append("## Validation (872 rows unless --limit-train)\n")
    lines.append("| mode | model | accuracy | macro-F1 | AUC | trial count |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for res in results:
        m = res["mode"]
        b, t = res["baseline"], res["tuned"]
        lines.append(f"| {m} | baseline | {b['accuracy']:.4f} | {b['macro_f1']:.4f} | {b['auc']:.4f} | - |")
        lines.append(f"| {m} | tuned | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['auc']:.4f} | {res['n_trials']} |")
    lines.append("")
    for res in results:
        lines.append(f"### Best Optuna parameters — {res['mode']}\n")
        lines.append("```")
        lines.append(f"best trial #: {res['best_trial']}")
        lines.append(f"best value (val accuracy): {res['best_value']:.4f}")
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
    emb_dir = os.path.join(outdir, "sst2_emb")
    os.makedirs(emb_dir, exist_ok=True)
    tag = args.modes.replace(",", "+")
    status_path = os.path.join(outdir, f"xgboost_sst2_status_{tag}.json")
    pid_path = os.path.join(outdir, f"xgboost_sst2_{tag}.pid")
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))
    log(f"pid={os.getpid()} modes={args.modes} trials={args.trials} limit_train={args.limit_train} "
        f"outdir={outdir} threads={args.threads}")
    write_json(status_path, dict(stage="start", pid=os.getpid(),
                                 updated=datetime.now().isoformat(timespec="seconds")))
    torch.set_num_threads(args.threads)

    t_data = time.time()
    train_txt, y_train, val_txt, y_val = load_sst2(args.limit_train)
    log(f"SST-2 train={len(train_txt)} val={len(val_txt)} ({time.time() - t_data:.1f}s)")

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
        if X_train.shape[0] != len(y_train) or X_val.shape[0] != len(y_val):
            raise RuntimeError(f"embedding/row mismatch for {mode}: {X_train.shape} vs {len(y_train)}")
        write_json(status_path, dict(stage="embedding_done", mode=mode,
                                     updated=datetime.now().isoformat(timespec="seconds")))
        results.append(run_mode(mode, X_train, y_train, X_val, y_val, args.trials,
                                args.threads, outdir, status_path))
        write_json(os.path.join(outdir, f"xgboost_sst2_results_{tag}.json"),
                   dict(config=dict(modes=modes, trials=args.trials, limit_train=args.limit_train,
                                    n_train=len(y_train), n_val=len(y_val), threads=args.threads,
                                    model=MODEL_ID, max_bin=MAX_BIN), results=results))
        write_md(results, dict(threads=args.threads, n_train=len(y_train), n_val=len(y_val)),
                 os.path.join(outdir, f"xgboost_sst2_results_{tag}.md"))
    write_json(status_path, dict(stage="done", updated=datetime.now().isoformat(timespec="seconds")))
    log("DONE")


if __name__ == "__main__":
    main()
