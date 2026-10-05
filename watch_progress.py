"""Minute-refresh progress board for the four single-mode XGBoost jobs.

Reads the per-mode status JSONs and trials CSVs of xgboost_sst2.py / xgboost_emotion.py and
rewrites `progress_live.txt` every 60 s (own pid in `watch_progress.pid`). Run hidden:
  wscript //B //Nologo launch_hidden.vbs "cmd /c <python> -u watch_progress.py > watch_progress_run.log 2>&1"
"""
import csv
import json
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
JOBS = [("sst2", "mean"), ("sst2", "bos"), ("emotion", "mean"), ("emotion", "bos")]


def read_status(ds, mode):
    p = os.path.join(BASE, f"xgboost_{ds}_status_{mode}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def from_csv(ds, mode):
    p = os.path.join(BASE, f"xgboost_{ds}_trials_{mode}.csv")
    if not os.path.exists(p):
        return None, None, 0
    best = last = None
    n = 0
    try:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    v = row.get("value_auc") or row.get("value_accuracy")
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                n += 1
                last = v
                best = v if best is None else max(best, v)
    except OSError:
        pass
    return best, last, n


def main():
    with open(os.path.join(BASE, "watch_progress.pid"), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    while True:
        lines = [f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                 f"{'job':<14} {'stage':<15} {'trial':<8} {'last_acc':<9} {'best_acc':<9} {'done':<5}"]
        for ds, mode in JOBS:
            st = read_status(ds, mode)
            stage = st.get("stage", "?") if st else "not started"
            trial = "-"
            if st and st.get("stage") == "optuna":
                trial = f"{st.get('trial', '-')}/{st.get('trials', '-')}"
            elif st and st.get("stage") == "embedding":
                trial = f"batch {st.get('batch', '-')}"
            best, last, n = from_csv(ds, mode)
            lines.append(f"{ds + ' ' + mode:<14} {stage:<15} {trial:<8} "
                         f"{(f'{last:.4f}' if last is not None else '-'):<9} "
                         f"{(f'{best:.4f}' if best is not None else '-'):<9} {n:<5}")
        with open(os.path.join(BASE, "progress_live.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
