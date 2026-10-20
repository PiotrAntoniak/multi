# E1/E1b audit report — 2026-10-19

Scope: `C:\Users\pedro\Documents\random_projects\multi\EXPERIMENTS.md` lines 127-137 (E1/E1b).
Read-only audit of the repository. No repository source file was edited. Importing the helpers refreshed
the ignored `__pycache__/lang_shap_dims.cpython-313.pyc`. The only intentional writes are this report and
`E1_E1b.md` under `C:\Users\pedro\AppData\Local\Temp\opencode\multi_experiments_parts\`.

## Verified

1. **Missing scripts.** `langdims_probe.py` and `lang_inlp.py` are absent from the working tree, from
   `main`/`origin/main`, from every commit reachable by `git rev-list --all`, and from the rest of
   `C:\Users\pedro\Documents\random_projects`. `git log --all --oneline -- langdims_probe.py` and
   `git log --all --oneline -- lang_inlp.py` return no output. The pickaxe (`git log --all -S`) finds only
   prose references in `EXPERIMENTS.md`, `dim_removal.py`, `mel_close.py`, `agg_pr_2models.py`, and
   `agg_pr2.*`; it never finds a script body.

2. **Provenance of the entry.** Commit `21ab888` (`2026-10-12 08:05:54 +0200`, "Cross-lingual study ...")
   added `EXPERIMENTS.md` with the E1/E1b table and the citations, plus `lang_shap_dims.*`,
   `lang_neutralize.py`, `mel_close.py`, and `agg_pr_2models.py`. It did **not** add `langdims_probe.py`,
   `lang_inlp.py`, or any `lang_inlp.*` output. The entry was therefore unsupported from its first commit.

3. **Surviving INLP artifact.** `lang_inlp_mel.npz` exists (12,552 bytes; mtime 2026-10-12 04:55:04).
   It contains one key, `basis`, shape `(2, 768)`, dtype `float64`. The rows are orthonormal
   (`||V V^T - I|| = 1.1e-16`). It is untracked: `git status --short --ignored -- lang_inlp_mel.npz`
   prints `!! lang_inlp_mel.npz`; `.gitignore` contains `*.npz`.

4. **η² ranking and k=557.** On the 2006 kept FLORES rows (the raw `embeddings/mean/emb_*.npy` arrays are
   2009 rows; `train_flores12.load_sent_embeddings` slices out rows 997-999), `lang_neutralize.eta2_per_dim`
   gives `total_eta2 = 255.83565478262994`, and `top_dims(e2, 0.90)` gives `k = 557`. This matches
   `lang_shap_dims.json` (`datasets.sent.part_a.k_cum["90"] = 557`,
   `total_eta2 = 255.83565478262994`). The unsliced 2009-row matrices give `total_eta2 = 255.84003839507932`.

5. **Claim 1 (probe 1.000).** Stored `lang_shap_dims.json` has
   `datasets.sent.part_a.probe.full = 1.0`. A re-run with the capped, non-stratified `language_probe`
   (2006 rows, cap 2000, seed 0) gives `1.0`. On a stratified 70/30 split, seeds 1, 3, and 5 give
   `1.00000`; seeds 0 and 2 give `0.99958`; seed 4 gives `0.99917`.

6. **Claim 2 (top-557 zeroed -> 0.999).** Re-run with a stratified 70/30 split and
   `StandardScaler + LogisticRegression(C=1.0, max_iter=300)` on the 2006 kept rows: seed 5 gives
   `0.99917` at k=557 (`0.999` at 3 dp); seeds 0, 1, 3, and 4 give `0.99958` (`1.000` at 3 dp);
   seed 2 gives `1.00000`. The table's `0.999` is reproduced for seed 5. The k=557 count is exact, and
   the qualitative result (the probe survives zeroing 90% of the η² mass) is reproducible.

7. **Claim 3 (zero all 768 -> 0.250).** With all 768 columns set to zero, the probe predicts one class;
   its test accuracy is the test-set class frequency. On a stratified 70/30 split of the 2006 rows
   (2408 test rows; counts 602/602/602/602) it is exactly `0.25`. The nominal chance for 4 balanced
   classes is exactly 0.25. This claim is correct as a nominal chance value and the measured value
   agrees exactly on this split.

8. **Claim 4 is not supported by any surviving artifact.** See "Discrepancies" item 4.

## Discrepancies

1. **The script citation is false.** The entry cites `langdims_probe.py` and `lang_inlp.*` as if they were
   repository artifacts. Neither script was ever committed. `EXPERIMENTS.md` has been carrying this
   citation since `21ab888`, and `REPRODUCE.md` has no E1/E1b section at all (it lists E14b, E15, E16,
   E17, E18, E19 only).

2. **No cited `lang_inlp.*` output survives.** `lang_inlp.npz`, `lang_inlp_mean.npz`,
   `lang_inlp_agg.npz`, and `lang_inlp_agg_610.npz` are all absent. The only INLP cache in the tree is
   the mel-only `lang_inlp_mel.npz`. `mel_close.py:25-26` explicitly says
   "`lang_inlp_mean.npz` [is] absent" and fits its own mel-only basis. `agg_pr_2models.py:18-22` says the
   210m `lang_inlp_agg.npz` was produced by `crosslingual_topic_agg.run_inlp` (itself absent), and
   `agg_pr2.json:44` describes that basis as "3 dirs". Neither the basis nor the producing script is in git.

3. **The exact E1 triple is split-sensitive.** The table's `1.000 / 0.999 / 0.250` is not uniquely
   determined by the surviving data. A seed sweep with a stratified 70/30 split on the 2006 kept rows gives:
   - seed 0: full `0.99958`, k=557 `0.99958`
   - seed 1: full `1.00000`, k=557 `0.99958`
   - seed 2: full `0.99958`, k=557 `1.00000`
   - seed 3: full `1.00000`, k=557 `0.99958`
   - seed 4: full `0.99917`, k=557 `0.99958`
   - seed 5: full `1.00000`, k=557 `0.99917`
   Zero-all is `0.25000` for every seed. Seed 5 reproduces the table exactly at 3 dp; the other seeds
   agree at 2 dp. The missing script's split/seed is unknown.

4. **Claim 4 (`~30` directions to chance) conflicts with the surviving evidence.** There are three
   independent problems:
   - `backbone_grid.py:118` sets `INLP_MAX_DIRS = 30`. `_fit_inlp` (`backbone_grid.py:452-484`) stops
     only when `sum(q.shape[0] for q in Qs) >= 30` or when a round returns no directions. It never
     measures probe accuracy. `backbone_grid.json` reports `inlp_dirs = 30` for FLORES-sent and
     FLORES-agg on both 210m and 610m. That means the 30-direction cap was hit, not that the probe
     reached chance. `backbone_grid.md` calls the method "inlp (~30-dir)" because that is the configured
     cap, not because 30 directions were shown to be sufficient.
   - The only surviving run that explicitly measures "reached chance" is `agg_pr2.md:22`: FLORES-agg
     610m, `n_removed = 3`, `rounds = 1`, `final probe acc = 0.26553625340262316`,
     `reached chance(<=0.30) = True`. That is 3 directions, not about 30. The 210m basis
     (`lang_inlp_agg.npz`) is also described as 3 directions, but the file is missing.
   - The surviving mel basis has 2 directions. Projecting it out of the MultiEURLEX en/de/fr mean pool
     leaves a held-out 3-way probe at `0.5455555555555556` (chance 0.3333), not at chance. So the one
     cached INLP artifact does not demonstrate collapse.

5. **Two surviving probe protocols disagree at k=557.** `lang_shap_dims.language_probe` (2006 rows,
   cap 2000, non-stratified 30% split, seed 0) gives `full = 1.0`, `k557 = 1.0`, `k625 = 0.9975`, and
   `zero-all = 0.2408333`; it reproduces the stored JSON exactly. The E1 table's `0.999 / 0.250` instead
   requires a stratified 70/30 split (seed 5). Neither protocol is documented in the entry, so the exact
   source of the table triple is ambiguous. The two protocols are not interchangeable.

6. **The older E1 bullet had two more unsupported numbers.** The first committed version of the entry
   (`21ab888:EXPERIMENTS.md`) also claimed "60-72% variance retained" after INLP. No artifact stores a
   variance-retention value. The current table dropped that number, which is the right call, but the
   number still appears in the historical record.

7. **`dim_removal.py` propagates the unverified prose.** Lines 404-408 hard-code
   "a 4-language linear probe is 1.000 ... still 0.999 after zeroing the top-557 eta^2 dims (E1/E17)".
   A reader of `dim_removal.md` sees the E1 numbers presented as context, not as UNK.

## Bugs / risks

1. **The only INLP cache is untracked and git-ignored.** `lang_inlp_mel.npz` is matched by `*.npz` and
   has no committed copy. A clean checkout loses it. It is also not the artifact the entry needs
   (MultiEURLEX, not FLORES mean).
2. **`backbone_grid._fit_inlp` has no accuracy stop.** A returned 30-row basis is a cap hit. Any downstream
   sentence that reads `inlp_dirs = 30` as "needed 30 directions to reach chance" is wrong.
3. **`mel_close.fit_inlp` has no chance check.** It stores `{"source": "fit", "n_removed": 2,
   "rounds": [2]}`. One round produced a rank-2 basis; the loop then exited. The code does not record the
   probe accuracy at that point. A held-out probe on the same pool still reaches 0.546 after projection.
4. **`agg_pr_2models.compute_inlp` uses a loose chance threshold.** For 4 classes, chance is 0.25; the stop
   rule is `acc <= 0.30` (`INLP_STOP_ACC`), i.e. chance + 0.05. The reported 0.2655 passes, but the rule
   would also pass a probe that is still above chance.
5. **`dim_removal.py` restates E1 as fact.** This is the main vector by which an UNK result becomes
   "verified-looking" in a derived report.
6. **`lang_shap_dims.py` is not a drop-in reproduction of the missing probe.** Its `language_probe`
   caps each language at 2000 rows and uses a non-stratified permutation split, whereas the E1 triple
   needs a stratified split (or a lucky seed) to match. Treating one as the other overstates provenance.
7. **No E1/E1b entry in `REPRODUCE.md`.** The reproduction guide omits the experiment entirely, so the
   repository has no runnable path for it.
8. **Title says DONE.** The current heading is `DONE` with no verification qualifier, despite the missing
   script and the unverified INLP claim.

## Missing scripts

| path | status | evidence |
|---|---|---|
| `langdims_probe.py` | never in git | empty `git log --all --`; absent from all trees; no copy under `random_projects` |
| `lang_inlp.py` | never in git | same; cited only in prose (`agg_pr_2models.py:21-22`) |
| `lang_inlp.npz` | no artifact | no file; no git history |
| `lang_inlp_mean.npz` | no artifact | `mel_close.py:25-26` says absent |
| `lang_inlp_agg.npz` | no artifact | `agg_pr_2models.py:18`, `agg_pr2.json:44` refer to it; file absent |
| `lang_inlp_agg_610.npz` | no artifact | `agg_pr_2models.py:85` writes it; file absent |
| `lang_inlp.json` / `lang_inlp.md` | no artifact; never in git | cited only as `lang_inlp.*`; no file, no git history |
| `crosslingual_topic*.py` | never in git | cited by EXPERIMENTS E3/E6/E7; absent from all trees |
| `crosslingual_inlp.py` | never in git | cited by EXPERIMENTS E3/E6/E7; absent |
| `pc_ablation*.py` | never in git | cited by EXPERIMENTS E12/E13; absent |
| `translate_datasets.*` | never in git | cited by EXPERIMENTS E2; absent |
| `optuna_inlp_compare.*` | never in git | cited by EXPERIMENTS E8; absent |

The last five rows are outside E1/E1b scope. They show the same provenance pattern in other sections;
see the follow-up suggestion at the end.

## Recommended changes

1. **Replace the E1/E1b section** with `E1_E1b.md` (UNK script line; claims 1-3 VERIFIED, claim 4
   UNVERIFIED; PARTIAL verdict; 0.975 = N/A).
2. **Change the title qualifier** from `DONE` to `DONE (partially verified 2026-10-19)` or `UNK`. Do not
   leave a bare `DONE`.
3. **Stop citing `langdims_probe.py` / `lang_inlp.py` as if they exist.** Either restore them with the
   exact protocol and commit them, or remove the citations and keep the UNK line. Do not reconstruct
   them from memory without labelling the reconstruction.
4. **If re-running the probe**, commit a small JSON with: embedding paths, row counts, split type and
   seed, probe `C`/`max_iter`, k25/k50/k80/k90/k95, full and zero-top-k accuracies, and the zero-all
   baseline. State the seed sweep (the third decimal moves).
5. **If re-running INLP**, use an explicit accuracy stop (e.g. `acc <= chance + 0.05`), record the
   per-round accuracy curve and the number of directions at the stop, and commit the basis plus its shape
   and a hash. Do not call a 30-direction cap hit "reached chance".
6. **Replace the "~30 dirs to chance" wording** in EXPERIMENTS with: "30-direction cap; chance reached in
   3 directions on FLORES-agg 610m; FLORES-sent not measured." Add the 3-direction result only if it is
   clearly labelled as a different dataset/pool.
7. **Fix `dim_removal.py:404-408`.** Remove the hard-coded E1 numbers or mark them "E1, unverified" until
   the scripts are restored.
8. **Add E1/E1b to `REPRODUCE.md`** only after a runnable script exists. Until then, list it under a
   "not reproducible (missing script)" heading.
9. **Resolve the untracked artifact.** Either force-add `lang_inlp_mel.npz` with documented provenance,
   or delete it and regenerate it from a committed script. Do not cite it as evidence for a FLORES claim.
10. **Do not use the 0.975 bar for this experiment.** It has no reference ratio; the correct value is N/A.

## Commands run

All commands ran in `C:\Users\pedro\Documents\random_projects\multi` with PowerShell 5.1 and
`C:\Users\pedro\anaconda3\python.exe`. No repository source file was written; `git status` shows no
tracked changes. Python refreshed the ignored `__pycache__/lang_shap_dims.cpython-313.pyc` on import.

Git and filesystem checks:
```
git log --all --oneline -- langdims_probe.py
git log --all --oneline -- lang_inlp.py
git log --all --oneline -- lang_inlp.npz
git log --all --oneline -- lang_metrics.py
git log --all --oneline -- lang_neutralize.py
git log --all --oneline -- dim_removal.py
git rev-list --all | ForEach-Object { git ls-tree -r --name-only $_ } | Sort-Object -Unique
git show --stat 21ab888
git show 21ab888:EXPERIMENTS.md
git show b16a962:EXPERIMENTS.md
git grep -n -e "langdims_probe" -e "lang_inlp" 21ab888
git log --all --oneline -S "langdims_probe"
git log --all --oneline -S "lang_inlp"
git log --all --oneline -S "top-557"
git status --short --ignored -- lang_inlp_mel.npz
git ls-files "*.npz"
Get-Content .gitignore
Get-ChildItem -Recurse -File -Filter "langdims_probe.py"
Get-ChildItem -Recurse -File -Filter "lang_inlp*.py"
```

Artifact and data checks:
```
python -c "import numpy as np; d=np.load('lang_inlp_mel.npz', allow_pickle=True); print(d.files, d['basis'].shape, d['basis'].dtype)"
python -c "import json; d=json.load(open('lang_shap_dims.json')); ..."   # printed sent/agg/mel part_a
python -c "import json; d=json.load(open('mel_close.json')); print(d['transform_meta']['inlp'])"
python -c "import json; d=json.load(open('backbone_grid.json')); ..."    # printed inlp_dirs per cell
```

Reproduction runs (PowerShell here-string piped to `python -`; no temp script written):
```
# 1. eta^2 and k=557 on the 2006 kept rows
@'
from lang_neutralize import eta2_per_dim, top_dims
... 4 x 2006 x 768 mean embeddings (2009 raw sliced to kept rows) ...
'@ | python -
# output: k90 557, total_eta2 255.83565478262994

# 2. surviving lang_shap_dims.language_probe (2006 rows, cap 2000, seed 0)
# output: full 1.0; k557 1.0; k625 0.9975; all 0.24083333333333334

# 3. stratified 70/30, seed 5, 2006 rows/lang
# output: full 1.00000; k557 0.99917; all 0.25000

# 4. seed sweep 0-5, stratified 70/30, 2006 rows/lang
# output:
# 0 full 0.99958 k557 0.99958 all 0.25000
# 1 full 1.00000 k557 0.99958 all 0.25000
# 2 full 0.99958 k557 1.00000 all 0.25000
# 3 full 1.00000 k557 0.99958 all 0.25000
# 4 full 0.99917 k557 0.99958 all 0.25000
# 5 full 1.00000 k557 0.99917 all 0.25000

# 5. mel INLP basis check
# output: basis (2, 768); orthonorm err 1.1102230246251565e-16
#         probe_full 1.0; probe_after_2_dirs 0.5455555555555556
```

## Suggested follow-up subagent

The repo-wide provenance pattern (experiment entries citing scripts that were never committed) is larger
than E1/E1b. If the orchestrator wants it quantified, spawn one subagent:

- **Type:** `explore-high`
- **Prompt:** "In `C:\Users\pedro\Documents\random_projects\multi`, audit every other section of
  `EXPERIMENTS.md` (E2-E32, excluding E1/E1b) against git. For each cited script, report: exists in working
  tree, exists in any commit (`git log --all -- <path>`), and whether its cited output files exist. Produce
  a table of `section | cited script | in HEAD | in git history | output exists | verdict`, and list every
  never-committed script. Do not edit repo files; write the report to
  `C:\Users\pedro\AppData\Local\Temp\opencode\multi_experiments_parts\provenance_audit.md`."
- **Why:** E1/E1b is not an isolated case. `crosslingual_topic*.py`, `crosslingual_inlp.py`,
  `pc_ablation*.py`, `translate_datasets.*`, and `optuna_inlp_compare.*` are also cited but absent from
  every commit. A repo-wide table would tell the orchestrator which other sections need the same UNK
  treatment, without duplicating this E1 audit.
