# Reproduction test — clean environment

Run during the final pre-publication audit. The candidate CSV was **not** modified, no tracked file
was modified, and nothing was published.

## Environment (isolated, outside the repository)

| item | value |
|---|---|
| venv | `<temp>/mtah_repro_venv` |
| code copy | `<temp>/mtah_repro_project` |
| interpreter | Python **3.11.0** (CPython, 64-bit, MSC v.1933) |
| packages installed | **numpy 2.2.6**, **pandas 3.0.6** — nothing else was requested |
| transitive (pulled by pandas) | python-dateutil 2.9.0.post0, six 1.17.0, tzdata 2026.5 |
| installation | `python -m venv <venv>` then `pip install numpy==2.2.6 pandas==3.0.6` |
| global environment | untouched |

The venv contains exactly the two packages pinned in `requirements-submission.txt`; no other
third-party package was installed.

## What was run

The code copy holds `games.py`, `starter/*.py`, `evaluation/possession_metrics.py`,
`evaluation/validate_submission.py` and `evaluation/gt/BAR-ATH_possession_gt_public.csv`. The match
bundle was **not** copied: `MTAH_DATA_DIR` pointed at the original `data/`, so the ATD, the tracking
XMLs and the 7 MB feature cache were read from the original location and nothing under `data/` was
written or deleted.

```bash
# from the temp copy, with MTAH_DATA_DIR set to the original data folder
python starter/experiments.py --only hybrid_match_bounds --no-baseline
```

Output went to the temp copy's `evaluation/results/`, because `experiments.py` derives its output
directory from the project root it lives in. The original `evaluation/results/` was never a target.

## Result

| | MD5 | bytes |
|---|---|---|
| produced in the clean venv | `1D660D34E320E2388C6373A4E4FF796F` | 1 882 093 |
| `evaluation/results/hybrid_match_bounds.csv` (untouched) | `1D660D34E320E2388C6373A4E4FF796F` | 1 882 093 |

**BYTE-IDENTICAL**, exit code 0. Reproduced public metrics from the run's own output: `spellF1`
0.507, `macroF1` 0.690, `F1 D` 0.427 (precision 0.682, recall 0.311), 13 spells, 303 whole-match
possessions at 13.0 s mean.

Validator on the produced file:

```
OK    hybrid_match_bounds.csv
      199319 rows, frame 0..199318
      H 58088 (29.1%)  A 40451 (20.3%)  D 100780 (50.6%)  X 0 (0.0%)
      reports 303 possessions of mean 13.0 s
Valid.
```

## Repository state after the test

- `evaluation/results/hybrid_match_bounds.csv` — MD5 unchanged (`1D660D34…`).
- `evaluation/results/MG_TEAM.csv` (named `TEAM_NAME_PLACEHOLDER.csv` during the test) — MD5
  unchanged, still identical to the candidate.
- `git status`: **0 tracked files modified**.
- `evaluation/reports/experiments.csv` — untouched (110 lines); the test's own log row was written
  into the temp copy.

## Two facts this test established about the pipeline

1. **`evaluation/results/` must exist.** `write_submission()` opens the output path without creating
   its folder, so a project copy without that directory fails with `FileNotFoundError`. A normal
   clone ships the folder (and a `.gitignore` rule for its contents), so this only affects a hand-made
   copy — which is what the first attempt of this test hit.
2. **The Smart Tagging XML is read unconditionally**, at `starter/experiments.py:224`, before the
   variant is selected — even though `hybrid_match_bounds` sets `respect_halves` and never uses a
   tag. A data folder without `BAR-ATH_tactical_smart_tagging.xml` fails with an `IOError` naming it.
