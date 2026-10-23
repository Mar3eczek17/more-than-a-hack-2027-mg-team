# Publishing checklist (nothing here has been executed)

This file is a plan. No commit, push, remote change or publication was performed while preparing the
submission, and the current clone must not be used to publish anything: its `origin` is the
organisers' repository (`github.com/metrica-sports/more-than-a-hack-2027.git`), so a `git push` here
would attempt to write to their repository.

Team name: **`MG_TEAM`** is the name used for the prepared files. It must match the name stated in the
registration form before anything is uploaded; nothing in this repository can verify that match.

---

## 1. What must go into the team's own public repository

The repository has to let someone with the data bundle regenerate the submission exactly
(`SUBMISSION.md`, "A public code repository"; `RULES.md` §3). That means the team's files **and** the
organisers' modules they import:

**Team code and documentation**

- `starter/tracking.py` — ATD reading, feature cache, ball speed, gaps, out-of-play
- `starter/possession_model.py` — evidence, dead-ball mask, team vote, boundary refinement
- `starter/experiments.py` — the variant runner; the documented command lives here
- `starter/smart_tagging.py` — the parser for the experiment that was rejected
- `starter/explore_features.py`, `starter/explore_smart_tagging.py` — the calibration tables
- `starter/sensitivity.py`, `starter/diagnostics.py`, `starter/full_match_report.py` — audits
- `starter/README_possession_model.md` — the technical document (or its `submission/` copy)
- `requirements-submission.txt` — pinned `numpy==2.2.6`, `pandas==3.0.6`

**Organisers' files the pipeline imports (already public, keep them)**

- `games.py` — pitch size, half boundaries, asset paths
- `evaluation/possession_metrics.py` — imported by `experiments.py` for scoring
- the rest of the public skeleton (`starter/baseline.py`, `README.md`, `DATA.md`, …) if the repository
  is a clone rather than a fresh one, which is the simplest route

**Deliverables uploaded through the form, not committed as predictions**

- `evaluation/results/MG_TEAM.csv` → upload as `<team_name>.csv` if the form name differs from `MG_TEAM`
- `submission/MG_TEAM.md` → `<team_name>.md` (or convert to `<team_name>.pdf`)

## 2. What must never be published

| path | why |
|---|---|
| `data/**` — everything under it | gated bundle: ATD tracking, homography, both videos, Smart Tagging XML. `RULES.md`: do not redistribute the data |
| `data/tactical/_cache/*.npz` | a frame-level tracking derivative; exactly the "derived data that would reconstruct it" the rules forbid |
| `*.mp4` | the match video itself |
| `evaluation/results/*.csv` | one row per frame for the whole match. A perfect one is a copy of the ground truth; the candidate is a prediction. Upload it through the form, keep it out of git |
| `evaluation/reports/**` | per-variant metrics and full-match audits. Derived from the public ground truth, so not a leak — but generated output, and `.gitignore` already excludes it |
| non-public `*_possession_gt.csv`, `*_possession_gt_meta.json` | the private full-match ground truth; `.gitignore` covers them, never force-add |
| `submission/*.csv` | same reason as `evaluation/results/*.csv` |

## 3. The `git add -A` trap

`.gitignore` ignores `evaluation/results/*.csv` **but re-allows `baseline_*.csv`** on purpose, so a
fresh clone has something to score. This working tree holds **24 `baseline_smooth*.csv` files,
45,170,232 bytes in total** (verified on disk). Three of them ship with the repository; the other 21
were generated here and account for about **39.5 MB**. They are predictions for the whole match,
derived from the ATD alone, so they are not a rule violation — but a blanket add would publish them
and inflate the repository for no benefit.

Do not use `git add -A`. Add files explicitly, by name:

```bash
git add starter/tracking.py starter/possession_model.py starter/experiments.py \
        starter/smart_tagging.py starter/explore_features.py starter/explore_smart_tagging.py \
        starter/sensitivity.py starter/diagnostics.py starter/full_match_report.py \
        starter/README_possession_model.md requirements-submission.txt \
        submission/MG_TEAM.md submission/PUBLISH_CHECKLIST.md submission/REPRODUCTION_TEST.md
git status --short          # read it before committing
```

A `.gitignore` line for `evaluation/results/baseline_smooth*.csv` would also fix it, if the team
prefers to keep the blanket habit.

## 4. Order of operations, only after the team's own repository exists

1. Create the team's public repository. Do **not** reuse this clone's `origin`.
2. Copy the files listed in section 1 into it, keeping the layout (`games.py` at the root, `starter/`
   and `evaluation/` as siblings).
3. Check that the data is absent: `git ls-files data` must print only `data/README.md`.
4. Install and regenerate from that repository on a clean machine:
   `pip install -r requirements-submission.txt`, then
   `python starter/experiments.py --only hybrid_match_bounds --no-baseline`.
   Expected: `evaluation/results/hybrid_match_bounds.csv`, 1 882 093 bytes, MD5
   `1d660d34e320e2388c6373a4e4ff796f`.
5. Validate the exact file to be uploaded:
   `python evaluation/validate_submission.py evaluation/results/MG_TEAM.csv`.
6. Only then set the remote, commit and push — by hand, after reading `git status`.

The repository's `.gitignore` ignores `*.pdf`. If the technical document is published as a PDF, add it
explicitly (`git add -f <team_name>.pdf`) or keep the Markdown version.

## 5. Before the form is submitted

- [ ] `MG_TEAM` confirmed against the registration form; CSV named `<team_name>.csv` and document
      named `<team_name>.md` / `<team_name>.pdf` accordingly.
- [ ] The uploaded CSV is the validated one (same MD5 as above), not an earlier build.
- [ ] The repository URL in the form is the team's public repository and opens in a private browser
      window without logging in.
- [ ] The repository contains no `data/`, no `.npz`, no `.mp4` and no prediction CSV.
- [ ] The technical document is attached as `.pdf` or `.md`.
- [ ] Team size is **3-5 members** — the eligibility rule in `RULES.md`; smaller teams are not
      eligible for the Innovation Challenge and the Final Pitch.
- [ ] One form response per team, and it is the last one before the deadline (17 November 2026).
