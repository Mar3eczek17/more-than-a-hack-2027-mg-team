# Possession model: what changed, and why

The baseline (`starter/baseline.py`) decides each frame from one frame: the nearest tracked player
to the ball wins, and the result is passed through a centred majority vote. That is the floor, and
its two weaknesses are separable.

1. **Dead ball is nearly never predicted.** The vote only writes `D` where *no* ball was tracked
   anywhere in a 225-frame window. Real stoppages are far shorter, so on the public five minutes
   the baseline scores `D` recall 0.097 at precision 0.947 -- it finds about a tenth of the dead
   ball. Since a possession ends at a stoppage, that one failure also misplaces most spell
   boundaries.
2. **Every turnover has to be expressed as a team flip.** With no notion of a stoppage, a
   possession that ends and restarts is one long predicted spell, or a spell split in the wrong
   place.

Everything here is in `starter/` and is scored with the repository's own evaluator, unchanged.

| file | role |
|---|---|
| `starter/tracking.py` | chunked ATD reader, feature cache, ball speed, gap, out-of-play |
| `starter/possession_model.py` | evidence, dead-ball mask, team vote, boundary refinement, HMM |
| `starter/explore_features.py` | the calibration tables the thresholds come from |
| `starter/experiments.py` | the experiment runner: writes each variant and scores it |

## The pipeline that won

`predict_hybrid`, four stages, each with a different job:

1. **Team signal** -- nearest player to the ball gives a team; a majority vote over +/-112 frames
   removes the frame-to-frame flicker. This is the baseline's own team signal, kept deliberately:
   it is unbiased over a few seconds even though it is noisy frame to frame.
2. **Dead-ball evidence** -- a per-frame score in [0, 1], the max of four cues that each mean
   something in football: the ball outside the pitch rectangle (98% of such frames are `D` in the
   public window), nobody within 8 m of a stationary ball, a tracking gap of 20-60+ frames that
   *opened with a slow ball* (a gap that opens at 8+ m/s is a pass in flight), and no player
   tracked at all. Evidence is averaged over 0.6 s first, because the cues are noisy at frame
   level and possession is not.
3. **Stoppage state** -- a Schmitt trigger over that evidence (enter at 0.55, leave below 0.30),
   with two constraints that the public window justifies: a stoppage must last at least 100 frames
   (4 s -- all four real stoppages there are 138-511 frames), and short holes inside a stoppage are
   closed. Predicted `D` is therefore a state, not a per-frame score.
4. **Boundary refinement** -- a majority vote lags every boundary by up to half its window, which
   on a 14 s possession is a fifth of its length and most of the IoU that `spellF1` measures. Each
   boundary is re-placed within +/-30 frames at the split that maximises the frame-level evidence
   for the two labels it separates. The vote decides the structure; the evidence decides the frame.

Three-state Viterbi over the same evidence is implemented too (`predict`, `viterbi`,
`transition_matrix`) and is reported for comparison. It loses badly -- 0.22 `spellF1` against 0.51
-- because per-frame emission log-odds are large enough (several nats when the ball is close to one
side) that the accumulated evidence of a contested second outweighs the transition prior, and the
match ends up fragmented into 900+ possessions of 4-5 s. The lesson is that the team signal needs
*seconds* of averaging, not a frame-level likelihood.

## Parameters and where they come from

Nothing is fitted to the public labels by search. The values are either physical (metres, m/s,
seconds), read off the public window's *segment structure* (how many segments, how long, how many
touch a stoppage), or taken from `explore_features.py`'s calibration table, which is the table that
decides whether a cue is worth using at all:

| cue | fires on | share of those frames that are `D` | coverage of `D` |
|---|---|---|---|
| ball outside the pitch | 106 frames | 98.1% | 7.1% |
| ball not tracked | 885 frames | 55.8% | 33.6% |
| ball speed 1-2 m/s | 856 frames | 34.1% | -- |
| ball speed > 8 m/s | 2245 frames | 5.3% | -- |
| nearest player > 12 m | 35 frames | 100% | -- |

That table is also why the pipeline gates the gap cue on speed rather than treating every gap as
dead ball: "no ball" is only 56% dead, but a gap that started with a slow ball is a different event
from one that started with a shot.

## Results on the public five minutes

Scored with `evaluation/run_evaluation.py` (the same code the organisers run). `full poss` is the
number of `H`/`A` runs over the whole 199,319-frame match and `full mean` their mean duration -- a
sanity check against football (a match has roughly 300-400 possessions of ~14 s), not a score.

| variant | spellF1 | macroF1 | D F1 | transitions | full poss | full mean |
|---|---|---|---|---|---|---|
| `baseline_smooth112` | 0.4727 | 0.594 | 0.175 | 0.162 | 459 | 12.4 s |
| hybrid, no dead overlay | 0.452 | 0.617 | 0.232 | 0.114 | 392 | 14.0 s |
| hybrid, dead overlay, no refinement | 0.497 | 0.675 | 0.395 | 0.211 | 373 | 13.1 s |
| **hybrid, dead overlay + refinement** | **0.507** | **0.690** | **0.427** | **0.229** | **333** | **14.0 s** |
| same, +-60 frame refinement | 0.477 | 0.685 | 0.431 | 0.222 | 334 | 14.0 s |
| same, minimum stoppage 50 frames | 0.438 | 0.705 | 0.483 | 0.308 | 354 | 12.7 s |
| HMM (emission + Viterbi) | 0.219 | 0.659 | 0.491 | 0.261 | 929 | 4.7 s |

The chosen variant is the one that improves `spellF1` **and** `macroF1` **and** the transition
metric **and** the full-match possession statistics at the same time. The 50-frame variant has
better `D` and better transitions but worse `spellF1`; the HMM has the best `D` of all and the
worst spells.

## Smart Tagging: measured, integrated, and not adopted

`..._smart_tagging.xml` is not a dead-ball annotation. It contains 638 `<instance>` elements, of
which only **59 are restarts** (`SET PIECES` code): Throw-in 26, Corner Kick 12, Goal Kick 10,
Free Kick 5, Kick Off 6. The other 579 carry a phase code (`BUILD UP`, `BALL IN THE BOX`, ...) and
a 30-second duration, and have no `Type` label at all. Each instance has `start`/`end` in seconds
from the start of the video, a `Team` label (`Barcelona`, `Athletic Club`, or `N/A` — the kick-offs
are `N/A`), and a `Half` label.

**Synchronisation.** `video_frame = round(seconds * 25)` with no offset. The check is the kick-off
tag: it starts at 443.88 s = frame 11097 and ends at 448.88 s = frame 11222, while the kick-off
anchor in `games.py` is frame 11186 — inside the tag's own five-second window, 3.5 s after its
start. So the tags are on the same clock as the ATD, and they lead the event by a couple of
seconds.

**What a tag actually marks.** Both restart tags that fall inside the graded window are *inside* a
ground-truth dead-ball run, but 69 and 305 frames **before** that run ends:

| tag | type | frame | GT dead-ball run | tag vs run end | GT at tag |
|---|---|---|---|---|---|
| 1 | Free Kick (Athletic) | 17533 | 17380–17838 | −305 frames (12.2 s) | `D` |
| 2 | Goal Kick (Barcelona) | 18320 | 18027–18389 | −69 frames (2.8 s) | `D` |

A tag is therefore evidence that the ball is dead **at the tag**, and no evidence about when play
resumes. It is a `D` cue, not a boundary, and integrating it as a boundary would be inventing a
delay that the data does not support.

**Coverage is the other half of the problem**: two tags in five minutes against four ground-truth
stoppages, i.e. roughly half. Over the whole match there are 59 restarts (~330 possessions), and
the first throw-in tag does not appear until 781 s.

**Integration.** `smart_tagging.py` parses the file; `possession_model.tag_hint` turns restarts into
a short window of `D` evidence (30 frames before, 60 after, asymmetric because tags lead); the hint
enters `dead_evidence` as one more cue, optionally gated by `tag_min_evidence` so that a tag can
only *confirm* a stoppage where the tracker already has doubt. Neither the team label nor the phase
tags are used.

| variant | spellF1 | macroF1 | D F1 | transF1 | full poss |
|---|---|---|---|---|---|
| `hybrid_deadbound_refine` (control, kept) | **0.507** | 0.690 | 0.427 | 0.229 | 333 |
| `hybrid_st` (30/60 window) | 0.507 | 0.690 | 0.427 | 0.229 | 335 |
| `hybrid_st_gate15` (60/150, gated at 0.15) | 0.507 | 0.692 | 0.431 | 0.229 | 335 |
| `hybrid_st_wide` (60/150, ungated) | 0.490 | 0.705 | 0.487 | 0.211 | 334 |

**Verdict: not adopted.** The tag check explains why, frame by frame: tag 17533 was *already*
predicted `D` by the control, and for tag 18320 the control says `H` (a genuine miss) which only the
*wide* window corrects — and that window costs 0.017 `spellF1` elsewhere. The gated variants leave
the ranking metric exactly unchanged, because at 18320 the tracker's own evidence is too weak
(< 0.15) for a gate to admit the tag. Smart Tagging is therefore neutral at best on this data: two
tags per five minutes, one of them redundant, cannot move a 17-possession score.

What would make it worth using: a complete set of restarts (it is missing most throw-ins), or an
annotation of the restart frame rather than the incident. Neither is in the file.

On the official board `hybrid_st_gate15` appears first because the ranking key ties on `spellF1`
(0.50667 for five variants) and the first tie-break is `macroF1`, where it leads by 0.0014 -- about
five frames out of 7500, well inside the noise band. It is kept as a variant, not adopted as the
submission: a 0.0014 tie-break does not justify a dependency on a file that covers half the
stoppages.

## What was tried and rejected

Each of these was measured on the same pipeline, one change at a time; every row is in
`evaluation/reports/experiments.csv`.

- **HMM / Viterbi with a dwell-time transition matrix** (0.219). The transition prior is the right
  idea, but a frame-level emission model is the wrong observation model: a contested second
  accumulates more log-odds than a 350-frame dwell prior can pay for, and the match fragments.
- **Median filter instead of the majority vote** (0.096 at 21 frames, 0.213 at 41, 0.292 at 61,
  0.258 at 81). The vote is not merely a smoother: it is an *integrator* of a noisy but unbiased
  signal. A median keeps edges -- including the noise's edges -- and the raw nearest-player team
  flips on a 1-5 frame scale, so the median has to be wide enough to lose short possessions anyway,
  and by then it is no better than the vote.
- **Frame-wise argmax of the emission model** (0.062, 1952 full-match possessions). Included as
  the ablation that shows how much of the score is temporal structure rather than geometry.
- **Stoppage minimum of 25 or 50 frames** (0.434 / 0.438). More dead ball, better `macroF1` and
  better transitions (0.308 at 50 frames), worse spells: false `D` inside a possession cuts it in
  two. 100 frames was the point where the `D` gain stopped costing spells.
- **Dead-ball `enter` threshold 0.3 / 0.4 / 0.7, growth 80/120, hole closing** -- all within 0.01
  of the default. Those knobs are not where the score is.
- **Vote window 60 / 140 / 175** (0.309 / 0.477 / 0.371). The window is still the most sensitive
  parameter in the pipeline; boundary refinement narrows the gap but does not remove it.

## Sensitivity: which knobs the score actually rests on

`starter/sensitivity.py` runs a one-factor-at-a-time matrix and writes
`evaluation/reports/sensitivity.csv`. On the public window:

| parameter | values → spellF1 | reading |
|---|---|---|
| `vote_window` | 60 **0.309**, 90 0.400, **112 0.507**, 140 0.477, 175 0.360 | dominates everything; sharp peak |
| `refine_shift` | 0 0.497, 15 0.490, **30 0.507**, 45 0.467, 60 0.477 | also peaked, non-monotone |
| `dead_min_frames` | 75 0.507, **100 0.507**, 125/150 0.510 | +0.003 spell but macro −0.031, D F1 −0.08: not a gain |
| `dead_enter` | 0.45 / 0.55 / 0.65 → 0.507 everywhere | inert |
| `dead_stay` | 0.20 / 0.30 / 0.40 → 0.507 everywhere | inert (macro ±0.001) |

So the ranking metric rests almost entirely on two numbers, both chosen on 17 possessions. That is
the largest open risk in the pipeline and it is a property of the *data*, not a bug that can be
fixed by more tuning: with a 225-frame vote, a possession shorter than ~4 s cannot be represented
at all, and the public window contains possessions of 19, 33, 71, 81 and 87 frames.

## What else was tried at this stage, and why it was not adopted

- **Hysteresis instead of the vote** (`switch_margin` 0.5/1.5/3.0 m, `switch_frames` 1/8/25):
  spellF1 **0.144-0.389**, 25-61 spells, 444-1164 full-match possessions. A local switching rule
  cannot replace a vote: the vote integrates a noisy signal over seconds and thereby decides the
  *segment structure*, which is what `spellF1` scores.
- **Extra splits inside vote segments** (`split_min_frames` 50/75/100, `split_margin` 1-3 m): at 75
  and 100 frames nothing changes; at 50 frames the spell count reaches the true 17 but spellF1 falls
  to 0.471 (transition F1 improves to 0.256). More segments in the right *number* is not the same
  as in the right *places*.
- **Stationary-ball cue** (`still_long_frames` 50/75/125, for restarts the other cues miss): inert
  at the 100-frame stoppage minimum, and at `dead_min_frames` 50 it buys macroF1 0.705 / D F1 0.483 /
  transition F1 0.308 at a spellF1 cost (0.438). Not adopted.

## Diagnostics: what still limits the score

`starter/diagnostics.py` writes `evaluation/reports/diagnostics_<name>.csv`. For the submitted
variant, on the 7500 graded frames:

- Confusion: `H` recall 0.913, `A` recall 0.825, `D` recall **0.311**. The single largest error is
  **`D` predicted as `H`: 864 frames, 59% of all dead-ball frames**. Three missed stoppages account
  for most of it (546, 257 and 203 frames long).
- Spell level at IoU 0.3: **4 true possessions split in two, 3 short ones missed, no merges and no
  spurious possessions.** The splits and misses are all possessions of 0.8-8 s.
- Chronological halves of the same five minutes: spellF1 **0.700** in the first half and **0.375**
  in the second (D F1 0.000 vs 0.457). The headline 0.507 is a mixture of two very different
  regimes, which is the honest scale against which any 0.02-0.03 difference between variants should
  be read.

## Full-match audit of the submission

`starter/full_match_report.py` reads one submission and the published half boundaries and checks
plausibility without any labels, so the same audit works on the hidden match. For
`hybrid_deadbound_refine.csv` it found one real defect and two known limits.

**Defect (fixed).** Ten `H`/`A` stretches were written **outside the two halves** — 1.6 s before the
kick-off, 59 s after the first half ended, **610 s during halftime**, 74 s just before the second
half, 6 s after the final whistle. The cause is simple: at halftime the players are still on the
pitch and the tracker still sees a ball, so the team vote and the dead-ball cues both behave as if
play were on. Those frames are `X` in every ground truth, so no metric moved; the file just claimed
a six-minute possession during the interval.

The repair is `respect_halves=True` in `possession_model.predict_hybrid`: `D` on every frame outside
`games.half_video_blocks`. It cannot change a score by construction — those frames are never graded —
and it removes the false statements. The result is `evaluation/results/hybrid_match_bounds.csv`:
identical public metrics, 30 fewer possessions over the whole match (333 → 303), no out-of-play
`H`/`A` at all.

**Known limits (not fixed, deliberately).** After the repair the audit still reports:

- two in-play possessions longer than 90 s (105.8 s and 96.4 s) — almost certainly two or three real
  possessions merged, which is the same short-possession limitation the public diagnostics show
  (4 true possessions split, 3 short ones missed);
- 11 possessions shorter than 1 s and 14 `A B A` wobbles — the vote tipping back and forth inside a
  contested passage.

Both could be attacked with the split machinery already in `possession_model.apply_splits`, but that
was measured: splitting at 50 frames reaches the true spell count (17) and costs 0.036 `spellF1`
(0.471 vs 0.507), because the extra boundaries land in the wrong places. Trading 0.036 of the
ranking metric for tidier full-match statistics is not a good trade, so it is documented rather than
shipped.

## Honest reading of the gain


- **`spellF1` 0.4727 -> 0.507 is +0.034 on 17 possessions.** `EVALUATION.md` puts the noise band at
  ~0.04 on the *full match*, where the sample is ~20x larger, so on the public window this single
  number is not proof. `run_evaluation.py --bootstrap 1000` should be run before anyone quotes it.
- **The supporting evidence is stronger than the headline.** `macroF1` +0.096, `D` F1 +0.25,
  transition F1 +0.067, and full-match possessions 459 -> 333 at a mean of 14.0 s against the
  ground truth's own 14.19 s. Four independent metrics moving together, each with a mechanism, is a
  different claim from one number moving.
- **The `vote_window` is still the fragile part.** 60 / 112 / 140 frames give 0.31 / 0.51 / 0.44
  with refinement on. That is the trap `EVALUATION.md` describes for `baseline_smooth50` vs
  `baseline_smooth125`: a window tuned on five minutes need not be right for ninety. The dead-ball
  overlay is what makes 333 full-match possessions plausible, but the team signal's window is still
  a public-window choice.

## Inputs and assets

**Used.** The tactical ATD (`BAR-ATH_tactical_atd.txt` plus `_atd.xml` for the team of each track
slot): ball and player positions once per video frame, in normalised pitch coordinates, converted to
metres at the pitch size from `games.py`. That file, those coordinates and the published half
boundaries are the entire input to the model.

**Checked and dropped.**

- **Homography** (`..._homography.csv`) — never needed. The ATD is already in pitch coordinates, so
  going to pixels and back would add a projection and no information.
- **Both videos** — not decoded. The tactical video is the frame-index authority for the submission,
  but no pixels are read; the broadcast video is on its own timeline, needs its own alignment, and
  would have been a separate project.
- **Notebooks and `kloppy`** — `starter/load_atd.py` and the notebooks reach the same ATD through
  kloppy; the pipeline reads the raw text directly, which is faster and needs no extra dependency.
- **Smart Tagging** — parsed, measured and integrated as an optional dead-ball cue, then rejected
  (see the section above). It never improved `spellF1`, so the submission does not use it; the
  parser and the switch stay in the repository as the record of that experiment.

## Environment

Written and verified under **Python 3.11.0** (CPython, 64-bit) with **numpy 2.2.6** and
**pandas 3.0.6**. `numpy` is the only third-party import in the model; `pandas` comes in through
`evaluation/possession_metrics.py`, which `starter/experiments.py` imports for scoring.

```bash
pip install -r requirements-submission.txt        # numpy==2.2.6, pandas==3.0.6
```

`requirements.txt` is the organisers' evaluator file (deliberately unpinned) and is left untouched;
`requirements-submission.txt` pins the same two packages to the versions used here.

## Data location and cache

The match bundle is not in the repository. Unpack it into `data/` as `DATA.md` describes, or point
`MTAH_DATA_DIR` at it. The pipeline needs:

```
data/tactical/BAR-ATH_tactical_atd.txt             # required: positions
data/tactical/BAR-ATH_tactical_atd.xml             # required: which track slot is which team
data/tactical/BAR-ATH_tactical_smart_tagging.xml   # loaded by experiments.py -- see below
```

The first run parses the 196 MB ATD text (about 25 s) and caches the per-frame features in
`data/tactical/_cache/BAR-ATH_tactical_atd_features_v3.npz` (7 MB); later runs load that in under a
second. **Expect the first run to take 60-90 seconds.** Deleting the cache is safe — it is rebuilt.

## One behaviour worth knowing

`starter/experiments.py` loads the Smart Tagging XML unconditionally, at line 224, before it looks at
which variant was requested — even though `hybrid_match_bounds` does not use the tags. Without that
file the command stops with an `IOError` naming it. It ships in the provided bundle, so a normal
reproduction is unaffected; a trimmed data folder is not.

## Reproducing

The submission file is regenerated by one command:

```bash
python starter/experiments.py --only hybrid_match_bounds --no-baseline
```

It writes `evaluation/results/hybrid_match_bounds.csv` (1,882,093 bytes). This was verified byte for
byte, not just by score: re-running it in a clean virtual environment produced MD5
`1d660d34e320e2388c6373a4e4ff796f`, identical to the submitted file. The byte-identical copy prepared
for upload is `evaluation/results/MG_TEAM.csv`.

**`evaluation/results/` must exist before the command runs.** `write_submission()` opens the output
path but does not create its folder, so a project that lacks that directory fails with
`FileNotFoundError` after the expensive part has already run. A clone of this repository ships the
directory (with a `.gitignore` rule for its contents), which makes a normal checkout safe; a
hand-made copy of just the scripts is not.

Validate exactly the file you are about to upload:

```bash
python evaluation/validate_submission.py evaluation/results/MG_TEAM.csv
```

Full sequence from a fresh clone with the bundle unpacked into `data/`:

```bash
pip install -r requirements-submission.txt
python starter/experiments.py --only hybrid_match_bounds --no-baseline    # the submission
python evaluation/validate_submission.py evaluation/results/hybrid_match_bounds.csv
python evaluation/run_evaluation.py                                       # public 5-minute score
python starter/full_match_report.py evaluation/results/hybrid_match_bounds.csv   # whole-match audit
python starter/sensitivity.py                                             # parameter sensitivity
```

`evaluation/reports/experiments.csv` accumulates one row per variant per run: `spellF1` at the
headline IoU, `macroF1`, accuracy, predicted spell and segment counts, mean spell duration,
fragmentation, per-class precision/recall, transition F1 at both tolerances, and the full-match
possession count and mean duration.

## Submission files

- `evaluation/results/MG_TEAM.csv` — byte-identical copy of the regenerated file, ready to upload
  as it is (or `git`-free rename to `<team_name>.csv`). `MG_TEAM` is the working team name; confirm
  it matches the name given in the registration form before uploading (`SUBMISSION.md`).
- `submission/MG_TEAM.md` — this document, prepared as the technical deliverable; rename to
  `<team_name>.md`, or convert to `<team_name>.pdf`.
- `submission/PUBLISH_CHECKLIST.md` — what may and may not be published, and in what order.
- `submission/REPRODUCTION_TEST.md` — the clean-environment reproduction record.

The 5-minute public window is a measurement sample, not a validation set for the whole match: every
`spellF1` figure above is scored on those 150 seconds of play, while the full-match numbers are
predictions with **no ground truth behind them**. They are reported as plausibility checks only.
