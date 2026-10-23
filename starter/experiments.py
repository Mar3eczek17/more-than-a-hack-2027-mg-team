"""Controlled experiments: score possession variants against the public five minutes.

One pipeline, one parameter change at a time, every result written to CSV. The runner writes each
variant as a normal submission into evaluation/results/ (so run_evaluation.py picks it up and the
official board can be produced from the same files) and appends a row per variant to
evaluation/reports/experiments.csv.

    python starter/experiments.py                 # the default suite
    python starter/experiments.py --only default  # one variant, by name
    python starter/experiments.py --list

What to read in the report, and what not to
-------------------------------------------
`spell_f1_mean` on the public window is 17 possessions. One missed possession moves it by ~0.06, so
a 0.02 gap is not a result. The report therefore also carries the full-match possession count and
mean duration: a submission claiming 900 possessions of 5 s is wrong in a way the public window
cannot show, because 17 possessions is too small a sample to notice it. Those two columns are
priors from football (a match has roughly 200-300 possessions of 10-20 s), not scores.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import possession_model as model  # noqa: E402
import smart_tagging as st  # noqa: E402
import tracking  # noqa: E402
from games import GAME, GAMES, asset_paths, require, source_frames  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
REPORTS_DIR = os.path.join(ROOT, "evaluation", "reports")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
REPORT_PATH = os.path.join(REPORTS_DIR, "experiments.csv")
FPS = 25.0

LABELS = np.array(["H", "A", "D"])

# Variants: (name, parameter overrides, mode, use_smart_tagging). `mode` is how the per-frame
# decision is made:
#   hmm        -- emission model + Viterbi over the whole match
#   frame_wise -- argmax of the emission model, no temporal model (an ablation)
#   hybrid     -- baseline-style majority-vote teams + an evidence-driven dead-ball overlay
# `use_smart_tagging` adds the restart tags as one more dead-ball cue (`tag_hint`), never as a
# boundary: see smart_tagging.py for why.
VARIANTS = (
    ("hybrid_deadbound_refine", {}, "hybrid", False),          # current best, control
    ("hybrid_match_bounds", {"respect_halves": True}, "hybrid", False),  # + D outside the halves
    ("hybrid_st", {}, "hybrid", True),                          # + Smart Tagging cue
    ("hybrid_st_wide", {"tag_before": 60, "tag_after": 150}, "hybrid", True),
    ("hybrid_st_gate15", {"tag_before": 60, "tag_after": 150,
                          "tag_min_evidence": 0.15}, "hybrid", True),
    ("hybrid_st_gate25", {"tag_before": 60, "tag_after": 150,
                          "tag_min_evidence": 0.25}, "hybrid", True),
    ("hybrid_st_gate35", {"tag_before": 60, "tag_after": 150,
                          "tag_min_evidence": 0.35}, "hybrid", True),
    ("hybrid_no_refine", {"refine_shift": 0}, "hybrid", False),  # ablation: no boundary refinement
)

COLUMNS = (
    "name", "spell_f1_mean", "spell_f1_at_0.5", "macro_f1", "accuracy",
    "pred_spells", "gt_spells", "pred_segments", "fragmentation",
    "mean_spell_s", "gt_mean_spell_s",
    "f1_H", "f1_A", "f1_D", "precision_D", "recall_D",
    "team_acc", "share_err", "alive_dead_f1",
    "trans_f1_12", "trans_f1_25",
    "full_match_possessions", "full_match_mean_s",
)


def write_submission(path, codes):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, LABELS[code])
                          for frame, code in enumerate(codes))
    return path


def full_match_stats(codes):
    """(possessions, mean seconds) over the whole match, counting every H or A run.

    This is the sanity check the public window cannot provide: 199,319 frames is the whole match,
    and a possession count of 700 or 40 is a statement about football that is simply false. It is
    reported, not optimised -- there is no ground truth for it here.
    """
    pieces = [piece for piece in model.runs(codes) if piece[2] in (model.H, model.A)]
    alive = int(np.sum((codes == model.H) | (codes == model.A)))
    mean = (alive / FPS / len(pieces)) if pieces else 0.0
    return len(pieces), mean


def summarise(name, codes, gt_codes):
    result = pm.evaluate(gt_codes, codes)
    trans = result["transitions"]
    poss, mean_s = full_match_stats(codes)
    return {
        "name": name,
        "spell_f1_mean": result["spell_f1_mean"],
        "spell_f1_at_0.5": result["spell_f1_at_headline"],
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "pred_spells": result["n_pred_spells"],
        "gt_spells": result["n_gt_spells"],
        "pred_segments": result["n_pred_segments"],
        "fragmentation": result["fragmentation"],
        "mean_spell_s": result["mean_spell_duration_s"],
        "gt_mean_spell_s": result["gt_mean_spell_duration_s"],
        "f1_H": result["per_class"]["H"]["f1"],
        "f1_A": result["per_class"]["A"]["f1"],
        "f1_D": result["per_class"]["D"]["f1"],
        "precision_D": result["per_class"]["D"]["precision"],
        "recall_D": result["per_class"]["D"]["recall"],
        "team_acc": result["team_acc_gt_alive"],
        "share_err": result["share_err"],
        "alive_dead_f1": result["alive_dead_f1"],
        "trans_f1_12": trans[12]["f1"],
        "trans_f1_25": trans[25]["f1"],
        "full_match_possessions": poss,
        "full_match_mean_s": mean_s,
    }


def format_row(row):
    return ("{:<18} spell {:>6.3f}  macro {:>6.3f}  acc {:>6.3f}  D f1 {:>5.3f} "
            "(P {:>5.3f} R {:>5.3f})  spells {:>4}/{:<3} (full {:>4}, {:>4.1f}s)  "
            "trans {:>5.3f}").format(
                row["name"], row["spell_f1_mean"], row["macro_f1"], row["accuracy"],
                row["f1_D"], row["precision_D"], row["recall_D"],
                row["pred_spells"], row["gt_spells"],
                row["full_match_possessions"], row["full_match_mean_s"],
                row["trans_f1_25"])


def window_runs(codes, first, last):
    """[(start, length, label)] of runs overlapping the public window, in window coordinates."""
    out = []
    for start, stop, code in model.runs(codes):
        if start < last + 1 and stop > first:
            out.append((max(start, first) - first, min(stop, last + 1) - max(start, first),
                        LABELS[code]))
    return out


def print_detail(codes, gt_codes, first=11185, last=18684):
    """Show the predicted segment structure next to the truth for the graded window.

    The scores say *that* a submission over-segments; this says *where*. A run of 20 s split into
    three is a different bug from 40 one-second islands, and the fix differs.
    """
    gt_runs = window_runs(gt_codes, first, last)
    pred_runs = window_runs(codes, first, last)
    print("\npublic window segment structure ({}-{})".format(first, last))
    print("  GT   {:>2} segments: {}".format(len(gt_runs),
                                             " ".join("{}{}".format(label, length)
                                                      for _, length, label in gt_runs)))
    print("  pred {:>2} segments: {}".format(len(pred_runs),
                                             " ".join("{}{}".format(label, length)
                                                      for _, length, label in pred_runs)))
    lengths = sorted(length for _, length, _ in pred_runs)
    if lengths:
        print("  predicted segment lengths: min {} median {} max {}".format(
            lengths[0], lengths[len(lengths) // 2], lengths[-1]))


def print_tag_check(produced, gt_codes, tag_frames, first=11185, last=18684):
    """For every tag inside the graded window: what the truth says, and what each variant says.

    This is the only honest way to answer "did Smart Tagging do anything?": either a tag changes a
    frame's label, or it does not, and the score difference is then exactly the number of frames it
    changed in the right direction.
    """
    inside = [frame for frame in tag_frames if first <= frame <= last]
    print("\nSmart Tagging tags inside the graded window ({}):".format(len(inside)))
    header = "  frame  GT" + "".join("{:>26}".format(name) for name in produced)
    print(header)
    for frame in inside:
        row = "  {:>6}  {}".format(frame, LABELS[gt_codes[frame]])
        for name, codes in produced.items():
            # A tag is a hint about a stretch, so report what the model says about the whole hint.
            window = codes[max(0, frame - 30):frame + 61]
            share = float(np.mean(window == 2))
            row += "{:>26}".format("{} ({:.0f}% D)".format(LABELS[codes[frame]], 100.0 * share))
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", help="variant names to run (see --list)")
    parser.add_argument("--list", action="store_true", help="list variants and exit")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip scoring baseline_smooth112.csv for the comparison row")
    parser.add_argument("--detail", action="store_true",
                        help="print the predicted segment structure for the last variant")
    args = parser.parse_args()

    if args.list:
        for name, overrides, mode, use_tags in VARIANTS:
            print("{:<26} mode={:<10} tags={:<5} {}".format(name, mode, use_tags, overrides))
        return 0

    variants = VARIANTS
    if args.only:
        wanted = set(args.only)
        variants = tuple(v for v in VARIANTS if v[0] in wanted)
        if not variants:
            raise SystemExit("no such variant; try --list")

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    print("loading tracking features ...", flush=True)
    features = tracking.build_features(raw_path, meta_path, n_frames, pitch)
    gt_codes = pm.label_codes(pm.read_label_csv(GT_PATH))

    tag_events = st.restarts(st.load(asset_paths(game=GAME)["smart_tagging"]))
    tag_frames = [st.to_frame(event.start) for event in tag_events]
    print("smart tagging: {} restart tags, first at frame {}, last at {}".format(
        len(tag_frames), tag_frames[0], tag_frames[-1]), flush=True)

    rows = []
    if not args.no_baseline:
        baseline_path = os.path.join(RESULTS_DIR, "baseline_smooth112.csv")
        if os.path.isfile(baseline_path):
            codes = pm.label_codes(pm.read_label_csv(baseline_path))
            row = summarise("baseline112(ref)", codes, gt_codes)
            rows.append(row)
            print(format_row(row), flush=True)

    produced = {}
    for name, overrides, mode, use_tags in variants:
        param = model.params(**overrides)
        prepared = model.prepare(features, param, pitch, tag_frames if use_tags else None)
        if mode == "hybrid":
            codes = model.predict_hybrid(prepared, param)
        else:
            codes = model.predict(prepared, param, frame_wise=(mode == "frame_wise"))
        produced[name] = codes
        write_submission(os.path.join(RESULTS_DIR, name + ".csv"), codes)
        row = summarise(name, codes, gt_codes)
        rows.append(row)
        print(format_row(row), flush=True)
        if args.detail and name == variants[-1][0]:
            print_detail(codes, gt_codes, 11185, 18684)

    if args.detail:
        print_tag_check(produced, gt_codes, tag_frames)

    if not os.path.isdir(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    with open(REPORT_PATH, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in COLUMNS})
    print("\nappended {} rows to {}".format(len(rows), os.path.relpath(REPORT_PATH, ROOT)))
    print("score the official board with: python evaluation/run_evaluation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
