"""One-factor-at-a-time sensitivity matrix around the chosen configuration.

The point is not to find a better score by search -- 17 public possessions cannot support that.
The point is to find out which knobs the score depends on, and how steeply, so the model can be
described honestly and so a parameter that needs a lucky value can be replaced by one that does
not.

Every row changes exactly one thing from `reference`. Results go to
evaluation/reports/sensitivity.csv; no submissions are written.

    python starter/sensitivity.py
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
import tracking  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
REPORT_PATH = os.path.join(ROOT, "evaluation", "reports", "sensitivity.csv")
FPS = 25.0

# (group, label, overrides, mode). One factor at a time; `reference` is the current best model.
MATRIX = (
    ("reference", "hybrid defaults", {}, "hybrid"),
    ("vote_window", "60", {"vote_window": 60}, "hybrid"),
    ("vote_window", "90", {"vote_window": 90}, "hybrid"),
    ("vote_window", "140", {"vote_window": 140}, "hybrid"),
    ("vote_window", "175", {"vote_window": 175}, "hybrid"),
    ("dead_min_frames", "75", {"dead_min_frames": 75}, "hybrid"),
    ("dead_min_frames", "125", {"dead_min_frames": 125}, "hybrid"),
    ("dead_min_frames", "150", {"dead_min_frames": 150}, "hybrid"),
    ("dead_enter", "0.45", {"dead_enter": 0.45}, "hybrid"),
    ("dead_enter", "0.65", {"dead_enter": 0.65}, "hybrid"),
    ("dead_stay", "0.20", {"dead_stay": 0.20}, "hybrid"),
    ("dead_stay", "0.40", {"dead_stay": 0.40}, "hybrid"),
    ("refine_shift", "0", {"refine_shift": 0}, "hybrid"),
    ("refine_shift", "15", {"refine_shift": 15}, "hybrid"),
    ("refine_shift", "45", {"refine_shift": 45}, "hybrid"),
    ("refine_shift", "60", {"refine_shift": 60}, "hybrid"),
    ("hysteresis", "margin 1.5 / 8 frames", {}, "hysteresis"),
    ("hysteresis", "margin 0.5 / 8 frames", {"switch_margin": 0.5}, "hysteresis"),
    ("hysteresis", "margin 3.0 / 8 frames", {"switch_margin": 3.0}, "hysteresis"),
    ("hysteresis", "margin 1.5 / 1 frame", {"switch_frames": 1}, "hysteresis"),
    ("hysteresis", "margin 1.5 / 25 frames", {"switch_frames": 25}, "hysteresis"),
    ("hysteresis", "owner radius 6 m", {"owner_radius": 6.0}, "hysteresis"),
    ("hysteresis", "owner radius 20 m", {"owner_radius": 20.0}, "hysteresis"),
    ("split", "75 frames / 2.0 m", {"split_min_frames": 75}, "hybrid"),
    ("split", "50 frames / 2.0 m", {"split_min_frames": 50}, "hybrid"),
    ("split", "100 frames / 2.0 m", {"split_min_frames": 100}, "hybrid"),
    ("split", "75 frames / 1.0 m", {"split_min_frames": 75, "split_margin": 1.0}, "hybrid"),
    ("split", "75 frames / 3.0 m", {"split_min_frames": 75, "split_margin": 3.0}, "hybrid"),
    ("still_long", "50 frames (2 s)", {"still_long_frames": 50}, "hybrid"),
    ("still_long", "75 frames (3 s)", {"still_long_frames": 75}, "hybrid"),
    ("still_long", "125 frames (5 s)", {"still_long_frames": 125}, "hybrid"),
    ("still_long", "75 frames, dead_min 50", {"still_long_frames": 75,
                                               "dead_min_frames": 50}, "hybrid"),
)

COLUMNS = ("group", "label", "spell_f1_mean", "macro_f1", "f1_H", "f1_A", "f1_D",
           "precision_D", "recall_D", "trans_f1_25", "accuracy", "pred_spells", "gt_spells",
           "mean_spell_s", "pred_segments", "fragmentation", "full_possessions", "full_mean_s",
           "delta_spell_f1", "delta_macro_f1")


def full_stats(codes):
    pieces = [p for p in model.runs(codes) if p[2] in (model.H, model.A)]
    alive = int(np.sum((codes == model.H) | (codes == model.A)))
    return len(pieces), (alive / FPS / len(pieces) if pieces else 0.0)


def score(codes, gt_codes):
    result = pm.evaluate(gt_codes, codes)
    poss, mean_s = full_stats(codes)
    return {
        "spell_f1_mean": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "f1_H": result["per_class"]["H"]["f1"],
        "f1_A": result["per_class"]["A"]["f1"],
        "f1_D": result["per_class"]["D"]["f1"],
        "precision_D": result["per_class"]["D"]["precision"],
        "recall_D": result["per_class"]["D"]["recall"],
        "trans_f1_25": result["transitions"][25]["f1"],
        "accuracy": result["accuracy"],
        "pred_spells": result["n_pred_spells"],
        "gt_spells": result["n_gt_spells"],
        "mean_spell_s": result["mean_spell_duration_s"],
        "pred_segments": result["n_pred_segments"],
        "fragmentation": result["fragmentation"],
        "full_possessions": poss,
        "full_mean_s": mean_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=REPORT_PATH)
    parser.add_argument("--only", help="run one group only (reference, vote_window, dead_*, "
                                      "refine_shift, hysteresis)")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    features = tracking.build_features(raw_path, meta_path, n_frames, pitch)
    gt_codes = pm.label_codes(pm.read_label_csv(GT_PATH))

    rows = []
    reference = None
    for group, label, overrides, mode in MATRIX:
        if args.only and group != args.only and group != "reference":
            continue
        param = model.params(**overrides)
        prepared = model.prepare(features, param, pitch)
        if mode == "hybrid":
            codes = model.predict_hybrid(prepared, param)
        else:
            codes = model.predict_hysteresis(prepared, param)

        row = {"group": group, "label": label}
        row.update(score(codes, gt_codes))
        if group == "reference":
            reference = row
        rows.append(row)
        print("{:<16} {:<24} spell {:.4f}  macro {:.4f}  D {:.4f}  trans {:.4f}  "
              "spells {:>3}  full {:>4} @ {:.1f}s".format(
                  group, label, row["spell_f1_mean"], row["macro_f1"], row["f1_D"],
                  row["trans_f1_25"], row["pred_spells"], row["full_possessions"],
                  row["full_mean_s"]), flush=True)

    for row in rows:
        row["delta_spell_f1"] = row["spell_f1_mean"] - reference["spell_f1_mean"]
        row["delta_macro_f1"] = row["macro_f1"] - reference["macro_f1"]

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in COLUMNS})
    print("\nwrote {} ({} rows)".format(os.path.relpath(args.out, ROOT), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
