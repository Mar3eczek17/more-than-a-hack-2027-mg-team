"""Diagnostics: what do the tracking features look like under each ground-truth label?

This is the evidence behind every threshold in possession_model.py. It exists so the thresholds
can be argued from the data rather than invented, and so the same table can be re-printed after
any change. It never writes a submission and it only reads the public five minutes.

    python starter/explore_features.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import tracking  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
WINDOW = (11185, 18684)  # the graded slice, from the GT metadata


def quantile_line(values, name):
    values = values[np.isfinite(values)]
    if not len(values):
        return "{}: (all NaN)".format(name)
    points = np.percentile(values, [5, 25, 50, 75, 95])
    return "{}: n {:>6}  ".format(name, len(values)) + "  ".join(
        "{:>7.2f}".format(v) for v in points)


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]

    print("loading ATD (first run parses 196 MB and caches the features) ...", flush=True)
    features = tracking.build_features(raw_path, meta_path, n_frames, pitch, verbose=True)

    seen = tracking.ball_visible(features)
    gap = tracking.gap_to_observation(features)
    speed = tracking.ball_speed(features, pitch)
    outside = tracking.outside_pitch(features)

    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    first, last = WINDOW
    window = slice(first, last + 1)

    print("\n=== share of each GT label by ball visibility, public window ===")
    for code, name in enumerate(pm.LABELS[:3]):
        mask = (gt == code)[window]
        print("  {} {:>5} frames | visible {:5.1f}% | outside pitch {:5.1f}% | "
              "gap>25 {:5.1f}% | players tracked mean {:4.1f}".format(
                  name, int(mask.sum()),
                  100.0 * seen[window][mask].mean(),
                  100.0 * outside[window][mask].mean(),
                  100.0 * (gap[window][mask] > 25).mean(),
                  features["n_players"][window][mask].mean()))

    print("\n=== feature quantiles (5/25/50/75/95) by GT label, public window ===")
    for code, name in enumerate(pm.LABELS[:3]):
        mask = (gt == code)[window]
        print(" {}:".format(name))
        print("   " + quantile_line(features["nearest"][window][mask], "nearest d1 (m)"))
        print("   " + quantile_line(features["margin"][window][mask], "margin d2-d1 (m)"))
        print("   " + quantile_line(speed[window][mask], "ATD ball speed"))
        print("   " + quantile_line(gap[window][mask].astype(float), "frames to seen ball"))
        print("   " + quantile_line(features["n_near3"][window][mask].astype(float),
                                    "players within 3 m"))

    print("\n=== GT label x d1 bucket, % of the label's frames ===")
    edges = [0, 2, 4, 8, 15, np.inf]
    print("  {:>8}".format("") + "".join("{:>9}".format("<{}m".format(e)) for e in edges[:-1])
          + "{:>10}".format("no d1"))
    for code, name in enumerate(pm.LABELS[:3]):
        mask = (gt == code)[window]
        d1 = features["nearest"][window]
        row = "  {:>8}".format(name)
        for low, high in zip(edges[:-1], edges[1:]):
            hit = mask & (d1 >= low) & (d1 < high)
            row += "{:>9.1f}".format(100.0 * hit.sum() / mask.sum())
        row += "{:>10.1f}".format(100.0 * (mask & ~np.isfinite(d1)).sum() / mask.sum())
        print(row)

    print("\n=== GT label x ball-speed bucket, % of the label's frames ===")
    edges = [0, 1, 3, 8, np.inf]
    print("  {:>8}".format("") + "".join("{:>9}".format("<{}".format(e)) for e in edges[:-1])
          + "{:>10}".format("no speed"))
    for code, name in enumerate(pm.LABELS[:3]):
        mask = (gt == code)[window]
        sp = speed[window]
        row = "  {:>8}".format(name)
        for low, high in zip(edges[:-1], edges[1:]):
            hit = mask & (sp >= low) & (sp < high)
            row += "{:>9.1f}".format(100.0 * hit.sum() / mask.sum())
        row += "{:>10.1f}".format(100.0 * (mask & ~np.isfinite(sp)).sum() / mask.sum())
        print(row)

    print("\n=== GT run lengths, public window (frames) ===")
    runs = pm.valid_runs(gt)
    lengths = {}
    for start, stop, code in pm.segments(gt, runs):
        if start < last + 1 and stop > first:
            lengths.setdefault(pm.LABELS[code], []).append(stop - start)
    for name, values in sorted(lengths.items()):
        print("  {}  n {:>2}  {}".format(name, len(values), sorted(values)))

    print("\n=== how well the raw nearest player matches the GT team ===")
    team = features["nearest_team"][window]
    for code, name in enumerate(pm.LABELS[:2]):
        mask = (gt == code)[window]
        print("  GT {}: nearest player agrees {:5.1f}%  (where margin>1.5 m: {:5.1f}%)".format(
            name, 100.0 * np.mean(team[mask] == code),
            100.0 * np.mean(team[mask & (features["margin"][window] > 1.5)] == code)))

    calibration(features, seen, gap, speed, outside, gt, window)


def calibration(features, seen, gap, speed, outside, gt, window):
    """P(GT label | cue bucket): the table that decides whether a cue is worth using at all.

    A cue earns its place only if the bucket it flags is mostly one label. Coverage matters just as
    much: a rule that is 95% precise and fires on 1% of the dead-ball frames cannot carry the
    class, and the HMM cannot extend a stoppage it never sees.
    """
    d1 = features["nearest"][window]
    n_players = features["n_players"][window]
    gap_w = gap[window]
    speed_w = speed[window]
    outside_w = outside[window]
    gt_w = gt[window]

    def table(title, edges, values, labels=("H", "A", "D")):
        print("\n" + title)
        print("  {:>12} {:>9}".format("bucket", "frames") +
              "".join("{:>8}".format(name) for name in labels) + "{:>10}".format("D share"))
        for low, high in zip(edges[:-1], edges[1:]):
            hit = (values >= low) & (values < high) & np.isfinite(values)
            count = int(hit.sum())
            if not count:
                continue
            shares = [100.0 * np.mean(gt_w[hit] == index) for index in range(len(labels))]
            print("  {:>5}-{:<6} {:>9}".format(
                fmt(low), fmt(high), count) + "".join("{:>8.1f}".format(s) for s in shares))

    def fmt(value):
        return "inf" if not np.isfinite(value) else "{:g}".format(value)

    print("\n=== cue calibration, public window (share of frames in the bucket that are each label) ===")
    table("distance to nearest player (m)", [0, 1, 2, 3, 5, 8, 12, np.inf], d1)
    table("frames since the ball was seen", [-1, 0, 3, 8, 15, 25, 50, np.inf], gap_w)
    table("ball speed (m/s)", [0, 0.5, 1, 2, 4, 8, np.inf], speed_w)
    table("players tracked", [-1, 8, 12, 16, 20, np.inf], n_players.astype(float))

    print("\n  ball outside the pitch: {} frames, of which D {:5.1f}%  "
          "(coverage of GT D: {:4.1f}%)".format(
              int(outside_w.sum()), 100.0 * np.mean(gt_w[outside_w] == pm.D),
              100.0 * outside_w[gt_w == pm.D].sum() / max(1, int((gt_w == pm.D).sum()))))

    still = np.isfinite(speed_w) & (speed_w < 1.0)
    print("  ball still (<1 m/s):    {} frames, of which D {:5.1f}%  "
          "(coverage of GT D: {:4.1f}%)".format(
              int(still.sum()), 100.0 * np.mean(gt_w[still] == pm.D),
              100.0 * still[gt_w == pm.D].sum() / max(1, int((gt_w == pm.D).sum()))))

    still_far = still & np.isfinite(d1) & (d1 > 8)
    print("  still and >8 m away:    {} frames, of which D {:5.1f}%  "
          "(coverage of GT D: {:4.1f}%)".format(
              int(still_far.sum()), 100.0 * np.mean(gt_w[still_far] == pm.D),
              100.0 * still_far[gt_w == pm.D].sum() / max(1, int((gt_w == pm.D).sum()))))

    no_ball = ~seen[window]
    print("  no ball at all:         {} frames, of which D {:5.1f}%  "
          "(coverage of GT D: {:4.1f}%)".format(
              int(no_ball.sum()), 100.0 * np.mean(gt_w[no_ball] == pm.D),
              100.0 * no_ball[gt_w == pm.D].sum() / max(1, int((gt_w == pm.D).sum()))))


if __name__ == "__main__":
    main()
