"""Full-match audit of one submission, using no labels at all.

Everything here is computed from the submission and from `games.py`'s published half boundaries, so
it works on the hidden match as well. It answers the questions a public score cannot:

  * is the label mix plausible inside the two halves, and is anything other than `D` written outside
    play (pre-match, halftime, after the final whistle)?
  * how many possessions does the file claim, how long are they, and are the shortest and longest
    ones physically possible?
  * how many direct H<->A handovers are there, and how many go through a stoppage?
  * does every predicted stoppage obey the model's own minimum duration?
  * are there flicker patterns (A B A B) or very short islands that no analysis would accept?

Writes evaluation/reports/full_match_<name>.csv (one row per segment and per anomaly) and prints a
summary. No ground truth is read, and nothing is scored: this is a plausibility audit, not a
measurement of correctness.

    python starter/full_match_report.py evaluation/results/hybrid_deadbound_refine.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = 25.0
LABELS = ("H", "A", "D", "X")
BUCKETS = (0, 25, 50, 100, 200, 400, 800, 1600, np.inf)


def segments(codes):
    """[(start, stop, code)] maximal runs of equal codes."""
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return [(int(a), int(b), int(codes[a])) for a, b in zip(starts, stops)]


def distribution(lengths, label):
    out = []
    total = max(1, len(lengths))
    for low, high in zip(BUCKETS[:-1], BUCKETS[1:]):
        count = int(np.sum((lengths >= low) & (lengths < high)))
        if count:
            out.append("{}-{}{}:{} ({:.0f}%)".format(
                int(low), "" if np.isinf(high) else int(high), "s" if False else "",
                count, 100.0 * count / total))
    return "{} n={} mean={:.1f}s median={:.1f}s min={:.1f}s max={:.1f}s\n    {}".format(
        label, len(lengths), float(np.mean(lengths)) / FPS if len(lengths) else 0.0,
        float(np.median(lengths)) / FPS if len(lengths) else 0.0,
        float(np.min(lengths)) / FPS if len(lengths) else 0.0,
        float(np.max(lengths)) / FPS if len(lengths) else 0.0, "  ".join(out))


def anomalies(codes, pieces, blocks, min_dead):
    """Structural oddities, each with the frame range that shows it."""
    found = []

    # 1. Anything other than D outside play: the ball is not on the pitch, so a team label is a
    #    claim the tracking cannot support.
    in_play = np.zeros(len(codes), dtype=bool)
    for _, first, last in blocks:
        in_play[first:last + 1] = True
    wrong = (~in_play) & np.isin(codes, (pm.H, pm.A))
    if wrong.any():
        edges = np.flatnonzero(np.diff(wrong.astype(np.int8)))
        starts = np.concatenate(([0], edges + 1))
        stops = np.concatenate((edges + 1, [len(codes)]))
        for start, stop in zip(starts, stops):
            if wrong[start]:
                found.append(("label_outside_play", int(start), int(stop),
                              "H/A written outside the two halves"))

    dead_lengths = []
    for start, stop, code in pieces:
        length = stop - start
        if code == pm.D:
            dead_lengths.append(length)
            # 2. A stoppage shorter than the model's own minimum contradicts its own rule.
            if length < min_dead:
                found.append(("short_stoppage", start, stop,
                              "{} frames < min_dead {}".format(length, min_dead)))
        elif code in (pm.H, pm.A):
            if length < 25:
                found.append(("flicker_possession", start, stop,
                              "{} frames".format(length)))
            if length > 2250:
                found.append(("very_long_possession", start, stop,
                              "{:.0f} s".format(length / FPS)))

    # 3. A B A within a second and a half: a label that returns immediately is a wobble.
    for index in range(len(pieces) - 2):
        first, second, third = pieces[index], pieces[index + 1], pieces[index + 2]
        if first[2] == third[2] != second[2] and second[1] - second[0] < 40:
            found.append(("aba_wobble", second[0], second[1],
                          "{} -> {} -> {}".format(LABELS[first[2]], LABELS[second[2]],
                                                  LABELS[third[2]])))
    return found, dead_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "reports"))
    parser.add_argument("--min-dead", type=int, default=100,
                        help="the model's own minimum stoppage length, in frames")
    args = parser.parse_args()

    codes = pm.label_codes(pm.read_label_csv(args.submission))
    name = os.path.splitext(os.path.basename(args.submission))[0]
    n_frames = len(codes)
    blocks = half_video_blocks(GAME)

    print("submission: {}  ({} frames, {:.1f} min)".format(
        name, n_frames, n_frames / FPS / 60.0))
    print("\nlabel distribution")
    print("  file   " + "  ".join(
        "{} {:>7} ({:5.1f}%)".format(label, int(np.sum(codes == index)),
                                     100.0 * np.sum(codes == index) / n_frames)
        for index, label in enumerate(LABELS)))
    for half, first, last in blocks:
        part = codes[first:last + 1]
        print("  half {:<2} " .format(half) + "  ".join(
            "{} {:>7} ({:5.1f}%)".format(label, int(np.sum(part == index)),
                                         100.0 * np.sum(part == index) / len(part))
            for index, label in enumerate(LABELS)))

    pieces = segments(codes)
    spells = [(a, b, c) for a, b, c in pieces if c in (pm.H, pm.A)]
    spell_lengths = np.array([b - a for a, b, _ in spells])
    found, dead_lengths = anomalies(codes, pieces, blocks, args.min_dead)

    pairs = {}
    for left, right in zip(pieces, pieces[1:]):
        key = (LABELS[left[2]], LABELS[right[2]])
        pairs[key] = pairs.get(key, 0) + 1

    print("\npossessions (maximal H or A runs), whole file")
    print("  " + distribution(spell_lengths, "H/A spells"))
    print("  " + distribution(np.array(dead_lengths), "D runs"))
    print("\ntransitions between consecutive segments")
    for key in sorted(pairs):
        print("  {} -> {}  {}".format(key[0], key[1], pairs[key]))

    print("\npossessions per 5-minute block inside each half")
    for half, first, last in blocks:
        counts = []
        start = first
        while start <= last:
            stop = min(start + 300 * 25, last + 1)
            part = codes[start:stop]
            counts.append(sum(1 for a, b, c in segments(part) if c in (pm.H, pm.A)))
            start = stop
        print("  half {}: {}".format(half, " ".join(str(c) for c in counts)))

    print("\nlongest segments by label")
    for code, label in ((pm.H, "H"), (pm.A, "A"), (pm.D, "D")):
        group = sorted((piece for piece in pieces if piece[2] == code),
                       key=lambda piece: piece[1] - piece[0], reverse=True)[:3]
        print("  {}: ".format(label) + "  ".join(
            "{}-{} ({:.1f} s)".format(a, b, (b - a) / FPS) for a, b, _ in group))

    print("\nanomalies: {}".format(len(found)))
    counts = {}
    for kind, _, _, _ in found:
        counts[kind] = counts.get(kind, 0) + 1
    for kind, value in sorted(counts.items()):
        print("  {:<24} {}".format(kind, value))
    for kind, start, stop, note in found[:15]:
        print("  {:<24} {:>7}-{:<7} ({:>5.1f} s)  {}".format(
            kind, start, stop, (stop - start) / FPS, note))

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    path = os.path.join(args.out, "full_match_{}.csv".format(name))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("kind", "frame_start", "frame_end", "frames", "seconds",
                         "label", "note"))
        for start, stop, code in pieces:
            writer.writerow(("segment", start, stop, stop - start,
                             round((stop - start) / FPS, 2), LABELS[code], ""))
        for kind, start, stop, note in found:
            writer.writerow((kind, start, stop, stop - start,
                             round((stop - start) / FPS, 2), "", note))
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
