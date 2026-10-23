"""Per-frame and per-spell diagnostics for one submission.

Answers the questions a score cannot: which label is confused with which, where the longest errors
sit, and whether a spell was split or merged rather than merely misplaced.

Only the graded frames are used (the public ground truth is `X` everywhere else), and the window is
contiguous, so runs inside it are well defined. Nothing here writes a submission or changes the
evaluator; it reads one CSV and the public ground truth and reports.

    python starter/diagnostics.py evaluation/results/hybrid_deadbound_refine.csv
    python starter/diagnostics.py <csv> --top 15 --out evaluation/reports

Writes evaluation/reports/diagnostics_<name>.csv with one row per finding:
  error_run   a contiguous stretch where the prediction is wrong
  gt_split    one true possession covered by two or more predicted possessions
  pred_merge  one predicted possession covering two or more true possessions
  gt_missed   a true possession no prediction overlaps meaningfully
  pred_spurious  a predicted possession that overlaps no true possession
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

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25.0
LABELS = np.array(["H", "A", "D"])


def runs_of(codes, labels=(pm.H, pm.A)):
    """[(start, stop, code)] maximal runs of *equal* codes whose code is in `labels`.

    Runs of equal labels, not runs of "either possession label": `H A` next to each other are two
    possessions (a turnover), not one, and treating them as one invented 66-second "spells" in the
    first version of this report.
    """
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return [(int(a), int(b), int(codes[a])) for a, b in zip(starts, stops) if codes[a] in labels]


def iou(first, second):
    low = max(first[0], second[0])
    high = min(first[1], second[1])
    inter = max(0, high - low)
    union = (first[1] - first[0]) + (second[1] - second[0]) - inter
    return inter / float(union) if union else 0.0


def overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def error_runs(gt, pred, graded):
    """Contiguous stretches inside the graded window where the prediction is wrong."""
    wrong = graded & (gt != pred)
    out = []
    start = None
    for frame in range(len(wrong)):
        if wrong[frame] and start is None:
            start = frame
        elif not wrong[frame] and start is not None:
            out.append((start, frame))
            start = None
    if start is not None:
        out.append((start, len(wrong)))
    return [(a, b) for a, b in out if b > a]


def spell_findings(gt, pred, graded):
    """Split / merge / missed / spurious, at IoU 0.3, inside the graded window."""
    window = np.flatnonzero(graded)
    if not len(window):
        return []
    first, last = int(window[0]), int(window[-1]) + 1
    gt_spells = runs_of(gt[first:last])
    pred_spells = runs_of(pred[first:last])

    findings = []
    matched_pred = set()
    for gs in gt_spells:
        overlaps = [(index, overlap(gs, ps)) for index, ps in enumerate(pred_spells)]
        overlaps = [(index, size) for index, size in overlaps if size > 0]
        length = gs[1] - gs[0]
        pieces = [(index, size) for index, size in overlaps if size >= 0.2 * length]
        if not overlaps:
            findings.append(("gt_missed", gs, None, 0.0,
                             "no predicted possession overlaps it"))
            continue
        best_index = max(overlaps, key=lambda item: iou(gs, pred_spells[item[0]]))[0]
        best = pred_spells[best_index]
        best_iou = iou(gs, best)
        matched_pred.add(best_index)
        if len(pieces) >= 2:
            matched_pred.update(index for index, _ in pieces)
            findings.append(("gt_split", gs, best, best_iou,
                             "{} predicted pieces".format(len(pieces))))
        elif best_iou < 0.3:
            findings.append(("gt_missed", gs, best, best_iou, "best IoU below 0.3"))

    for index, ps in enumerate(pred_spells):
        if index in matched_pred:
            continue
        covering = [(gs, overlap(gs, ps)) for gs in gt_spells]
        covering = [(gs, size) for gs, size in covering if size >= 0.2 * (ps[1] - ps[0])]
        if len(covering) >= 2:
            findings.append(("pred_merge", ps, covering[0][0], 0.0,
                             "covers {} true possessions".format(len(covering))))
        elif not covering:
            findings.append(("pred_spurious", ps, None, 0.0,
                             "overlaps no true possession"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--gt", default=GT_PATH)
    parser.add_argument("--top", type=int, default=10, help="longest error runs to print")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "reports"))
    parser.add_argument("--halves", action="store_true",
                        help="also score the two chronological halves of the graded window")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    pred = pm.label_codes(pm.read_label_csv(args.submission))
    if len(gt) != len(pred):
        raise SystemExit("length mismatch: gt {} vs pred {}".format(len(gt), len(pred)))
    graded = gt != pm.X
    name = os.path.splitext(os.path.basename(args.submission))[0]

    print("submission: {}  ({} graded frames)".format(name, int(graded.sum())))
    print("\nconfusion on graded frames (rows = truth, cols = prediction)")
    matrix = np.zeros((4, 4), dtype=int)
    for truth in range(4):
        for guess in range(4):
            matrix[truth, guess] = int(np.sum(graded & (gt == truth) & (pred == guess)))
    print("        {:>8}{:>8}{:>8}{:>8}".format(*pm.LABELS))
    for truth in range(4):
        print("  {:<6}{:>8}{:>8}{:>8}{:>8}".format(pm.LABELS[truth], *matrix[truth]))

    print("\nper class (graded):")
    for code in (pm.H, pm.A, pm.D):
        hit = matrix[code, code]
        support = matrix[code, :3].sum()
        predicted = matrix[:3, code].sum()
        precision = hit / predicted if predicted else 0.0
        recall = hit / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print("  {}  precision {:6.3f}  recall {:6.3f}  F1 {:6.3f}  support {:>5}".format(
            pm.LABELS[code], precision, recall, f1, support))

    errors = error_runs(gt, pred, graded)
    errors.sort(key=lambda item: item[1] - item[0], reverse=True)
    print("\n{} error runs; {} longest:".format(len(errors), min(args.top, len(errors))))
    print("  {:>7} {:>7} {:>6} {:>7}   {} -> {}".format(
        "start", "end", "frames", "seconds", "truth", "prediction"))
    for start, stop in errors[:args.top]:
        truth = np.bincount(gt[start:stop], minlength=4).argmax()
        guess = np.bincount(pred[start:stop], minlength=4).argmax()
        print("  {:>7} {:>7} {:>6} {:>7.1f}   {:<6} -> {}".format(
            start, stop, stop - start, (stop - start) / FPS,
            pm.LABELS[truth], pm.LABELS[guess]))

    findings = spell_findings(gt, pred, graded)
    counts = {}
    for kind, _, _, _, _ in findings:
        counts[kind] = counts.get(kind, 0) + 1
    print("\nspell-level findings at IoU 0.3: {}".format(
        ", ".join("{} {}".format(v, k) for k, v in sorted(counts.items())) or "none"))
    for kind, subject, other, value, note in findings[:args.top]:
        print("  {:<14} {:>6}-{:<6} ({:>5.1f} s)  IoU {:.2f}  {}".format(
            kind, subject[0], subject[1], (subject[1] - subject[0]) / FPS, value, note))

    if args.halves:
        print("\nchronological halves of the graded window (a crude stability check)")
        window = np.flatnonzero(graded)
        middle = (int(window[0]) + int(window[-1])) // 2
        index = np.arange(len(gt))
        for part, low, high in (("first half", int(window[0]), middle),
                                ("second half", middle, int(window[-1]) + 1)):
            masked = np.where((index >= low) & (index < high), gt, pm.X)
            result = pm.evaluate(masked, pred)
            print("  {:<12} {:>6}-{:<6}  spellF1 {:.3f}  macroF1 {:.3f}  D F1 {:.3f}  "
                  "spells {:>2}/{:<2}".format(
                      part, low, high, result["spell_f1_mean"], result["macro_f1"],
                      result["per_class"]["D"]["f1"], result["n_pred_spells"],
                      result["n_gt_spells"]))
    return _write(name, findings, errors, gt, pred, args.out)


def _write(name, findings, errors, gt, pred, folder):
    """One CSV row per finding, plus the error runs, for later comparison across candidates."""
    if not os.path.isdir(folder):
        os.makedirs(folder)
    path = os.path.join(folder, "diagnostics_{}.csv".format(name))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("kind", "frame_start", "frame_end", "frames", "seconds",
                         "truth_label", "pred_label", "iou", "note"))
        for kind, subject, other, value, note in findings:
            writer.writerow((kind, subject[0], subject[1], subject[1] - subject[0],
                             round((subject[1] - subject[0]) / FPS, 2),
                             LABELS[subject[2]],
                             LABELS[other[2]] if other else "",
                             round(value, 3), note))
        for start, stop in errors:
            truth = np.bincount(gt[start:stop], minlength=4).argmax()
            guess = np.bincount(pred[start:stop], minlength=4).argmax()
            writer.writerow(("error_run", start, stop, stop - start,
                             round((stop - start) / FPS, 2), LABELS[truth], LABELS[guess],
                             "", ""))
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
