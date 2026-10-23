"""Does Smart Tagging line up with the tracking, and what is it actually good for?

Three questions, answered with numbers rather than assumptions:

  1. Where do the restart tags fall relative to the video timeline? The kick-off tag is checked
     against the kick-off anchor in games.py, and every restart in the public window is printed
     next to the ground truth's dead-ball runs.
  2. Is a tagged set piece inside a stoppage, at its start, or at its end? That decides whether a
     tag can be used to *place* a boundary or only to *confirm* one.
  3. Does the `Team` label identify the side that restarts play? Checked against the ground-truth
     label shortly after the tag, which is the only way to know rather than assume.

Nothing here writes a submission; the ground truth is used for measurement only.

    python starter/explore_smart_tagging.py
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
import smart_tagging as st  # noqa: E402
from games import GAME, GAMES, anchor_video_frame, asset_paths  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FIRST, LAST = 11185, 18684
FPS = 25.0


def gt_window():
    """GT segments and dead-ball runs inside the graded window."""
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    runs = pm.valid_runs(gt)
    segments = [(s, t, c) for s, t, c in pm.segments(gt, runs) if s < LAST + 1 and t > FIRST]
    dead = [(s, t) for s, t, c in segments if c == pm.D]
    return gt, segments, dead


def main():
    path = asset_paths(game=GAME)["smart_tagging"]
    events = st.load(path)
    print("file:", os.path.basename(path))
    print(st.summarise(events))

    restarts = st.restarts(events)
    print("\n{} restart events; by type:".format(len(restarts)))
    for name in sorted(set(event.type for event in restarts)):
        group = [event for event in restarts if event.type == name]
        print("  {:<24} n {:>2}   first {:8.2f}s   last {:8.2f}s".format(
            name, len(group), group[0].start, group[-1].start))

    kickoff_tag = [e for e in restarts if e.type == "Kick Off (Start)"][0]
    anchor = anchor_video_frame(GAMES[GAME]["kickoff"])
    print("\nsynchronisation check against the kick-off anchor in games.py:")
    print("  tag {:8.2f}s -> frame {:>7} (tag start) / {:>7} (tag end)   anchor frame {:>7}".format(
        kickoff_tag.start, st.to_frame(kickoff_tag.start), st.to_frame(kickoff_tag.end), anchor))

    gt, segments, dead = gt_window()
    print("\nground truth in the window: {} segments, {} dead-ball runs".format(
        len(segments), len(dead)))
    print("  " + "  ".join("{}-{}".format(s, t) for s, t in dead))

    in_window = [e for e in restarts if FIRST <= st.to_frame(e.start) <= LAST]
    print("\nrestart tags in the public window: {}".format(len(in_window)))
    print("  {:<20} {:>8} {:>8} {:>8} {:<16} {:>14} {:>10} {:>7} {:>9}".format(
        "type", "start s", "start f", "end f", "team", "nearest D run",
        "to D end", "in D?", "GT here"))
    for event in in_window:
        start, end = st.to_frame(event.start), st.to_frame(event.end)
        inside = any(low <= start < high for low, high in dead)
        distance, low, high = min((min(abs(start - low), abs(start - high)), low, high)
                                  for low, high in dead)
        print("  {:<20} {:>8.2f} {:>8} {:>8} {:<16} {:>14} {:>+10} {:>7} {:>9}".format(
            event.type, event.start, start, end, event.team,
            "{}-{}".format(low, high), start - high, "yes" if inside else "no",
            pm.LABELS[gt[start]]))

    print("\nrestart tags per half, against the half's video block:")
    for name in sorted(set(event.half for event in restarts)):
        group = [event for event in restarts if event.half == name]
        print("  {:<10} n {:>2}  {:8.2f}s .. {:8.2f}s  frames {:>7} .. {:>7}".format(
            name, len(group), group[0].start, group[-1].start,
            st.to_frame(group[0].start), st.to_frame(group[-1].start)))

    print("\nis every stoppage tagged? restart tags per 5-minute block of the match:")
    counts = {}
    for event in restarts:
        block = int(event.start // 300)
        counts[block] = counts.get(block, 0) + 1
    print("  " + "  ".join("{}:{}".format(k * 5, counts.get(k, 0))
                           for k in range(max(counts) + 1)))

    print("\nteam label: GT label at the first live frame after the tag (does Team = who restarts?)")
    for delay in (0, 100, 200):
        agree = total = 0
        for event in in_window:
            if event.team not in ("Barcelona", "Athletic Club"):
                continue
            frame = st.to_frame(event.start) + delay
            while frame <= LAST and gt[frame] == pm.D:
                frame += 1
            if frame > LAST:
                continue
            total += 1
            side = pm.H if event.team.startswith("Barcelona") else pm.A
            agree += int(gt[frame] == side)
        if total:
            print("  from tag +{:>3} frames: {} of {} tags agree ({:.0f}%)".format(
                delay, agree, total, 100.0 * agree / total))
        else:
            print("  from tag +{:>3} frames: no usable tags".format(delay))

    print("\nhow close are tags to dead-ball edges (all tags, whole match, window only):")
    distances = []
    for event in restarts:
        frame = st.to_frame(event.start)
        if not FIRST <= frame <= LAST:
            continue
        distances.append(min(min(abs(frame - low), abs(frame - high))
                             for low, high in dead))
    if distances:
        print("  n {}  median {:.0f} frames ({:.1f} s)  min {}  max {}".format(
            len(distances), float(np.median(distances)), float(np.median(distances)) / FPS,
            min(distances), max(distances)))


if __name__ == "__main__":
    main()
