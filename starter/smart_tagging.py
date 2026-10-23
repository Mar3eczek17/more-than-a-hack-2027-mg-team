"""Smart Tagging: the match's own event markers, as frames on the video timeline.

What the file is
----------------
`..._smart_tagging.xml` is a list of `<instance>` elements. Each has an `<ID>`, a `<start>` and
`<end>` in **seconds from the start of the video**, a `<code>` (one category: `SET PIECES`), and a
set of `<label>` children with a `<group>` and a `<text>`. The groups that matter here are `Type`
(what the restart was: throw-in, corner, goal kick, free kick, kick off, ...), `Team` (who took it,
or the literal `N/A`), and `Half`.

Two things are deliberately *not* assumed:

  * that `<start>` is the frame of the restart. The instances are five seconds long, so the event
    sits at one end of a window or somewhere inside it; which end is measured in
    `explore_smart_tagging.py` against the tracking, not guessed here.
  * that `Team` names a side. The kick-off instances carry `N/A`, and the label is free text, so it
    is exposed as-is and any use of it has to justify itself.

Usage
-----
    import smart_tagging as st
    events = st.load(path)
    restarts = st.restarts(events)          # only the types that actually occur
    frame = st.to_frame(event.start)

    python starter/smart_tagging.py         # print what the file contains
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

FPS = 25.0

# Restart types, exactly as the XML spells them (`Throw-in`, `Corner Kick`). Anything not in this
# set is not treated as a restart -- inventing types would be inventing data. All six occur.
RESTART_TYPES = (
    "Kick Off (Start)",
    "Kick Off (After Goal)",
    "Throw-in",
    "Corner Kick",
    "Goal Kick",
    "Free Kick",
)


class Event(object):
    """One Smart Tagging instance. Times are seconds; `labels` is the raw group -> text mapping."""

    __slots__ = ("ident", "start", "end", "code", "labels")

    def __init__(self, ident, start, end, code, labels):
        self.ident = ident
        self.start = start
        self.end = end
        self.code = code
        self.labels = labels

    @property
    def type(self):
        return self.labels.get("Type", "")

    @property
    def team(self):
        return self.labels.get("Team", "")

    @property
    def half(self):
        return self.labels.get("Half", "")

    @property
    def is_restart(self):
        return self.type in RESTART_TYPES

    def __repr__(self):
        return "Event({} {:.2f}-{:.2f}s {!r} team={!r})".format(
            self.ident, self.start, self.end, self.type, self.team)


def load(path):
    """Parse the XML into `Event` objects, in file order."""
    if not os.path.isfile(path):
        raise IOError("Smart Tagging file not found: {}".format(path))
    root = ET.parse(path).getroot()
    events = []
    for instance in root.iter("instance"):
        ident = (instance.findtext("ID") or "").strip()
        start = float(instance.findtext("start"))
        end = float(instance.findtext("end"))
        code = (instance.findtext("code") or "").strip()
        labels = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip()
            text = (label.findtext("text") or "").strip()
            labels[group] = text
        events.append(Event(ident, start, end, code, labels))
    return events


def to_frame(seconds, offset=0.0):
    """Video frame for a Smart Tagging timestamp. `offset` is added before rounding (seconds)."""
    return int(round((seconds + offset) * FPS))


def restarts(events):
    """The events whose Type is a restart, sorted by time."""
    return sorted((event for event in events if event.is_restart), key=lambda e: e.start)


def by_half(events):
    """{half: [events]} using the Half label, which is the only half information in the file."""
    out = {}
    for event in events:
        out.setdefault(event.half, []).append(event)
    return out


def summarise(events):
    """Human-readable summary of what the file actually contains."""
    lines = ["{} events".format(len(events)),
             "codes: {}".format(dict(Counter(e.code for e in events))),
             "types: {}".format(dict(Counter(e.type for e in events))),
             "halves: {}".format(dict(Counter(e.half for e in events))),
             "teams: {}".format(dict(Counter(e.team for e in events))),
             "durations: {}".format(dict(Counter(round(e.end - e.start, 2) for e in events)))]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import games

    path = games.asset_paths()["smart_tagging"]
    events = load(path)
    print(summarise(events))
    print("\nrestart types present: {}".format(
        sorted(set(e.type for e in restarts(events)))))
    print("\nfirst five restarts, video frames at 25 fps:")
    for event in restarts(events)[:5]:
        print("  {:>6.2f}s -> frame {:>7}  {:<22} team {}".format(
            event.start, to_frame(event.start), event.type, event.team))
