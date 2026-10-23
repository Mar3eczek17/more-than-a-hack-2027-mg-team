"""Possession model: dead-ball evidence, a team vote, and boundary refinement.

The baseline makes two choices that cost most of its score, and they are separable:

  1. Every frame is decided from one frame. The nearest player to the ball wins, with no notion of
     whether the ball is even in play or whether the same side had it a tenth of a second earlier.
  2. Dead ball is "no ball was tracked anywhere in a 225-frame window". Real stoppages are much
     shorter than that, so the class is predicted almost never -- on the public five minutes the
     baseline scores D recall 0.10 at precision 0.95. Since a possession ends at a stoppage, that
     single failure also misplaces most spell boundaries.

Two ways of fixing that are implemented here, and they were compared on the repository's own
evaluator:

  * `predict_hybrid` -- the one that wins (spellF1 0.507 vs the baseline's 0.473 on the public
    window). A majority vote over the nearest-player team signal gives the segment structure, an
    evidence-driven Schmitt trigger gives the stoppages, and each boundary is re-placed at the
    frame where the evidence best switches sides.
  * `predict` -- emission model plus Viterbi over a three-state HMM with dwell-time transitions.
    It loses badly (0.219): a frame-level likelihood is the wrong observation model for a signal
    that needs seconds of averaging, so the match fragments. It is kept because the comparison is
    the argument, and because its dead-ball evidence is shared with the hybrid.

Nothing here is fitted to the public labels by search: the parameters are physical (metres, m/s,
seconds) or read off the public window's segment structure, and `explore_features.py` prints the
calibration tables the thresholds come from.

Label codes match evaluation/possession_metrics.py: H=0, A=1, D=2.
"""

import numpy as np

H, A, D = 0, 1, 2
N_STATES = 3
NOBODY = -1  # no tracked player close enough to be trusted as the owner

DEFAULTS = {
    # -- geometry -------------------------------------------------------------------------------
    # Control radius: a player contributes exp(-d / tau) to his team's claim on the ball. 2 m is
    # about a stride: at 1 m the claim is 0.61, at 4 m it is 0.14, at 8 m it is 0.02.
    "tau": 2.0,
    # -- dead-ball evidence ---------------------------------------------------------------------
    # A gap only counts as a stoppage once it is longer than any ball-flight occlusion in the
    # public window: the gap table there shows 3-25 frame gaps are 60% away possession and only
    # 37-42% dead, while 50+ frame gaps are 65% dead. Starting late keeps D out of live play.
    "dead_gap_soft": 20,
    "dead_gap_hard": 60,
    "fast_ball": 8.0,      # m/s; above this the ball is in flight and D is rare (5% of such frames)
    "still_speed": 1.0,    # m/s; below this the ball is stationary
    "unattended": 8.0,     # metres; further than this and no side is playing a still ball
    "dead_scale": 1.0,     # how much dead evidence outweighs one unit of player claim
    "dead_ramp": 4.0,
    # -- temporal model -------------------------------------------------------------------------
    # Mean dwell times, in frames at 25 fps, taken from the public window's own segment structure:
    # 17 H/A runs averaging 14.2 s and 4 D runs averaging 14.7 s. Both are segment statistics over
    # 21 runs, which is a far more stable quantity than any per-frame or per-spell score.
    "dwell_possession": 350.0,
    "dwell_dead": 350.0,
    # Share of possession exits that go straight to the other team rather than through a stoppage.
    # The same window settles this: of its 20 transitions, 8 touch a D run and 12 are direct
    # team-to-team. With the 0.1 this started at, every turnover found it cheaper to insert a
    # spurious dead-ball island (two cheap transitions) than to flip teams, which fragmented every
    # possession in the match into three.
    "turnover_share": 0.6,
    # -- evidence aggregation -------------------------------------------------------------------
    # Moving average over the geometry before it is turned into a probability. 15 frames is 0.6 s:
    # long enough to average away the frame-to-frame flips of "who is nearest", short enough that a
    # real turnover is not smeared across half a second.
    "evidence_frames": 15,
    "emission_weight": 1.0,
    # -- trust ----------------------------------------------------------------------------------
    # Evidence is trusted fully on a seen ball and decays as exp(-gap / trust_frames) inside a gap.
    "trust_frames": 12.0,
    # -- output ---------------------------------------------------------------------------------
    # Spells shorter than this are relabelled into their neighbour: 3 frames is 0.12 s, below any
    # real possession, and removing them cannot change a real spell's identity.
    "min_spell": 3,
    "min_dead": 3,
    # -- hybrid mode: majority-vote teams + an evidence-driven dead overlay ----------------------
    "vote_window": 112,     # the baseline's own window; the team signal is unchanged from it
    "dead_enter": 0.55,     # evidence needed to declare a stoppage
    "dead_stay": 0.30,      # evidence below which the stoppage is over (the hysteresis band)
    "dead_min_frames": 100,  # shortest dead stretch worth keeping: 4 s, and every GT stoppage in
                            # the public window is at least 138 frames. Shorter candidates are the
                            # tracking losing the ball, not a stoppage.
    "dead_max_hole": 15,    # holes this short inside a stoppage are closed
    "dead_grow": 40,        # how far a stoppage may expand into weaker evidence, in frames
    "refine_shift": 30,     # +-frames a boundary may move when re-placed on frame-level evidence
    # -- Smart Tagging restarts (only applied when tag frames are supplied) ----------------------
    # A tag is evidence that the ball is dead *at* the tag (every in-window tag is inside a GT
    # stoppage), and says nothing about where play resumes (they sit 69-305 frames before the end).
    # So the hint is a short window around the tag: 30 frames before, 60 after.
    "tag_before": 30,
    "tag_after": 60,
    # A tag only confirms a stoppage where the tracker's own evidence is at least this strong.
    # 0.0 lets the tag act alone; a small positive value makes it a tie-breaker, which is what the
    # public window supports (one tag landed on a correctly detected stoppage, one on a miss).
    "tag_min_evidence": 0.0,
    # -- hysteresis team evidence (used by predict_hysteresis) -----------------------------------
    # A challenger must be closer by `switch_margin` metres and keep it for `switch_frames` frames.
    # 1.5 m is about the width of a player's control zone, and 8 frames is 0.3 s: a tackle or an
    # interception takes longer than that to become visible, so this cannot block a real turnover.
    "switch_margin": 1.5,
    "switch_frames": 8,
    # Beyond `owner_radius` metres nobody is playing the ball, so the geometry is not trusted to
    # change possession -- the ball is in flight and the passing side keeps it.
    "owner_radius": 12.0,
    # -- extra splits (used by predict_hybrid) ---------------------------------------------------
    # A vote segment can swallow a short opposite possession. These stretches are emphatic and
    # persistent enough to be believed on their own: 2 m clear for 75 frames (3 s). 0 disables.
    "split_min_frames": 0,
    "split_margin": 2.0,
    # -- stationary-ball cue (restarts) ----------------------------------------------------------
    # A ball that has not moved for this many frames is a restart. 0 disables the cue. The default
    # is 0 until the sweep in sensitivity.py shows it pays for itself on the public window.
    "still_long_frames": 0,
    # -- submission hygiene ----------------------------------------------------------------------
    # Write D outside the two halves (pre-match, halftime, after the final whistle). Those frames
    # are never graded, so this cannot change a metric; it stops the file claiming possession while
    # the match is not being played.
    "respect_halves": False,
}


def params(**overrides):
    """DEFAULTS plus overrides, validated so a typo cannot pass silently."""
    unknown = sorted(set(overrides) - set(DEFAULTS))
    if unknown:
        raise KeyError("unknown parameter(s): {}".format(unknown))
    out = dict(DEFAULTS)
    out.update(overrides)
    return out


def _ramp(values, low, high):
    """0 below `low`, 1 above `high`, linear between. NaN -> 0 (absence is not evidence here)."""
    with np.errstate(invalid="ignore"):
        out = (values - low) / float(high - low)
    out = np.where(np.isfinite(out), np.clip(out, 0.0, 1.0), 0.0)
    return out


def smooth_nan(values, width):
    """Centred NaN-aware moving average, edges clipped. `width` <= 1 returns the input.

    Possession evidence is not independent frame to frame: the tracker's idea of who is nearest
    flips for a frame or two whenever the ball passes between two players, and a per-frame decision
    turns each of those flips into a new possession. Averaging over ~0.5 s removes that noise while
    moving a real boundary by at most half the window -- and unlike a majority vote it does not
    discard the magnitude of the evidence.
    """
    values = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return values
    good = np.isfinite(values)
    filled = np.where(good, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0], np.cumsum(good)))
    index = np.arange(len(values))
    half = width // 2
    low = np.maximum(0, index - half)
    high = np.minimum(len(values), index + half + 1)
    total = cumulative[high] - cumulative[low]
    count = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def tag_hint(n_frames, tag_frames, before, after):
    """Frames covered by a Smart Tagging restart hint, as a 0/1 array.

    Measured, not assumed: every restart tag inside the public window falls *inside* a ground-truth
    dead-ball run, but 69 to 305 frames before that run ends. So a tag is evidence that the ball is
    dead at the tag, and is not evidence about where play resumes -- the hint is therefore a window
    around the tag, never a boundary.

    `before` and `after` are deliberately asymmetric and small: a tag leads the true event by about
    3.5 s (kick-off tag frame 11097 against the kick-off anchor at 11186), and the stoppage continues
    for an unmeasured time after it.
    """
    hint = np.zeros(n_frames, dtype=np.float64)
    if not tag_frames:
        return hint
    for frame in tag_frames:
        low = max(0, int(frame) - int(before))
        high = min(n_frames, int(frame) + int(after) + 1)
        hint[low:high] = 1.0
    return hint


def still_run_length(features, param):
    """Consecutive frames the ball has been (nearly) stationary, as a run length.

    A restart is the one stoppage the other cues miss: the ball is on the spot, visible, with the
    taker standing over it, so "out of play", "no ball" and "nobody near it" all fail. What does
    hold is that the ball does not move -- for seconds. In live play it never does: it is either
    being carried (speed above walking pace) or travelling.

    Uses the smoothed ball speed, which is held across short tracking gaps, so a brief occlusion
    during a stoppage does not reset the count.
    """
    speed = features["ball_speed"]
    still = np.isfinite(speed) & (speed < param["still_speed"])
    run = np.zeros(len(still), dtype=np.float64)
    count = 0
    for frame in range(len(still)):
        count = count + 1 if still[frame] else 0
        run[frame] = count
    return run


def dead_evidence(features, param):
    """How strongly the tracking says "the ball is not in play", in [0, 1].

    Four independent cues, each a statement about the world rather than a tuned constant:

      out of play   the ball is outside the pitch rectangle -- dead by definition;
      unattended    nobody is within `unattended` metres and the ball is not moving, so no side is
                    playing it. A stationary ball with a player standing over it is a restart, not
                    a dead ball, which is why proximity vetoes this cue rather than adding to it;
      gap           the ball has not been tracked for a while *and* it was slow when last seen. A
                    fast disappearance is a pass in flight and the passing side still has the ball;
      empty         no player was tracked at all, so no claim exists anywhere.

    Cues are combined with max, not summed: they describe the same underlying event, and adding
    them would let three weak hints outvote one certain one.
    """
    visible = np.isfinite(features["d_home"]) | np.isfinite(features["d_away"])
    gap = features["gap"].astype(np.float64)
    speed = features["ball_speed"].astype(np.float64)
    nearest = features["nearest"].astype(np.float64)
    known = np.isfinite(speed)

    outside = np.where(visible, outside_pitch_mask(features), 0.0).astype(np.float64)

    # Still: 1 at rest, 0 at `still_speed` and above.
    still = np.where(known, np.clip((param["still_speed"] - np.nan_to_num(speed))
                                    / (0.5 * param["still_speed"]), 0.0, 1.0), 0.0)
    far = _ramp(np.nan_to_num(nearest, nan=0.0), param["unattended"],
                param["unattended"] + param["dead_ramp"])
    unattended = np.where(visible, np.minimum(still, far), 0.0)

    gap_score = _ramp(gap, param["dead_gap_soft"], param["dead_gap_hard"])
    # A gap that opened with a fast ball is a pass: scale the gap cue away there.
    fast = np.where(known, np.clip((np.nan_to_num(speed) - 0.5 * param["fast_ball"])
                                   / (0.5 * param["fast_ball"]), 0.0, 1.0), 0.0)
    gap_score = gap_score * (1.0 - fast)

    empty = (features["n_players"] == 0).astype(np.float64)

    cues = [outside, unattended, gap_score, empty]
    # The ball sitting still for seconds is a restart: it is the only stoppage cue that fires when
    # the ball is visible with a player standing over it. Off by default (`still_long_frames` 0).
    if param["still_long_frames"]:
        hold = still_run_length(features, param)
        cues.append(_ramp(hold, param["still_long_frames"], param["still_long_frames"] * 2.0))

    tracker_dead = np.clip(np.maximum.reduce(cues), 0.0, 1.0)

    # Smart Tagging restarts, when the caller supplied them. A tag is not allowed to declare a
    # stoppage on its own: it only confirms one where the tracker already has some doubt
    # (`tag_min_evidence`), which keeps a mis-tagged event from cutting a possession in half.
    if "tag_hint" in features:
        gate = (tracker_dead >= param["tag_min_evidence"]).astype(np.float64)
        cues.append(features["tag_hint"] * gate)

    return np.clip(np.maximum.reduce(cues), 0.0, 1.0)


def outside_pitch_mask(features, margin=0.0):
    ball = features["ball"]
    x, y = ball[:, 0], ball[:, 1]
    inside = np.isfinite(x) & np.isfinite(y)
    out = ((x < -margin) | (x > 1.0 + margin) | (y < -margin) | (y > 1.0 + margin))
    return out & inside


def emissions(features, param):
    """(n, 3) log probability of H, A, D per frame.

    For a frame with the ball seen, each side's claim is exp(-d/tau) and the dead-ball claim is
    `dead_scale * dead`, so the three compete on one scale: dead evidence of 1 outweighs a player
    1 tau away, and evidence of 0 makes D unreachable no matter how far both sides are. That second
    half matters -- during a long pass the ball is far from everyone, and the correct answer is
    still the passing side.

    Frames inside a tracking gap carry no positional claim at all: the observation is mixed with a
    uniform distribution with weight exp(-gap / trust_frames). Forward-filling the last known
    distance instead would assert that the geometry has not changed, which over a two-second gap it
    certainly has.
    """
    visible = np.isfinite(features["d_home"]) | np.isfinite(features["d_away"])
    d_home = np.nan_to_num(smooth_nan(features["d_home"], param["evidence_frames"]), nan=50.0)
    d_away = np.nan_to_num(smooth_nan(features["d_away"], param["evidence_frames"]), nan=50.0)
    dead = smooth_nan(features["dead"], param["evidence_frames"])
    dead = np.clip(np.where(np.isfinite(dead), dead, 0.0), 0.0, 1.0)

    claim_home = np.exp(-d_home / param["tau"])
    claim_away = np.exp(-d_away / param["tau"])
    claim_dead = param["dead_scale"] * dead

    raw = np.stack([claim_home * (1.0 - dead),
                    claim_away * (1.0 - dead),
                    claim_dead], axis=1)
    raw += 1e-9
    observed = raw / raw.sum(axis=1, keepdims=True)

    trust = np.where(visible, 1.0, np.exp(-features["gap"] / param["trust_frames"]))
    mixed = trust[:, None] * observed + (1.0 - trust[:, None]) / N_STATES
    return param["emission_weight"] * np.log(np.clip(mixed, 1e-12, None))


def transition_matrix(param):
    """3x3 log transition matrix from dwell times.

    Rows are "from", columns "to". A possession ends either in a stoppage (the common case in
    football: the ball goes out, a foul is given) or directly in a turnover; `turnover_share` sets
    the split. A stoppage ends into either side with equal probability -- the tracking cannot tell
    who restarts play, and pretending otherwise would be a guess dressed as a prior.

    Because these are dwell times in frames rather than a smoothing window, the model keeps a
    segment's interior intact while still allowing a boundary where the evidence is strong. A
    majority vote cannot do that: it moves every boundary by the window's half-width.
    """
    possession_exit = 1.0 / max(param["dwell_possession"], 1.0)
    dead_exit = 1.0 / max(param["dwell_dead"], 1.0)
    turnover = param["turnover_share"]

    matrix = np.array([
        [1.0 - possession_exit, turnover * possession_exit,
         (1.0 - turnover) * possession_exit],
        [turnover * possession_exit, 1.0 - possession_exit,
         (1.0 - turnover) * possession_exit],
        [0.5 * dead_exit, 0.5 * dead_exit, 1.0 - dead_exit],
    ])
    matrix = np.clip(matrix, 1e-12, None)
    return np.log(matrix)


def viterbi(log_emission, log_transition):
    """Most likely state sequence. O(n * 9), a second or two for a whole match."""
    n_frames, n_states = log_emission.shape
    score = log_emission[0].copy()
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    columns = np.arange(n_states)
    for frame in range(1, n_frames):
        candidates = score[:, None] + log_transition  # (from, to)
        best = candidates.argmax(axis=0)
        back[frame] = best
        score = candidates[best, columns] + log_emission[frame]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(score.argmax())
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def runs(codes):
    """[(start, stop, code)] maximal runs of equal codes, stop exclusive."""
    edges = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1, [len(codes)]))
    return [(int(edges[i]), int(edges[i + 1]), int(codes[edges[i]]))
            for i in range(len(edges) - 1)]


def merge_short_runs(codes, min_length):
    """Relabel runs shorter than `min_length` into the preceding run (or the following one).

    The HMM already penalises flicker, but a 2-frame island can still survive a weak emission tie.
    At 25 fps two frames is 0.08 s: too short to be a possession or a stoppage, and exactly the
    kind of artefact that turns one true possession into two predicted ones. Merging backwards
    rather than forwards is a deliberate asymmetry: possession is sticky, and the frame before a
    spurious island is more likely to be right than the frame after it.
    """
    codes = codes.astype(np.int8, copy=True)
    for _ in range(3):
        changed = False
        pieces = runs(codes)
        for index, (start, stop, code) in enumerate(pieces):
            if stop - start >= min_length[code]:
                continue
            if index > 0:
                codes[start:stop] = pieces[index - 1][2]
            elif index + 1 < len(pieces):
                codes[start:stop] = pieces[index + 1][2]
            else:
                continue
            changed = True
        if not changed:
            break
    return codes


def prepare(features, param=None, pitch=None, tag_frames=None):
    """Add the derived signals the model needs. Returns a new dict; `features` is not modified.

    `tag_frames` are Smart Tagging restart frames (see smart_tagging.py). Passing them adds the
    hint cue described in `tag_hint`; passing none leaves the model exactly as it was, which is what
    the control variants use.
    """
    import tracking  # local: tracking does not depend on this module, but keep the import cheap

    param = param or params()
    if pitch is None:
        from games import GAME, GAMES
        pitch = GAMES[GAME]["pitch"]

    out = dict(features)
    out["gap"] = tracking.gap_to_observation(features)
    out["ball_speed"] = tracking.ball_speed(features, pitch)
    if tag_frames:
        out["tag_hint"] = tag_hint(len(features["nearest"]), tag_frames,
                                   param["tag_before"], param["tag_after"])
    out["dead"] = dead_evidence(out, param)
    return out


def majority_team(features, window):
    """Nearest-player team labels smoothed by a centred majority vote; 2 where no vote exists.

    This is the baseline's proven team signal -- it is noisy frame to frame but unbiased over a
    window of a few seconds -- kept here as an alternative to the HMM's emission model, so the two
    can be compared on the same dead-ball logic.
    """
    team = features["nearest_team"]
    home = team == H
    away = team == A
    index = np.arange(len(team))
    home_cumulative = np.concatenate(([0], np.cumsum(home)))
    away_cumulative = np.concatenate(([0], np.cumsum(away)))
    low = np.maximum(0, index - window)
    high = np.minimum(len(team), index + window + 1)
    home_count = home_cumulative[high] - home_cumulative[low]
    away_count = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(team), D, dtype=np.int8)
    seen = (home_count + away_count) > 0
    labels[seen] = np.where(home_count[seen] >= away_count[seen], H, A)
    return labels


def dead_mask(dead, param):
    """Schmitt trigger over the smoothed dead-ball evidence: a state, not a per-frame score.

    Entering dead ball needs `dead_enter` evidence; leaving it only needs the evidence to fall
    below `dead_stay`. The gap between the two is what keeps a stoppage in one piece when a player
    briefly walks over the ball, and holes shorter than `dead_max_hole` are closed for the same
    reason. Short excursions in both directions are discarded (`dead_min_frames`), because a
    0.3 s dead island inside live play is a tracking artefact, not a stoppage.
    """
    enter = dead >= param["dead_enter"]
    stay = dead >= param["dead_stay"]

    # Keep only the excursions long enough to be a stoppage.
    mask = np.zeros(len(dead), dtype=bool)
    for start, stop, code in runs(enter.astype(np.int8)):
        if code == 1 and stop - start >= param["dead_min_frames"]:
            mask[start:stop] = True

    # Close short holes between two dead stretches where the evidence never fully dropped.
    pieces = runs(mask.astype(np.int8))
    for previous, following in zip(pieces, pieces[1:]):
        gap_start, gap_stop = previous[1], following[0]
        if (following[2] == 1 and previous[2] == 1 and gap_stop - gap_start <= param["dead_max_hole"]
                and stay[gap_start:gap_stop].all()):
            mask[gap_start:gap_stop] = True

    # Grow each stretch outwards while the evidence is still above the exit threshold. The test is
    # the run's own code, not `mask[start]`: after one stretch is grown, the *next* run starts
    # inside the grown region, and testing the mutated mask would cascade one expansion into every
    # later run until the whole match was dead.
    limit = param["dead_grow"]
    for start, stop, code in runs(mask.astype(np.int8)):
        if code != 1:
            continue
        left = start
        while left > 0 and start - left < limit and stay[left - 1] and not mask[left - 1]:
            left -= 1
        right = stop
        while right < len(mask) and right - stop < limit and stay[right] and not mask[right]:
            right += 1
        mask[left:right] = True
    return mask


def hysteresis_team(features, param):
    """Team labels from distance margins plus a bounded temporal hysteresis.

    Why not "whoever is nearest": in this tracking the nearest player to the ball is a defender
    stepping across a pass as often as it is the player in control, and on the public window the
    nearest player agrees with the winning side's ground-truth team on only 80% of home and 75% of
    away frames (before the margin filter). A single frame therefore cannot decide possession, and
    a mean vote over seconds throws away *when* the change happened -- which is exactly what
    `spellF1` measures.

    The rule here is a state machine with three guards:

      margin        the challenger must be closer by at least `switch_margin` metres before a
                    switch is even considered, so two players level with the ball do not trade it;
      persistence   the challenger must win for `switch_frames` consecutive frames, which rejects
                    the tracker's one-frame identity swaps;
      trust radius  if nobody is within `owner_radius` metres the geometry is not trusted to switch
                    at all, and possession stays with whoever had it. A ball in flight is not a
                    turnover -- the passing side keeps it until an opponent is actually near.

    Crucially the hysteresis is *bounded*: a real turnover needs `switch_frames` (default 0.3 s) at
    a real margin, far less than any genuine change of possession takes to become visible, so it
    cannot block a tackle. And it never forces a switch through `D`: `H -> A` directly is allowed.
    """
    d_home = smooth_nan(features["d_home"], param["evidence_frames"])
    d_away = smooth_nan(features["d_away"], param["evidence_frames"])
    n_frames = len(d_home)
    codes = np.full(n_frames, D, dtype=np.int8)

    margin = float(param["switch_margin"])
    needed = max(1, int(param["switch_frames"]))
    radius = float(param["owner_radius"])

    current = NOBODY
    challenger = NOBODY
    streak = 0
    for frame in range(n_frames):
        home, away = d_home[frame], d_away[frame]
        if not (np.isfinite(home) and np.isfinite(away)):
            # Nothing to judge by: keep the previous answer rather than inventing a change.
            codes[frame] = current if current != NOBODY else D
            continue

        nearest = min(home, away)
        closer = H if home <= away else A

        if current == NOBODY:
            current = closer
            codes[frame] = current
            continue

        if closer != current and nearest <= radius and abs(home - away) >= margin:
            if challenger == closer:
                streak += 1
            else:
                challenger, streak = closer, 1
            if streak >= needed:
                current, challenger, streak = closer, NOBODY, 0
        else:
            challenger, streak = NOBODY, 0

        codes[frame] = current
    return codes


def predict_hysteresis(features, param=None, pitch=None):
    """Hysteresis team evidence + the same dead-ball overlay and boundary refinement as the vote."""
    param = param or params()
    if "dead" not in features:
        features = prepare(features, param, pitch)

    labels = hysteresis_team(features, param)
    mask = dead_mask(features["dead"], param)
    labels[mask] = D
    labels = merge_short_runs(labels, [param["min_spell"], param["min_spell"], param["min_dead"]])
    if param["refine_shift"]:
        labels = refine_boundaries(labels, emissions(features, param),
                                   max_shift=int(param["refine_shift"]))
        labels = merge_short_runs(labels, [param["min_spell"], param["min_spell"], param["min_dead"]])
    return labels


def strong_side_runs(features, param):
    """Stretches where one side is clearly and persistently closer to the ball.

    A majority vote over +/-112 frames is an integrator: it gives the segment structure, but by
    construction it cannot contain a possession shorter than its own half-width, and the public
    window has possessions of 19 to 87 frames. This function finds the opposite: stretches where
    the geometry is emphatic (`split_margin` metres, for `split_min_frames` frames) and which the
    vote may therefore have swallowed.

    Returns [(start, stop, code)] with code in {H, A}, longest first.
    """
    d_home = smooth_nan(features["d_home"], param["evidence_frames"])
    d_away = smooth_nan(features["d_away"], param["evidence_frames"])
    margin = float(param["split_margin"])

    with np.errstate(invalid="ignore"):
        home_clear = d_home + margin <= d_away
        away_clear = d_away + margin <= d_home
    strong = np.full(len(d_home), NOBODY, dtype=np.int8)
    strong[home_clear] = H
    strong[away_clear] = A

    out = []
    for start, stop, code in runs(strong):
        if code in (H, A) and stop - start >= param["split_min_frames"]:
            out.append((start, stop, code))
    return out


def apply_splits(codes, features, param):
    """Relabel persistent opposite stretches inside a vote segment, splitting it.

    Only applied inside a segment that the vote gave to the *other* team, and only for stretches
    that survive `split_min_frames`, so this can add a short possession the vote could not express
    but cannot fragment a segment into noise.
    """
    if not param["split_min_frames"]:
        return codes
    out = codes.copy()
    for start, stop, code in strong_side_runs(features, param):
        if out[start] != code:
            out[start:stop] = code
    return merge_short_runs(out, [param["min_spell"], param["min_spell"], param["min_dead"]])


def outside_play_mask(n_frames, game=None):
    """True on frames outside the two halves, using the published half boundaries.

    `games.half_video_blocks` is data the repository publishes as fact about the video ("boundaries,
    not answers") and every such frame is `X` in the ground truth, so this cannot change a score.
    What it does change is the file's internal consistency: without it the model happily reports a
    6-minute possession during halftime, because the players are still on the pitch and the tracking
    still sees a ball. A submission claiming possession while the match is not being played is
    simply wrong, whether or not that frame is graded.
    """
    import games as games_module

    blocks = games_module.half_video_blocks(game or games_module.GAME)
    in_play = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        in_play[first:last + 1] = True
    return ~in_play


def predict_hybrid(features, param=None, pitch=None):
    """Majority-vote team labels with an evidence-driven dead-ball overlay.

    The two halves of the problem have different statistics. Deciding *which team* is a noisy
    estimate that needs averaging over seconds; deciding *whether the ball is in play at all* is a
    state that can be read from one clear cue (out of play, nobody near a still ball, a long gap
    that opened with a slow ball) and should not be averaged with the same brush. So the team vote
    keeps the baseline's window, and the dead stretches come from `dead_mask`, where they are
    placed by evidence rather than by where a vote happened to tip.
    """
    param = param or params()
    if "dead" not in features:
        features = prepare(features, param, pitch)

    labels = majority_team(features, param["vote_window"])
    mask = dead_mask(features["dead"], param)
    labels[mask] = D
    labels = merge_short_runs(labels, [param["min_spell"], param["min_spell"], param["min_dead"]])
    labels = apply_splits(labels, features, param)
    if param["refine_shift"]:
        labels = refine_boundaries(labels, emissions(features, param),
                                   max_shift=int(param["refine_shift"]))
        labels = merge_short_runs(labels, [param["min_spell"], param["min_spell"], param["min_dead"]])
    if param["respect_halves"]:
        # Never claim possession while the match is not being played. Those frames are X in every
        # ground truth, so the score cannot move; the file just stops asserting something false
        # about halftime (the audit found a 610-second "possession" there).
        labels[outside_play_mask(len(labels))] = D
    return labels


def refine_boundaries(codes, log_emission, max_shift=60, min_side=5):
    """Move each segment boundary to the frame where the evidence best switches sides.

    A majority vote is good at *how many* segments there are and roughly where they are, and bad
    at where exactly one ends: every boundary is pulled by up to half the window, which on a 14 s
    possession is a fifth of its length and most of the IoU that spellF1 measures. The vote is
    left to decide the structure; this pass re-places each boundary using the frame-level
    evidence, searching +-`max_shift` frames for the split that maximises the evidence for the two
    labels it separates.

    It is a local, bounded version of the same Viterbi idea: the neighbours are fixed, so this
    cannot invent or delete a segment, and it cannot move a boundary outside its own segments.
    """
    pieces = runs(codes)
    if len(pieces) < 2:
        return codes

    cumulative = np.concatenate([np.zeros((1, N_STATES)),
                                 np.cumsum(log_emission, axis=0)], axis=0)
    edges = [0] + [piece[1] for piece in pieces[:-1]] + [len(codes)]
    new_edges = list(edges)

    for index in range(1, len(pieces)):
        left_code, right_code = pieces[index - 1][2], pieces[index][2]
        if left_code == right_code:
            continue
        low = max(edges[index] - max_shift, edges[index - 1] + min_side)
        high = min(edges[index] + max_shift, edges[index + 1] - min_side)
        if high <= low:
            continue
        candidates = np.arange(low, high + 1)
        left_score = cumulative[candidates, left_code] - cumulative[edges[index - 1], left_code]
        right_score = cumulative[edges[index + 1], right_code] - cumulative[candidates, right_code]
        new_edges[index] = int(candidates[np.argmax(left_score + right_score)])

    out = np.empty_like(codes)
    for index, code in enumerate(piece[2] for piece in pieces):
        out[new_edges[index]:new_edges[index + 1]] = code
    return out


def predict(features, param=None, frame_wise=False, pitch=None):
    """Per-frame H/A/D codes for the whole match, from prepared or raw features."""
    param = param or params()
    if "dead" not in features:
        features = prepare(features, param, pitch)
    log_emission = emissions(features, param)
    if frame_wise:
        path = log_emission.argmax(axis=1).astype(np.int8)
    else:
        path = viterbi(log_emission, transition_matrix(param))
    return merge_short_runs(path, [param["min_spell"], param["min_spell"], param["min_dead"]])
