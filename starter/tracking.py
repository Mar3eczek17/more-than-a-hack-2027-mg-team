"""Cached, memory-light access to the ATD tracking file, and the per-frame features it supports.

Why this module exists
----------------------
`baseline.py` re-parses the raw text with a Python loop over 46 track slots on every run. That is
the right choice for the reference implementation -- it shows the format with nothing in the way --
but an experiment pipeline that compares twenty variants cannot pay a minute each time.

Here the file is parsed in chunks by numpy (the separators `:` and `;` become commas, and `NaN`
parses to NaN for free), the per-frame features the model needs are reduced on the fly, and only
those features are cached. The cache is a few megabytes and loads in well under a second, so every
experiment after the first is cheap. The 46 x 3 position table itself is never kept: 199k frames x
46 slots x 3 would be 220 MB of arrays, and every feature the model uses is a reduction over it.

Coordinates are normalised 0-1 on a 105.3 x 68.0 m pitch. Everything here converts to metres
before comparing anything: raw normalised distance is 55% too large along the short axis, which is
exactly where the ball spends its time near the touchline.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

# Params
N_SLOTS = 46  # the tactical ATD; the broadcast file has 51, so it is not interchangeable
COLUMNS = 1 + 3 * N_SLOTS + 3  # frame, per-slot x/y/speed, ball x/y/speed
CHUNK_LINES = 8000
CACHE_VERSION = 3  # bump when the feature set changes, so a stale cache cannot be read as new
CACHE_DIRNAME = "_cache"

HOME, AWAY, NOBODY = 0, 1, -1
FPS = 25.0


def team_of_each_track(meta_path):
    """0 for the home side, 1 for away -- the convention baseline.py uses, first Team wins."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots],
                    dtype=np.int8)


def _parse_block(lines):
    """One chunk of raw lines -> (frame ids, (n, 46, 3) players, (n, 3) ball), normalised units."""
    text = ",".join(lines).replace(":", ",").replace(";", ",")
    values = np.fromstring(text, dtype=np.float64, sep=",")  # noqa: text mode, not binary
    if values.size % COLUMNS:
        raise ValueError("chunk of {} lines parsed to {} values, not a multiple of {}".format(
            len(lines), values.size, COLUMNS))
    table = values.reshape(-1, COLUMNS)
    frames = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * N_SLOTS].reshape(-1, N_SLOTS, 3)
    ball = table[:, 1 + 3 * N_SLOTS:]
    return frames, players, ball


def iter_chunks(raw_path, chunk_lines=CHUNK_LINES):
    """Yield (frames, players, ball) blocks, keeping memory flat on the 196 MB source file."""
    block = []
    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            block.append(line)
            if len(block) >= chunk_lines:
                yield _parse_block(block)
                block = []
    if block:
        yield _parse_block(block)


def _cache_path(raw_path, cache_dir):
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    folder = cache_dir or os.path.join(os.path.dirname(os.path.abspath(raw_path)), CACHE_DIRNAME)
    return os.path.join(folder, "{}_features_v{}.npz".format(stem, CACHE_VERSION))


FEATURE_KEYS = ("ball", "d_home", "d_away", "nearest", "margin", "nearest_team",
                "n_players", "n_near3", "n_near10")


def build_features(raw_path, meta_path, n_frames, pitch, cache_dir=None, use_cache=True,
                   verbose=False):
    """Per-frame geometry for the whole match, cached as .npz.

    Returned keys, all length `n_frames`:
      ball          (n, 3) normalised x, y and the ATD's own speed estimate (NaN if unseen)
      d_home        metres to the closest tracked home player  (NaN if none tracked)
      d_away        same for away
      nearest       the smaller of the two
      margin        d_other - d_nearest, NaN when only one side has a tracked player
      nearest_team  HOME / AWAY / NOBODY
      n_players     tracked slots at all, independent of the ball -- a tracking-quality proxy
      n_near3       tracked players within 3 m of the ball (a "crowded" proxy)
      n_near10      tracked players within 10 m (a "shape around the ball" proxy)
    """
    path = _cache_path(raw_path, cache_dir)
    if use_cache and os.path.isfile(path):
        with np.load(path) as cached:
            data = {key: cached[key] for key in cached.files}
        if len(data["nearest"]) == n_frames:
            return data

    team_of_slot = team_of_each_track(meta_path)
    length, width = pitch
    home_slots = team_of_slot == HOME
    away_slots = team_of_slot == AWAY

    data = {
        "ball": np.full((n_frames, 3), np.nan, dtype=np.float32),
        "d_home": np.full(n_frames, np.nan, dtype=np.float32),
        "d_away": np.full(n_frames, np.nan, dtype=np.float32),
        "nearest": np.full(n_frames, np.nan, dtype=np.float32),
        "margin": np.full(n_frames, np.nan, dtype=np.float32),
        "nearest_team": np.full(n_frames, NOBODY, dtype=np.int8),
        "n_players": np.zeros(n_frames, dtype=np.int16),
        "n_near3": np.zeros(n_frames, dtype=np.int16),
        "n_near10": np.zeros(n_frames, dtype=np.int16),
    }

    for frames, players, ball in iter_chunks(raw_path):
        index = frames - 1
        ok = (index >= 0) & (index < n_frames)
        index, players, ball = index[ok], players[ok], ball[ok]

        bx = ball[:, 0] * length
        by = ball[:, 1] * width
        px = players[:, :, 0] * length
        py = players[:, :, 1] * width
        # (n, 46) distances; NaN coordinates stay NaN and can never win a nanmin.
        distance = np.hypot(px - bx[:, None], py - by[:, None])

        with np.errstate(all="ignore"):
            d_home = np.nanmin(distance[:, home_slots], axis=1)
            d_away = np.nanmin(distance[:, away_slots], axis=1)
        finite_home = np.isfinite(d_home)
        finite_away = np.isfinite(d_away)
        both = finite_home & finite_away

        nearest_team = np.full(len(index), NOBODY, dtype=np.int8)
        nearest_team[both] = np.where(d_home[both] <= d_away[both], HOME, AWAY)
        nearest_team[finite_home & ~finite_away] = HOME
        nearest_team[finite_away & ~finite_home] = AWAY

        data["ball"][index] = ball.astype(np.float32)
        data["d_home"][index] = d_home
        data["d_away"][index] = d_away
        data["nearest"][index] = np.fmin(d_home, d_away)
        data["margin"][index] = np.where(both, np.abs(d_home - d_away), np.nan)
        data["nearest_team"][index] = nearest_team
        # Counted from the player block, not from the ball-relative distances: a frame where the
        # ball is unseen still has players on the pitch, and confusing the two made "nobody
        # tracked" fire on every unseen ball frame.
        data["n_players"][index] = np.isfinite(players[:, :, 0]).sum(axis=1)
        data["n_near3"][index] = np.nansum(distance < 3.0, axis=1)
        data["n_near10"][index] = np.nansum(distance < 10.0, axis=1)
        if verbose:
            print("  ... through frame {}".format(int(frames[-1])), flush=True)

    folder = os.path.dirname(path)
    if use_cache and folder and not os.path.isdir(folder):
        os.makedirs(folder)
    if use_cache:
        np.savez(path, **data)
    return data


# -- derived signals ----------------------------------------------------------------------------

def ball_visible(features):
    ball = features["ball"]
    return np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])


def gap_to_observation(features):
    """Frames to the nearest frame where the ball was seen (0 where it is seen).

    Symmetric on purpose: this is an offline label generator, and a 40-frame stretch in which the
    ball is seen at neither end is a different claim about the world than one seen 5 frames later.
    """
    index = np.arange(len(features["nearest"]))
    seen = ball_visible(features)
    if not seen.any():
        return np.full(len(index), np.iinfo(np.int32).max, dtype=np.int64)
    where = np.where(seen, index, -10 ** 9)
    prev = np.maximum.accumulate(where)
    where_next = np.where(seen, index, 10 ** 9)
    nxt = np.minimum.accumulate(where_next[::-1])[::-1]
    return np.minimum(np.abs(index - prev), np.abs(nxt - index))


def interpolated_ball(features, pitch, max_gap=25):
    """Ball position in metres, linearly interpolated across short tracking gaps.

    A pass in flight is exactly the kind of moment the tracker loses the ball, and dropping those
    frames throws away the most informative part of the sequence. Interpolating across a gap of up
    to a second is honest -- the ball travels in roughly a straight line over that span -- while
    anything longer is flagged invalid rather than guessed.

    Returns (x, y, valid), with NaN positions where no interpolation was possible.
    """
    length, width = pitch
    ball = features["ball"]
    index = np.arange(len(ball))
    seen = ball_visible(features)
    gap = gap_to_observation(features)

    x = np.full(len(ball), np.nan)
    y = np.full(len(ball), np.nan)
    if not seen.any():
        return x, y, np.zeros(len(ball), dtype=bool)

    known = index[seen]
    x = np.interp(index, known, ball[known, 0] * length)
    y = np.interp(index, known, ball[known, 1] * width)
    valid = gap <= max_gap
    x[~valid] = np.nan
    y[~valid] = np.nan
    return x, y, valid


def ball_speed(features, pitch, max_gap=25):
    """Ball speed in m/s from consecutive observed positions, held across short tracking gaps.

    **The ATD's own speed column is NaN for the ball on every frame of this file** -- it is
    populated for players only. So speed has to be differentiated from position, which is also the
    better measurement anyway: it is a real displacement over a known number of frames rather than
    a per-observation figure that resets whenever the tracker re-acquires the ball.

    Inside a gap the speed of the last observation is held for up to `max_gap` frames, because that
    is what the gap's first frames looked like -- a ball that disappeared at 18 m/s is in flight,
    one that disappeared at 0.2 m/s is not. Beyond that the value is NaN: after two seconds in the
    dark the last speed says nothing about now.
    """
    length, width = pitch
    ball = features["ball"]
    n_frames = len(ball)
    index = np.arange(n_frames)
    seen = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    out = np.full(n_frames, np.nan)
    known = index[seen]
    if len(known) < 2:
        return out

    x = ball[known, 0] * length
    y = ball[known, 1] * width
    step = np.hypot(np.diff(x), np.diff(y)) / np.diff(known) * FPS
    # Each observation's speed is the mean of the interval before and after it.
    at_known = np.empty(len(known))
    at_known[0] = step[0]
    at_known[-1] = step[-1]
    if len(known) > 2:
        at_known[1:-1] = 0.5 * (step[:-1] + step[1:])
    out[known] = at_known

    previous = np.maximum.accumulate(np.where(seen, index, -10 ** 9))
    gap = index - previous
    fill = (previous >= 0) & (gap <= max_gap)
    out[fill] = out[previous[fill]]
    return out


def _nanmedian(values, width):
    """Centred nan-median over `width` frames; edges are clipped, not padded with zeros."""
    if width <= 1:
        return values
    half = width // 2
    padded = np.concatenate([np.full(half, np.nan), values, np.full(half, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    with np.errstate(all="ignore"):
        return np.nanmedian(windows, axis=1)


def outside_pitch(features, margin=0.0):
    """True where the ball is outside the pitch rectangle, inflated by `margin` (normalised).

    A ball beyond the touchline or goal line is out of play, which is dead ball by definition and
    the one dead-ball cue that does not need a threshold. Only meaningful where the ball is seen:
    an interpolated position cannot be outside the pitch if both ends of the gap were inside.
    """
    ball = features["ball"]
    x, y = ball[:, 0], ball[:, 1]
    return ((x < -margin) | (x > 1 + margin) | (y < -margin) | (y > 1 + margin))
