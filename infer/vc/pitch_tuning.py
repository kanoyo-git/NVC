"""Model-aware pitch processing for offline RVC inference.

The RVC generator receives both a coarse pitch bin and a continuous F0 contour.
Small frame-to-frame errors in that contour are therefore audible.  This module
keeps the original melody intact while solving three separate problems:

* reject short octave/spike errors from the F0 extractor;
* tune stable note centres without flattening vibrato or glides;
* move the whole performance by octaves into a model's trained register.

Pitch calculations use MIDI/log-frequency space.  Linear interpolation in Hz is
not perceptually symmetric and was one of the causes of the old tuner's uneven
correction.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import median_filter


logger = logging.getLogger(__name__)

F0_MIN = 50.0
F0_MAX = 1100.0
PROFILE_QUANTILES = np.asarray((0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))


def hz_to_midi(frequency):
    frequency = np.asarray(frequency, dtype=np.float64)
    return 69.0 + 12.0 * np.log2(frequency / 440.0)


def midi_to_hz(midi):
    midi = np.asarray(midi, dtype=np.float64)
    return 440.0 * np.exp2((midi - 69.0) / 12.0)


@dataclass(frozen=True)
class PitchProfile:
    """Robust F0 distribution extracted from a model's training voice."""

    name: str
    quantiles: Mapping[float, float]
    voiced_frames: int = 0
    source: str = ""

    def __post_init__(self):
        cleaned = {}
        for key, value in self.quantiles.items():
            quantile = float(key)
            midi = float(value)
            if 0.0 < quantile < 1.0 and math.isfinite(midi):
                cleaned[quantile] = midi
        if len(cleaned) < 3 or not any(abs(key - 0.5) < 1e-6 for key in cleaned):
            raise ValueError("Pitch profile needs at least three quantiles including 0.5")
        keys = np.asarray(sorted(cleaned))
        values = np.asarray([cleaned[key] for key in keys])
        if np.any(np.diff(values) < 0):
            raise ValueError("Pitch profile quantiles must be monotonic")
        object.__setattr__(self, "quantiles", cleaned)

    def values_at(self, quantiles=PROFILE_QUANTILES):
        keys = np.asarray(sorted(self.quantiles), dtype=np.float64)
        values = np.asarray([self.quantiles[key] for key in keys], dtype=np.float64)
        return np.interp(np.asarray(quantiles, dtype=np.float64), keys, values)


def _profile_from_mapping(data):
    if not isinstance(data, Mapping):
        raise ValueError("Pitch profile must be a JSON object")
    raw_quantiles = data.get("midi_quantiles", data.get("quantiles"))
    if not isinstance(raw_quantiles, Mapping):
        raise ValueError("Pitch profile is missing midi_quantiles")
    return PitchProfile(
        name=str(data.get("name") or data.get("model") or "model"),
        source=str(data.get("source") or ""),
        voiced_frames=int(data.get("voiced_frames") or 0),
        quantiles={float(key): float(value) for key, value in raw_quantiles.items()},
    )


def load_pitch_profile(model_path, checkpoint=None):
    """Load an embedded or sidecar profile for ``model_path``.

    A stem-specific sidecar wins, which permits several models in one folder to
    have different ranges.  ``pitch-profile.json`` is convenient for packaged
    model directories containing one voice.
    """

    embedded_profile = (
        checkpoint.get("pitch_profile") if isinstance(checkpoint, Mapping) else None
    )
    if embedded_profile is not None:
        try:
            return _profile_from_mapping(embedded_profile), "checkpoint"
        except (TypeError, ValueError) as error:
            logger.warning("Ignoring invalid embedded pitch profile: %s", error)

    path = Path(model_path)
    candidates = (
        path.with_suffix(".pitch.json"),
        path.parent / "pitch-profile.json",
        path.parent / "pitch_profile.json",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                return _profile_from_mapping(json.load(handle)), str(candidate)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            logger.warning("Ignoring invalid pitch profile %s: %s", candidate, error)
    return None, None


def sanitize_f0(f0):
    result = np.asarray(f0, dtype=np.float64).copy()
    invalid = (~np.isfinite(result)) | (result < F0_MIN) | (result > F0_MAX)
    result[invalid] = 0.0
    return result


def _bridge_short_gaps(mask, maximum=3):
    bridged = np.asarray(mask, dtype=bool).copy()
    index = 0
    while index < bridged.size:
        if bridged[index]:
            index += 1
            continue
        end = index
        while end < bridged.size and not bridged[end]:
            end += 1
        if index > 0 and end < bridged.size and end - index <= maximum:
            bridged[index:end] = True
        index = end
    return bridged


def _runs(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _interpolate_active(f0, active):
    result = np.asarray(f0, dtype=np.float64).copy()
    known = result > 0
    missing = active & ~known
    if missing.any() and known.any():
        result[missing] = np.interp(
            np.flatnonzero(missing), np.flatnonzero(known), result[known]
        )
    return result


def stabilize_f0(f0, maximum_gap=3):
    """Remove short octave and spike errors without smoothing real note motion."""

    original = sanitize_f0(f0)
    voiced = original > 0
    active = _bridge_short_gaps(voiced, maximum_gap)
    work = _interpolate_active(original, active)

    for start, end in _runs(active):
        if end - start < 3:
            continue
        midi = hz_to_midi(work[start:end])
        window = min(5, end - start if (end - start) % 2 else end - start - 1)
        reference = median_filter(midi, size=max(window, 3), mode="nearest")
        octave = np.rint((midi - reference) / 12.0)
        candidate = midi - octave * 12.0
        octave_error = (np.abs(midi - reference) > 7.0) & (
            np.abs(candidate - reference) < 3.0
        )
        midi[octave_error] = candidate[octave_error]

        # A one-frame non-octave spike is also almost always an extractor error.
        if midi.size >= 3:
            before = midi[:-2]
            centre = midi[1:-1]
            after = midi[2:]
            spike = (
                (np.abs(centre - before) > 4.0)
                & (np.abs(centre - after) > 4.0)
                & (np.abs(before - after) < 1.5)
            )
            centre[spike] = (before[spike] + after[spike]) * 0.5
        work[start:end] = midi_to_hz(midi)

    work[~voiced] = 0.0
    return work


def _smooth_bidirectional(values, frame_rate=100.0, time_constant=0.045):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return values.copy()
    alpha = 1.0 - math.exp(-1.0 / max(frame_rate * time_constant, 1.0))

    def pass_once(source):
        output = source.copy()
        for index in range(1, source.size):
            output[index] = output[index - 1] + alpha * (
                source[index] - output[index - 1]
            )
        return output

    forward = pass_once(values)
    backward = pass_once(values[::-1])[::-1]
    return (forward + backward) * 0.5


def _hysteretic_notes(centres, margin=0.12):
    centres = np.asarray(centres, dtype=np.float64)
    if centres.size == 0:
        return centres.copy()
    notes = np.empty_like(centres)
    current = int(np.rint(centres[0]))
    notes[0] = current
    for index in range(1, centres.size):
        centre = centres[index]
        desired = int(np.rint(centre))
        if desired > current and centre >= current + 0.5 + margin:
            current = desired
        elif desired < current and centre <= current - 0.5 - margin:
            current = desired
        notes[index] = current
    return notes


class AdaptivePitchSmoother:
    """Smooth note centres while retaining the source's vibrato and transitions."""

    def __init__(self, frame_rate=100.0):
        self.frame_rate = float(frame_rate)

    def tune(self, f0):
        original = sanitize_f0(f0)
        if not np.any(original > 0):
            return original

        voiced = original > 0
        active = _bridge_short_gaps(voiced, maximum=3)
        work = _interpolate_active(original, active)
        tuned = work.copy()

        for start, end in _runs(active):
            length = end - start
            if length < 3:
                continue
            midi = hz_to_midi(work[start:end])
            window = min(9, length if length % 2 else length - 1)
            window = max(window, 3)
            centres = median_filter(midi, size=window, mode="nearest")
            centres = _smooth_bidirectional(
                centres, frame_rate=self.frame_rate, time_constant=0.035
            )
            targets = _hysteretic_notes(centres)
            correction = np.clip(targets - centres, -0.65, 0.65)
            correction = _smooth_bidirectional(
                correction, frame_rate=self.frame_rate, time_constant=0.080
            )

            # Fade correction at voiced boundaries to avoid an F0 discontinuity
            # on consonant attacks or at the edge of an inference pad.
            fade = min(4, length // 2)
            if fade:
                ramp = np.linspace(0.25, 1.0, fade)
                correction[:fade] *= ramp
                correction[-fade:] *= ramp[::-1]
            tuned[start:end] = midi_to_hz(midi + correction)

        tuned[~voiced] = 0.0
        return tuned


@dataclass(frozen=True)
class RegisterMatch:
    shift: int
    source_median_midi: float
    target_median_midi: float
    baseline_cost: float
    selected_cost: float
    profile_name: str


@dataclass(frozen=True)
class PhraseRegisterMatch:
    start_frame: int
    end_frame: int
    shift: int
    source_median_midi: float
    baseline_cost: float
    selected_cost: float


@dataclass(frozen=True)
class DynamicRegisterMatch:
    """A time-varying register decision over complete vocal phrases."""

    shift: int
    source_median_midi: float
    target_median_midi: float
    baseline_cost: float
    selected_cost: float
    profile_name: str
    phrases: tuple[PhraseRegisterMatch, ...]


def _register_cost(source, target, candidate):
    shifted = source + candidate
    weights = np.asarray((0.8, 1.0, 1.4, 2.2, 1.4, 1.0, 0.8))
    distribution = float(np.average(np.abs(shifted - target), weights=weights))
    low_limit = target[0] - 2.0
    high_limit = target[-1] + 2.0
    overflow = max(low_limit - shifted[0], 0.0) + max(
        shifted[-1] - high_limit, 0.0
    )
    # A small prior prevents octave changes when two registers are equally good.
    return distribution + 1.4 * overflow + 0.10 * abs(candidate) / 12.0


def select_register_shift(
    f0,
    profile=None,
    manual_shift=0.0,
    fallback_target_hz=155.0,
    limit=24,
):
    """Choose an octave-only model register shift.

    Restricting the automatic part to octaves keeps the original song key and
    accompaniment compatibility.  The user's manual semitone shift is applied
    before this match and remains fully supported.
    """

    clean = sanitize_f0(f0)
    voiced = clean[clean > 0]
    if voiced.size < 8:
        return 0, None
    midi = hz_to_midi(voiced)
    octave_limit = max(abs(int(limit)) // 12, 1)
    candidates = np.arange(-octave_limit, octave_limit + 1, dtype=int) * 12
    source = np.quantile(midi, PROFILE_QUANTILES) + float(manual_shift)

    if profile is not None:
        target = profile.values_at(PROFILE_QUANTILES)
        costs = np.asarray(
            [_register_cost(source, target, shift) for shift in candidates]
        )
        zero_index = int(np.flatnonzero(candidates == 0)[0])
        best_index = int(np.argmin(costs))
        best_shift = int(candidates[best_index])
        # Do not flip an octave for a marginal/ambiguous improvement.
        if best_shift and costs[zero_index] - costs[best_index] < 1.0:
            best_index = zero_index
            best_shift = 0
        return best_shift, RegisterMatch(
            shift=best_shift,
            source_median_midi=float(source[3]),
            target_median_midi=float(target[3]),
            baseline_cost=float(costs[zero_index]),
            selected_cost=float(costs[best_index]),
            profile_name=profile.name,
        )

    if not fallback_target_hz or fallback_target_hz <= 0:
        return 0, None
    target_median = float(hz_to_midi(float(fallback_target_hz)))
    costs = (
        np.abs(source[3] + candidates - target_median)
        + 0.10 * np.abs(candidates) / 12.0
    )
    zero_index = int(np.flatnonzero(candidates == 0)[0])
    best_index = int(np.argmin(costs))
    if candidates[best_index] and costs[zero_index] - costs[best_index] < 2.0:
        best_index = zero_index
    shift = int(candidates[best_index])
    return shift, RegisterMatch(
        shift=shift,
        source_median_midi=float(source[3]),
        target_median_midi=target_median,
        baseline_cost=float(costs[zero_index]),
        selected_cost=float(costs[best_index]),
        profile_name="fallback",
    )


def segment_vocal_phrases(f0, frame_rate=100.0, minimum_gap_seconds=0.14):
    """Return phrase ranges separated by a safe unvoiced interval.

    Short F0 dropouts are bridged so an extractor miss cannot split a sustained
    note.  Conversely, register changes are only allowed across a real pause;
    this is what prevents an octave decision from tearing a note in half.
    """

    clean = sanitize_f0(f0)
    voiced = clean > 0
    maximum_gap = max(int(round(float(frame_rate) * minimum_gap_seconds)) - 1, 3)
    phrase_mask = _bridge_short_gaps(voiced, maximum=maximum_gap)
    return [(int(start), int(end)) for start, end in _runs(phrase_mask)]


def _target_distribution(source, profile, fallback_target_hz):
    if profile is not None:
        return profile.values_at(PROFILE_QUANTILES), profile.name
    target_median = float(hz_to_midi(float(fallback_target_hz)))
    # With no target distribution, retain the source range and move its centre.
    return source - source[3] + target_median, "fallback"


def _phrase_emissions(
    midi,
    phrases,
    target,
    candidates,
    manual_shift,
    global_shift,
    frame_rate,
):
    emissions = np.empty((len(phrases), len(candidates)), dtype=np.float64)
    phrase_sources = []
    for phrase_index, (start, end) in enumerate(phrases):
        values = midi[start:end]
        values = values[np.isfinite(values)]
        if values.size >= 8:
            source = np.quantile(values, PROFILE_QUANTILES) + float(manual_shift)
        else:
            source = None
        phrase_sources.append(source)
        if source is None:
            emissions[phrase_index] = 0.0
            continue
        costs = np.asarray(
            [_register_cost(source, target, shift) for shift in candidates]
        )
        zero_index = int(np.flatnonzero(candidates == 0)[0])
        # A very short fragment may be a breath, separator residue, or a single
        # low/high melody note rather than a different singer.  Cap how much
        # evidence one phrase can contribute, scaled by its actually voiced
        # duration.  Consistent neighbouring phrases still accumulate enough
        # evidence to establish a new register section.
        evidence = min(values.size / max(float(frame_rate), 1.0), 1.0)
        maximum_advantage = 6.0 * evidence
        costs = np.maximum(costs, costs[zero_index] - maximum_advantage)
        for state, shift in enumerate(candidates):
            # A phrase needs real evidence before it may leave the unshifted
            # state.  The global decision is only a weak prior, not a lock.
            if shift and costs[zero_index] - costs[state] < 1.0:
                costs[state] += 2.0
            costs[state] += 0.20 * abs(int(shift) - int(global_shift)) / 12.0
        emissions[phrase_index] = costs
    return emissions, phrase_sources


def _viterbi_register_path(
    emissions, candidates, phrases, frame_rate, anchor_shift=0
):
    phrase_count, state_count = emissions.shape
    costs = np.full_like(emissions, np.inf)
    backpointers = np.zeros((phrase_count, state_count), dtype=np.int32)
    endpoint_penalty = 3.0 * np.abs(candidates - int(anchor_shift)) / 12.0
    costs[0] = emissions[0] + endpoint_penalty
    for phrase_index in range(1, phrase_count):
        gap_frames = max(phrases[phrase_index][0] - phrases[phrase_index - 1][1], 0)
        gap_seconds = gap_frames / float(frame_rate)
        # A long pause is a natural singer/section boundary.  Across short
        # pauses, require stronger evidence before changing octave state.
        transition_weight = 1.0 if gap_seconds >= 0.40 else 4.0
        for state, shift in enumerate(candidates):
            transition = (
                transition_weight
                * np.abs(candidates - shift)
                / 12.0
            )
            previous = costs[phrase_index - 1] + transition
            best_previous = int(np.argmin(previous))
            costs[phrase_index, state] = (
                previous[best_previous] + emissions[phrase_index, state]
            )
            backpointers[phrase_index, state] = best_previous
    path = np.empty(phrase_count, dtype=np.int32)
    path[-1] = int(np.argmin(costs[-1] + endpoint_penalty))
    for phrase_index in range(phrase_count - 1, 0, -1):
        path[phrase_index - 1] = backpointers[phrase_index, path[phrase_index]]
    return path


def select_dynamic_register_shifts(
    f0,
    profile=None,
    manual_shift=0.0,
    fallback_target_hz=155.0,
    limit=24,
    frame_rate=100.0,
    minimum_gap_seconds=0.14,
):
    """Select a key-safe octave shift independently for each vocal phrase.

    The returned curve has the same number of frames as ``f0``.  It changes
    only in unvoiced gaps, so applying it to RVC's F0 conditioning keeps the
    waveform duration, beat alignment, note attacks, vibrato, and glides.
    """

    clean = sanitize_f0(f0)
    curve = np.zeros(clean.shape, dtype=np.float64)
    voiced = clean > 0
    if np.count_nonzero(voiced) < 8:
        return curve, None

    midi = np.full(clean.shape, np.nan, dtype=np.float64)
    midi[voiced] = hz_to_midi(clean[voiced])
    global_source = np.quantile(midi[voiced], PROFILE_QUANTILES) + float(
        manual_shift
    )
    target, profile_name = _target_distribution(
        global_source, profile, fallback_target_hz
    )
    octave_limit = max(abs(int(limit)) // 12, 1)
    candidates = np.arange(-octave_limit, octave_limit + 1, dtype=int) * 12
    global_costs = np.asarray(
        [_register_cost(global_source, target, shift) for shift in candidates]
    )
    global_shift = int(candidates[int(np.argmin(global_costs))])
    phrases = segment_vocal_phrases(
        clean,
        frame_rate=frame_rate,
        minimum_gap_seconds=minimum_gap_seconds,
    )
    if not phrases:
        return curve, None

    emissions, phrase_sources = _phrase_emissions(
        midi,
        phrases,
        target,
        candidates,
        manual_shift,
        global_shift,
        frame_rate,
    )
    path = _viterbi_register_path(
        emissions, candidates, phrases, frame_rate, anchor_shift=global_shift
    )
    zero_index = int(np.flatnonzero(candidates == 0)[0])
    phrase_reports = []
    total_baseline = 0.0
    total_selected = 0.0
    duration_by_shift = {}
    for phrase_index, ((start, end), state) in enumerate(zip(phrases, path)):
        shift = int(candidates[state])
        curve[start:end] = shift
        duration = end - start
        duration_by_shift[shift] = duration_by_shift.get(shift, 0) + duration
        baseline = float(emissions[phrase_index, zero_index])
        selected = float(emissions[phrase_index, state])
        total_baseline += baseline
        total_selected += selected
        source = phrase_sources[phrase_index]
        phrase_reports.append(
            PhraseRegisterMatch(
                start_frame=start,
                end_frame=end,
                shift=shift,
                source_median_midi=float(source[3]) if source is not None else math.nan,
                baseline_cost=baseline,
                selected_cost=selected,
            )
        )
    dominant_shift = max(duration_by_shift, key=duration_by_shift.get)
    return curve, DynamicRegisterMatch(
        shift=int(dominant_shift),
        source_median_midi=float(global_source[3]),
        target_median_midi=float(target[3]),
        baseline_cost=total_baseline,
        selected_cost=total_selected,
        profile_name=profile_name,
        phrases=tuple(phrase_reports),
    )
