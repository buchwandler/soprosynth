"""Upstream-compatible Sopro text segmentation and output conditioning."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from .errors import ModelInferenceError, RuntimeCapabilityError

DEFAULT_MAX_SEGMENT_CHARS = 300
OUTPUT_TARGET_LEVEL_DB = -23.0
LIMITER_KNEE = 0.9
LEAD_IN_SECONDS = 0.08
SEGMENT_LEAD_SECONDS = 0.30
SEGMENT_SKIP_SECONDS = 0.10
TRAIL_SECONDS = 0.30
JOIN_FADE_SECONDS = 0.01
FINAL_FADE_SECONDS = 0.08
STREAM_GATE_HOLD_SECONDS = 2.0

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _pack(parts: Sequence[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        else:
            packed.append(current)
            current = part
    if current:
        packed.append(current)
    return packed


def split_text(text: str, max_segment_chars: int = DEFAULT_MAX_SEGMENT_CHARS) -> tuple[str, ...]:
    """Split text like Sopro's offline ``split_text`` without a native dependency."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if type(max_segment_chars) is not int or max_segment_chars < 1:
        raise ValueError("max_segment_chars must be a positive integer")

    normalized = " ".join(text.split())
    if len(normalized) <= max_segment_chars:
        return (normalized,) if normalized else ()

    segments: list[str] = []
    for sentence in _pack(_SENTENCE_END.split(normalized), max_segment_chars):
        if len(sentence) <= max_segment_chars:
            segments.append(sentence)
            continue
        for clause in _pack(_CLAUSE_END.split(sentence), max_segment_chars):
            if len(clause) <= max_segment_chars:
                segments.append(clause)
            else:
                segments.extend(_pack(clause.split(" "), max_segment_chars))
    return tuple(segments)


def _audio_parts(parts: Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(parts, np.ndarray):
        values: Sequence[np.ndarray] = (parts,)
    elif not isinstance(parts, (str, bytes)) and isinstance(parts, Sequence):
        values = parts
    else:
        raise ModelInferenceError("Sopro inference returned an invalid segment list")
    result: list[np.ndarray] = []
    for part in values:
        try:
            audio = np.asarray(part, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelInferenceError("Sopro inference returned invalid segment audio") from exc
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ModelInferenceError("Sopro inference returned invalid mono segment audio")
        if audio.size:
            result.append(np.ascontiguousarray(audio))
    return result


def condition_sopro_output(
    parts: Sequence[np.ndarray] | np.ndarray,
    *,
    sample_rate: int,
    mode: str,
    reference_level_db: Real | None,
    empty_audio_samples: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Sopro's native output chain, or concatenate untouched raw segments."""
    if mode not in ("native", "raw"):
        raise ValueError("mode must be 'native' or 'raw'")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if type(empty_audio_samples) is not int or empty_audio_samples < 0:
        raise ValueError("empty_audio_samples must be a non-negative integer")

    source_parts = _audio_parts(parts)
    level: float | None = None
    if isinstance(reference_level_db, Real) and not isinstance(reference_level_db, bool):
        candidate = float(reference_level_db)
        if math.isfinite(candidate):
            level = candidate

    metadata: dict[str, Any] = {
        "mode": mode,
        "reference_level_db": level,
        "target_level_db": None,
        "gain_db": None,
        "trimmed_lead_samples": 0,
        "trimmed_trail_samples": 0,
        "limiter_applied": False,
        "limited": False,
        "segment_count": len(source_parts),
    }
    if mode == "raw":
        audio = np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32), metadata

    if level is None:
        raise RuntimeCapabilityError(
            "Native Sopro output requires finite prepared reference_level_db metadata"
        )

    gain_db = OUTPUT_TARGET_LEVEL_DB - level
    if not math.isfinite(gain_db):
        raise RuntimeCapabilityError(
            "Prepared reference level produces a non-finite Sopro output gain"
        )
    metadata["target_level_db"] = OUTPUT_TARGET_LEVEL_DB
    metadata["gain_db"] = gain_db
    metadata["limiter_applied"] = bool(source_parts)
    if not source_parts:
        return np.zeros(empty_audio_samples, dtype=np.float32), metadata

    from audiosig import (
        apply_gain_db,
        fade_edges,
        join_with_edge_fades,
        soft_limit,
        trim_lead,
        trim_trail,
    )

    trimmed_parts: list[np.ndarray] = []
    trimmed_lead_samples = 0
    trimmed_trail_samples = 0
    for index, part in enumerate(source_parts):
        gained = apply_gain_db(part, gain_db)
        leading = trim_lead(
            gained,
            sample_rate=sample_rate,
            lead_seconds=LEAD_IN_SECONDS if index == 0 else SEGMENT_LEAD_SECONDS,
            skip_seconds=0.0 if index == 0 else SEGMENT_SKIP_SECONDS,
        )
        trailing = trim_trail(
            leading,
            sample_rate=sample_rate,
            trail_seconds=TRAIL_SECONDS,
        )
        trimmed_lead_samples += int(gained.size - leading.size)
        trimmed_trail_samples += int(leading.size - trailing.size)
        trimmed_parts.append(trailing)

    joined = join_with_edge_fades(
        trimmed_parts,
        sample_rate=sample_rate,
        fade_seconds=JOIN_FADE_SECONDS,
    )
    limited = soft_limit(joined, knee=LIMITER_KNEE)
    metadata["trimmed_lead_samples"] = trimmed_lead_samples
    metadata["trimmed_trail_samples"] = trimmed_trail_samples
    metadata["limited"] = bool(np.any(limited != joined))
    output = fade_edges(
        limited,
        sample_rate=sample_rate,
        fade_in=False,
        fade_out=True,
        fade_seconds=FINAL_FADE_SECONDS,
    )
    return np.ascontiguousarray(output, dtype=np.float32), metadata


class SoproStreamConditioner:
    """Incrementally apply Sopro output policy with bounded pending audio."""

    def __init__(
        self,
        *,
        sample_rate: int,
        mode: str,
        reference_level_db: float | None,
    ) -> None:
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if mode not in ("native", "raw"):
            raise ValueError("mode must be 'native' or 'raw'")
        self.sample_rate = sample_rate
        self.mode = mode
        self.gain_db: float | None = None
        if mode == "native":
            if isinstance(reference_level_db, bool) or not isinstance(reference_level_db, Real):
                raise RuntimeCapabilityError(
                    "Native Sopro output requires finite prepared reference_level_db metadata"
                )
            level = float(reference_level_db)
            if not math.isfinite(level):
                raise RuntimeCapabilityError(
                    "Native Sopro output requires finite prepared reference_level_db metadata"
                )
            self.gain_db = OUTPUT_TARGET_LEVEL_DB - level
            if not math.isfinite(self.gain_db):
                raise RuntimeCapabilityError("Prepared reference level produces a non-finite gain")
        self._active = False
        self._finished = False
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._first = True
        self._last = True
        self._gating = False
        self._pending = np.empty(0, dtype=np.float32)
        self._tail_pending = np.empty(0, dtype=np.float32)
        self._activity_window = np.empty(0, dtype=np.float32)
        self._processed_samples = 0
        self._emitted_samples = 0
        self._last_active_end: int | None = None

    def begin_segment(self, *, first: bool, last: bool) -> None:
        if self._finished:
            raise RuntimeCapabilityError("Sopro stream conditioner is already finished")
        if self._active:
            raise RuntimeCapabilityError("The previous Sopro stream segment is not finished")
        if type(first) is not bool or type(last) is not bool:
            raise ValueError("first and last must be bool values")
        self._reset_segment()
        self._first = first
        self._last = last
        self._gating = self.mode == "native"
        self._active = True

    def push(self, audio: np.ndarray) -> tuple[np.ndarray, ...]:
        if not self._active:
            raise RuntimeCapabilityError("begin_segment must be called before pushing audio")
        try:
            samples = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelInferenceError("ONNXVoice returned invalid streaming audio") from exc
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ModelInferenceError("ONNXVoice returned invalid mono streaming audio")
        samples = np.ascontiguousarray(samples)
        if samples.size == 0:
            return ()
        if self.mode == "raw":
            return (samples,)

        from audiosig import apply_gain_db

        assert self.gain_db is not None
        conditioned = np.ascontiguousarray(apply_gain_db(samples, self.gain_db), dtype=np.float32)
        if not self._gating:
            return self._queue_audio(conditioned)

        self._pending = np.concatenate((self._pending, conditioned))
        onset = self._find_onset(self._pending)
        if onset is None and self._pending.size < int(STREAM_GATE_HOLD_SECONDS * self.sample_rate):
            return ()
        return self._queue_audio(self._open_gate(onset))

    def _find_onset(self, audio: np.ndarray) -> int | None:
        if audio.size == 0:
            return None
        from audiosig import energy_fraction, energy_onset, speech_onset

        speech = speech_onset(audio, sample_rate=self.sample_rate)
        onset = speech if speech is not None else energy_onset(audio, sample_rate=self.sample_rate)
        if onset is None:
            return None
        guard_end = min(audio.size, onset + max(1, int(0.06 * self.sample_rate)))
        if (
            energy_fraction(
                audio,
                sample_rate=self.sample_rate,
                start=onset,
                end=guard_end,
            )
            <= 0.0
        ):
            return None
        return onset

    def _open_gate(self, onset: int | None) -> np.ndarray:
        audio = self._pending
        cut = 0
        if onset is not None:
            lead_seconds = LEAD_IN_SECONDS if self._first else SEGMENT_LEAD_SECONDS
            cut = max(0, onset - int(lead_seconds * self.sample_rate))
            if not self._first:
                cut = max(cut, int(SEGMENT_SKIP_SECONDS * self.sample_rate))
            cut = min(cut, max(0, onset - int(0.020 * self.sample_rate)))
        selected = np.ascontiguousarray(audio[cut:], dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._gating = False
        if selected.size and not self._first:
            from audiosig import fade_edges

            selected = np.ascontiguousarray(
                fade_edges(
                    selected,
                    sample_rate=self.sample_rate,
                    fade_in=True,
                    fade_out=False,
                    fade_seconds=JOIN_FADE_SECONDS,
                ),
                dtype=np.float32,
            )
        return selected

    def _update_activity(self, audio: np.ndarray) -> None:
        from audiosig import energy_fraction, energy_onset, speech_onset, trim_trail

        window_start = self._processed_samples - self._activity_window.size
        combined = np.concatenate((self._activity_window, audio))
        onset = speech_onset(combined, sample_rate=self.sample_rate)
        if onset is None:
            onset = energy_onset(combined, sample_rate=self.sample_rate)
        if onset is not None:
            frame = max(1, int(0.01 * self.sample_rate))
            last_active = 0
            for start in range(0, combined.size - frame + 1, frame):
                if (
                    energy_fraction(
                        combined,
                        sample_rate=self.sample_rate,
                        start=start,
                        end=start + frame,
                    )
                    > 0.0
                ):
                    last_active = start + frame
            if last_active:
                active = trim_trail(
                    combined,
                    sample_rate=self.sample_rate,
                    trail_seconds=0.0,
                )
                self._last_active_end = min(window_start + last_active, window_start + active.size)
        self._processed_samples += audio.size
        history = max(1, int(0.10 * self.sample_rate))
        self._activity_window = np.ascontiguousarray(combined[-history:], dtype=np.float32)

    def _queue_audio(self, audio: np.ndarray) -> tuple[np.ndarray, ...]:
        if audio.size == 0:
            return ()
        from audiosig import soft_limit

        limited = np.ascontiguousarray(soft_limit(audio, knee=LIMITER_KNEE), dtype=np.float32)
        self._update_activity(limited)
        self._tail_pending = np.concatenate((self._tail_pending, limited))
        trail_samples = int(TRAIL_SECONDS * self.sample_rate)
        safe_count = max(0, self._tail_pending.size - trail_samples)
        if self._last_active_end is None:
            release_count = safe_count
        else:
            target_end = self._last_active_end + trail_samples
            release_count = min(safe_count, max(0, target_end - self._emitted_samples))
        max_pending = int(STREAM_GATE_HOLD_SECONDS * self.sample_rate)
        release_count = max(release_count, self._tail_pending.size - max_pending)
        if release_count <= 0:
            return ()
        released = np.ascontiguousarray(self._tail_pending[:release_count], dtype=np.float32)
        self._tail_pending = np.ascontiguousarray(
            self._tail_pending[release_count:], dtype=np.float32
        )
        self._emitted_samples += released.size
        return (released,) if released.size else ()

    def finish_segment(self) -> tuple[np.ndarray, ...]:
        if not self._active:
            return ()
        if self.mode == "raw":
            self._active = False
            return ()

        outputs: list[np.ndarray] = []
        if self._gating and self._pending.size:
            onset = self._find_onset(self._pending)
            outputs.extend(self._queue_audio(self._open_gate(onset)))

        if self._tail_pending.size:
            from audiosig import fade_edges, soft_limit, trim_trail

            tail = self._tail_pending
            trimmed = trim_trail(
                tail,
                sample_rate=self.sample_rate,
                trail_seconds=TRAIL_SECONDS,
            )
            keep = trimmed.size
            if self._last_active_end is not None:
                target_end = self._last_active_end + int(TRAIL_SECONDS * self.sample_rate)
                keep = min(keep, max(0, target_end - self._emitted_samples))
            tail = np.ascontiguousarray(tail[:keep], dtype=np.float32)
            if tail.size:
                fade_seconds = JOIN_FADE_SECONDS if not self._last else FINAL_FADE_SECONDS
                faded = fade_edges(
                    tail,
                    sample_rate=self.sample_rate,
                    fade_in=False,
                    fade_out=True,
                    fade_seconds=fade_seconds,
                )
                limited = np.ascontiguousarray(
                    soft_limit(faded, knee=LIMITER_KNEE), dtype=np.float32
                )
                if limited.size:
                    outputs.append(limited)
        self._reset_segment()
        self._active = False
        return tuple(outputs)

    def finish(self) -> tuple[np.ndarray, ...]:
        outputs = self.finish_segment() if self._active else ()
        self._finished = True
        return outputs
