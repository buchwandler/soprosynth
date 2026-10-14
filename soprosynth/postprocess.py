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
