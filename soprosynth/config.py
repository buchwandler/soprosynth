"""Strict, backend-neutral generation and voice-level configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .errors import InvalidGenerationConfigError

VoiceLevelMode = Literal["off", "calibrated"]
VoiceLevelSource = Literal[
    "off",
    "override",
    "catalog",
    "missing_calibration",
]


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidGenerationConfigError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidGenerationConfigError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Validated generation controls shared by supported backends."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    steps: int | None = None
    max_seconds: float | None = None
    min_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and _number("temperature", self.temperature) <= 0:
            raise InvalidGenerationConfigError("temperature must be positive")
        if self.top_p is not None and not 0 < _number("top_p", self.top_p) <= 1:
            raise InvalidGenerationConfigError("top_p must be in (0, 1]")
        for name in ("top_k", "steps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise InvalidGenerationConfigError(f"{name} must be an integer >= 1")
        if self.max_seconds is not None and _number("max_seconds", self.max_seconds) <= 0:
            raise InvalidGenerationConfigError("max_seconds must be positive")
        if self.min_seconds is not None and _number("min_seconds", self.min_seconds) < 0:
            raise InvalidGenerationConfigError("min_seconds must be non-negative")
        if (
            self.max_seconds is not None
            and self.min_seconds is not None
            and self.min_seconds > self.max_seconds
        ):
            raise InvalidGenerationConfigError("min_seconds must not exceed max_seconds")

    def as_native_kwargs(self) -> dict[str, int | float]:
        """Return only explicitly configured native controls."""
        return {
            key: value
            for key, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("top_k", self.top_k),
                ("steps", self.steps),
                ("max_seconds", self.max_seconds),
                ("min_seconds", self.min_seconds),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class VoiceLevelConfig:
    """Opt-in static gain policy; no waveform analysis is performed."""

    mode: VoiceLevelMode = "off"
    gain_db: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("off", "calibrated"):
            raise InvalidGenerationConfigError("voice-level mode must be 'off' or 'calibrated'")
        if self.gain_db is not None:
            _number("gain_db", self.gain_db)

        if self.mode == "off" and self.gain_db is not None:
            raise InvalidGenerationConfigError("gain_db requires voice-level mode 'calibrated'")


@dataclass(frozen=True, slots=True)
class VoiceLevelApplication:
    mode: VoiceLevelMode
    gain_db: float
    source: VoiceLevelSource
    applied: bool

    def __post_init__(self) -> None:
        if self.mode not in ("off", "calibrated"):
            raise InvalidGenerationConfigError("invalid voice-level application mode")
        _number("gain_db", self.gain_db)
        if self.source not in ("off", "override", "catalog", "missing_calibration"):
            raise InvalidGenerationConfigError("invalid voice-level application source")
        if type(self.applied) is not bool:
            raise InvalidGenerationConfigError("applied must be a bool")
