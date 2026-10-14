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
    seed: int | None = None
    text_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and _number("temperature", self.temperature) < 0:
            raise InvalidGenerationConfigError("temperature must be non-negative")
        if self.top_p is not None and not 0 <= _number("top_p", self.top_p) <= 1:
            raise InvalidGenerationConfigError("top_p must be in [0, 1]")
        for name, minimum in (("top_k", 0), ("steps", 1)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise InvalidGenerationConfigError(f"{name} must be an integer >= {minimum}")
        for name in ("seed", "text_chunk_size"):
            value = getattr(self, name)
            minimum, maximum = (0, 2**32 - 1) if name == "seed" else (1, None)
            if value is not None and (
                type(value) is not int
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
                raise InvalidGenerationConfigError(f"{name} must be an integer {bounds}")
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

    def as_inference_kwargs(self) -> dict[str, int | float]:
        """Return only explicitly configured inference controls."""
        return {
            key: value
            for key, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("top_k", self.top_k),
                ("steps", self.steps),
                ("max_seconds", self.max_seconds),
                ("min_seconds", self.min_seconds),
                ("seed", self.seed),
            )
            if value is not None
        }


SoproOutputMode = Literal["native", "raw"]


@dataclass(frozen=True, slots=True)
class SoproOutputConfig:
    """Select native-compatible Sopro conditioning or unmodified ONNX audio."""

    mode: SoproOutputMode = "native"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in ("native", "raw"):
            raise InvalidGenerationConfigError("Sopro output mode must be 'native' or 'raw'")


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
