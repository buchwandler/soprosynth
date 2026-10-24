"""Backend-neutral public request, result, voice, and discovery types."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import InvalidLanguageError, InvalidRequestError

LANGUAGES = frozenset(("en", "pt", "fr", "de"))


def _freeze(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidRequestError(f"{label} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise InvalidRequestError(f"{label} mapping keys must be strings")
        return MappingProxyType({key: _freeze(item, label=label) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, label=label) for item in value)
    raise InvalidRequestError(f"{label} must contain JSON-compatible values")


def _immutable_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError(f"{label} must be a mapping")
    frozen = _freeze(value, label=label)
    return frozen


def _validated_audio(value: Any) -> np.ndarray:
    try:
        samples = np.asarray(value)
        if samples.ndim != 1:
            raise InvalidRequestError("audio must be one-dimensional mono samples")
        audio = np.ascontiguousarray(samples, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidRequestError("audio must be convertible to float32 samples") from exc
    if audio.size == 0:
        raise InvalidRequestError("audio must not be empty")
    if not np.isfinite(audio).all():
        raise InvalidRequestError("audio must contain only finite samples")
    return audio


def _sample_rate(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise InvalidRequestError("sample_rate must be a positive integer")
    return value


def _valid_id(value: Any, label: str = "id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequestError(f"{label} must be a non-empty string")
    return value


def _structured_items(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise InvalidRequestError(f"{label} must be a tuple")
    checked: list[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise InvalidRequestError(f"{label} mapping keys must be strings")
            checked.append(_immutable_mapping(item, label=label))
        elif is_dataclass(item) and not isinstance(item, type):
            checked.append(item)
        else:
            raise InvalidRequestError(f"each {label} item must be a mapping or dataclass value")
    return tuple(checked)


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    id: str
    text: str
    language: str | None = None
    tokens: tuple[Any, ...] = ()
    pronunciation_overrides: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        _valid_id(self.id)
        if not isinstance(self.text, str):
            raise InvalidRequestError("text must be a string")
        if self.language not in (None, "auto", *LANGUAGES):
            raise InvalidLanguageError("language must be None, 'auto', 'en', 'pt', 'fr', or 'de'")
        object.__setattr__(self, "tokens", _structured_items(self.tokens, "tokens"))
        object.__setattr__(
            self,
            "pronunciation_overrides",
            _structured_items(self.pronunciation_overrides, "pronunciation_overrides"),
        )


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    """Runtime-bound prepared reference state with a stable public identity."""

    state: Any = field(repr=False, compare=False)
    sample_rate: int
    source: str | Path | None
    model_id: str | None
    source_revision: str | None
    backend: str
    runtime_fingerprint: str | None
    fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _owner_id: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _sample_rate(self.sample_rate)
        if self.source is not None and not isinstance(self.source, (str, Path)):
            raise InvalidRequestError("voice source must be a path-like string or None")
        if not isinstance(self.backend, str) or not self.backend:
            raise InvalidRequestError("voice backend must be a non-empty string")
        _valid_id(self.fingerprint, "fingerprint")
        if self.model_id is not None and not isinstance(self.model_id, str):
            raise InvalidRequestError("model_id must be a string or None")
        if self.source_revision is not None and not isinstance(self.source_revision, str):
            raise InvalidRequestError("source_revision must be a string or None")
        if self.runtime_fingerprint is not None and not isinstance(self.runtime_fingerprint, str):
            raise InvalidRequestError("runtime_fingerprint must be a string or None")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata, label="metadata"))

    @property
    def identity(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "kind": "reference",
                "sha256": self.fingerprint,
                "sample_rate": self.sample_rate,
                "model_id": self.model_id,
                "source_revision": self.source_revision,
                "backend": self.backend,
            }
        )


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Word-level timing record reserved for backends that provide alignments."""

    text: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvalidRequestError("word timing text must be a string")
        for name, value in (
            ("start_seconds", self.start_seconds),
            ("end_seconds", self.end_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidRequestError(f"{name} must be a finite number")
            if not math.isfinite(float(value)) or value < 0:
                raise InvalidRequestError(f"{name} must be finite and non-negative")
        if self.end_seconds < self.start_seconds:
            raise InvalidRequestError("word timing end must not precede its start")


@dataclass(frozen=True, slots=True)
class SynthesisChunk:
    """One validated, immutable mono-audio item from a streaming synthesis request."""

    id: str
    audio: np.ndarray
    sample_rate: int
    index: int
    segment_index: int
    is_final: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _valid_id(self.id)
        object.__setattr__(self, "audio", _validated_audio(self.audio))
        _sample_rate(self.sample_rate)
        for name in ("index", "segment_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise InvalidRequestError(f"{name} must be a non-negative integer")
        if type(self.is_final) is not bool:
            raise InvalidRequestError("is_final must be a bool")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata, label="metadata"))


@dataclass(slots=True)
class SynthesisResult:
    id: str
    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    warnings: tuple[str, ...] = ()
    word_timings: tuple[WordTiming, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _valid_id(self.id)
        self.audio = _validated_audio(self.audio)
        self.sample_rate = _sample_rate(self.sample_rate)
        if not isinstance(self.text, str):
            raise InvalidRequestError("result text must be a string")
        if not isinstance(self.language, str):
            raise InvalidLanguageError("result language must be a string")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(item, str) for item in self.warnings
        ):
            raise InvalidRequestError("warnings must be a tuple of strings")
        if not isinstance(self.word_timings, tuple) or any(
            not isinstance(item, WordTiming) for item in self.word_timings
        ):
            raise InvalidRequestError("word_timings must be a tuple of WordTiming values")
        self.metadata = _immutable_mapping(self.metadata, label="metadata")

    def save_wav(self, path: str | Path) -> Path:
        """Write mono PCM16 WAV without mutating the in-memory audio."""
        from .audio import save_wav

        return save_wav(self.audio, self.sample_rate, path)


@dataclass(frozen=True, slots=True)
class RequestMeasure:
    amount: int
    maximum: int | None
    unit: str = "tokens"
    fits: bool | None = None

    def __post_init__(self) -> None:
        if type(self.amount) is not int or self.amount < 0:
            raise InvalidRequestError("measure amount must be a non-negative integer")
        if self.maximum is not None and (type(self.maximum) is not int or self.maximum < 0):
            raise InvalidRequestError("measure maximum must be a non-negative integer or None")
        if not isinstance(self.unit, str) or not self.unit:
            raise InvalidRequestError("measure unit must be a non-empty string")
        if self.fits is not None and type(self.fits) is not bool:
            raise InvalidRequestError("measure fits must be bool or None")

        derived = None if self.maximum is None else self.amount <= self.maximum
        if self.fits is not None and derived is not None and self.fits is not derived:
            raise InvalidRequestError("measure fits value is inconsistent with amount and maximum")
        if self.fits is None and derived is not None:
            object.__setattr__(self, "fits", derived)


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    backend: str
    model_id: str
    sample_rate: int
    source_revision: str | None = None
    device: str | None = None
    providers_requested: tuple[str, ...] = ()
    providers_active: tuple[str, ...] = ()
    sessions: tuple[str | Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise InvalidRequestError("diagnostic backend must be a non-empty string")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise InvalidRequestError("diagnostic model_id must be a non-empty string")
        _sample_rate(self.sample_rate)
        for label, values in (
            ("providers_requested", self.providers_requested),
            ("providers_active", self.providers_active),
        ):
            if not isinstance(values, tuple) or any(not isinstance(item, str) for item in values):
                raise InvalidRequestError(f"{label} must be a tuple of strings")
        if not isinstance(self.sessions, tuple):
            raise InvalidRequestError("sessions must be a tuple of normalized diagnostics")
        normalized_sessions: list[str | Mapping[str, Any]] = []
        for session in self.sessions:
            if isinstance(session, str):
                normalized_sessions.append(session)
            elif isinstance(session, Mapping):
                normalized_sessions.append(_immutable_mapping(session, label="session diagnostics"))
            else:
                raise InvalidRequestError("session diagnostics must not expose backend objects")
        object.__setattr__(self, "sessions", tuple(normalized_sessions))


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Capabilities of one selected runtime, not the producer-level union."""

    backend: str
    supports_linguistic_tokens: bool = False
    supports_pronunciation_overrides: bool = False
    supports_whole_request_phonemes: bool = False
    supports_named_voices: bool = False
    supports_reference_voice: bool = True
    supports_managed_reference_voice: bool = False
    supports_prepared_voice_reuse: bool = True
    supports_word_timings: bool = False
    supports_voice_level: bool = True
    supports_request_measurement: bool = False
    supports_streaming: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise InvalidRequestError("capability backend must be a non-empty string")
        for name in (
            "supports_linguistic_tokens",
            "supports_pronunciation_overrides",
            "supports_whole_request_phonemes",
            "supports_named_voices",
            "supports_reference_voice",
            "supports_managed_reference_voice",
            "supports_prepared_voice_reuse",
            "supports_word_timings",
            "supports_voice_level",
            "supports_request_measurement",
            "supports_streaming",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidRequestError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class DiscoveredBundle:
    id: str
    ref: str
    display_name: str
    aliases: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    sample_rate: int | None = None
    profiles: tuple[str, ...] = ()
    source_revision: str | None = None
    source_commit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "ref", "display_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise InvalidRequestError(f"{name} must be a non-empty string")
        for name in ("aliases", "languages", "profiles"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
                raise InvalidRequestError(f"{name} must be a tuple of strings")
        if self.sample_rate is not None:
            _sample_rate(self.sample_rate)
        for name in ("source_revision", "source_commit"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise InvalidRequestError(f"{name} must be a string or None")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata, label="metadata"))
