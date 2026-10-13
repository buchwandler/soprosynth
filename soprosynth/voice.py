"""Reference voice input loading, normalization, and identity helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np

from .audio import load_pcm_wav, normalize_audio, resample_linear
from .errors import InvalidVoiceError
from .types import PreparedVoice


@dataclass(frozen=True, slots=True)
class ReferenceAudio:
    audio: np.ndarray
    sample_rate: int
    source: str | None = None
    native_source: str | None = None


def fingerprint_audio(audio: np.ndarray, sample_rate: int) -> str:
    digest = hashlib.sha256()
    digest.update(sample_rate.to_bytes(8, "little", signed=False))
    digest.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
    return digest.hexdigest()


@overload
def load_reference_audio(
    source: str | Path,
    *,
    expected_sample_rate: int | None = None,
) -> ReferenceAudio: ...


@overload
def load_reference_audio(
    source: tuple[Any, int],
    *,
    expected_sample_rate: int | None = None,
) -> ReferenceAudio: ...


@overload
def load_reference_audio(
    source: PreparedVoice,
    *,
    expected_sample_rate: int | None = None,
) -> PreparedVoice: ...


@overload
def load_reference_audio(
    source: Any,
    *,
    expected_sample_rate: int | None = None,
) -> ReferenceAudio | PreparedVoice: ...


def load_reference_audio(
    source: Any,
    *,
    expected_sample_rate: int | None = None,
) -> ReferenceAudio | PreparedVoice:
    """Load WAV or ``(waveform, sample_rate)`` input into finite mono float32."""
    if isinstance(source, PreparedVoice):
        return source
    if expected_sample_rate is not None and (
        type(expected_sample_rate) is not int or expected_sample_rate <= 0
    ):
        raise InvalidVoiceError("expected_sample_rate must be a positive integer")
    source_path: str | None = None
    native_source: str | None = None
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        audio, sample_rate = load_pcm_wav(path)
        source_path = str(path.resolve())
        native_source = source_path
    elif isinstance(source, tuple) and len(source) == 2:
        waveform, sample_rate = source
        if type(sample_rate) is not int or sample_rate <= 0:
            raise InvalidVoiceError("reference sample rate must be a positive integer")
        audio = normalize_audio(waveform)
    else:
        raise InvalidVoiceError(
            "voice must be a PCM WAV path, (audio, sample_rate), or PreparedVoice"
        )
    if type(sample_rate) is not int or sample_rate <= 0:
        raise InvalidVoiceError("reference sample rate must be a positive integer")
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        audio = resample_linear(audio, sample_rate, expected_sample_rate)
        sample_rate = expected_sample_rate
        native_source = None
    return ReferenceAudio(
        audio=audio,
        sample_rate=sample_rate,
        source=source_path,
        native_source=native_source,
    )
