"""Dependency-free PCM WAV reading and writing helpers."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InvalidRequestError, InvalidVoiceError


def _pcm_samples(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        triples = np.frombuffer(raw, dtype=np.uint8)
        if triples.size % 3:
            raise InvalidVoiceError("PCM24 WAV data has an incomplete sample")
        triples = triples.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise InvalidVoiceError(f"Unsupported PCM WAV sample width: {sample_width} bytes")


def normalize_audio(audio: Any) -> np.ndarray:
    """Convert finite mono or frames-by-channels audio to contiguous float32 mono."""
    try:
        samples = np.asarray(audio)
        if samples.ndim == 1:
            mono = samples
        elif samples.ndim == 2 and samples.shape[1] > 0:
            mono = samples.astype(np.float64).mean(axis=1)
        else:
            raise InvalidVoiceError("reference audio must be mono or frames-by-channels")
        mono = np.ascontiguousarray(mono, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, InvalidVoiceError):
            raise
        raise InvalidVoiceError("reference audio must be numeric") from exc
    if mono.size == 0:
        raise InvalidVoiceError("reference audio must not be empty")
    if not np.isfinite(mono).all():
        raise InvalidVoiceError("reference audio must contain only finite samples")
    return mono


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with deterministic NumPy linear interpolation (no audio extras)."""
    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    count = max(1, round(audio.size * target_rate / source_rate))
    positions = np.arange(count, dtype=np.float64) * source_rate / target_rate
    source_positions = np.arange(audio.size, dtype=np.float64)
    result = np.interp(positions, source_positions, audio.astype(np.float64, copy=False))
    return np.ascontiguousarray(result, dtype=np.float32)


def load_pcm_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an uncompressed PCM WAV and return mono float32 samples and its rate."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with wave.open(str(source), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise InvalidVoiceError("Only uncompressed PCM WAV reference audio is supported")
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            if channels < 1 or sample_rate < 1 or frame_count < 1:
                raise InvalidVoiceError(
                    "PCM WAV must contain channels, frames, and a positive sample rate"
                )
            raw = stream.readframes(frame_count)
    except (wave.Error, EOFError, OSError) as exc:
        raise InvalidVoiceError(f"Could not read PCM WAV reference: {source}") from exc
    samples = _pcm_samples(raw, sample_width)
    if samples.size != frame_count * channels:
        raise InvalidVoiceError("PCM WAV data is truncated")
    if channels > 1:
        samples = samples.reshape(frame_count, channels).astype(np.float64).mean(axis=1)
    return normalize_audio(samples), sample_rate


def save_wav(audio: Any, sample_rate: int, path: str | Path) -> Path:
    """Write mono PCM16 WAV; clip only for conversion and preserve caller audio."""
    if type(sample_rate) is not int or sample_rate <= 0:
        raise InvalidRequestError("sample_rate must be a positive integer")
    samples = normalize_audio(audio)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output
