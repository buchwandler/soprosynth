import sys
import wave
from typing import Any

import numpy as np
import pytest

from soprosynth import PreparedVoice
from soprosynth.errors import InvalidVoiceError, OptionalDependencyError
from soprosynth.voice import fingerprint_audio, load_reference_audio


def _write_wav(path, audio, sample_rate=24000, channels=1, width=2):
    values = np.asarray(audio, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(sample_rate)
        writer.writeframes(values.tobytes())


def test_load_path_tuple_resampling_and_audio_identity(tmp_path):
    path = tmp_path / "ref.wav"
    _write_wav(path, [0, 16384, -16384], sample_rate=24000)
    loaded = load_reference_audio(path, expected_sample_rate=24000)
    assert loaded.source == str(path.resolve())
    assert loaded.native_source == str(path.resolve())
    assert loaded.sample_rate == 24000
    assert loaded.audio.dtype == np.float32
    assert fingerprint_audio(loaded.audio, loaded.sample_rate) == fingerprint_audio(
        loaded.audio.copy(), loaded.sample_rate
    )

    resampled = load_reference_audio(path, expected_sample_rate=48000)
    assert resampled.sample_rate == 48000
    assert resampled.native_source is None
    assert resampled.audio.size == 6

    tuple_audio = load_reference_audio((np.array([[0.0, 1.0], [1.0, 1.0]]), 24000))
    assert tuple_audio.source is None
    assert tuple_audio.audio.tolist() == pytest.approx([0.5, 1.0])


def test_prepared_voice_is_returned_for_runtime_compatibility_check():
    voice = PreparedVoice(
        state=object(),
        sample_rate=24000,
        source=None,
        model_id="model",
        source_revision=None,
        backend="native",
        runtime_fingerprint="runtime",
        fingerprint="f" * 64,
    )
    assert load_reference_audio(voice) is voice


def test_invalid_inputs_and_missing_paths_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_audio(tmp_path / "missing.wav")
    with pytest.raises(InvalidVoiceError):
        load_reference_audio((np.array([np.nan]), 24000))
    with pytest.raises(InvalidVoiceError):
        load_reference_audio((np.array([0.0]), True))
    with pytest.raises(InvalidVoiceError):
        load_reference_audio(np.array([0.0]))


def test_linear_resampling_of_tuple_audio():
    loaded = load_reference_audio(
        (np.array([0.0, 1.0], dtype=np.float32), 2), expected_sample_rate=4
    )
    assert loaded.audio.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_audiosig_resampling_is_opt_in_anti_aliased_and_rate_correct():
    pytest.importorskip("audiosig")
    source_rate = 48000
    positions = np.arange(source_rate, dtype=np.float64)
    above_target_nyquist = np.sin(2 * np.pi * 15000 * positions / source_rate).astype(np.float32)

    loaded = load_reference_audio(
        (above_target_nyquist, source_rate),
        expected_sample_rate=24000,
        resampler="audiosig",
    )

    assert loaded.sample_rate == 24000
    assert loaded.audio.shape == (24000,)
    assert loaded.audio.dtype == np.float32 and loaded.audio.flags.c_contiguous
    assert np.sqrt(np.mean(loaded.audio[256:-256] ** 2)) < 0.01


def test_audiosig_resampling_missing_extra_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "audiosig", None)
    with pytest.raises(OptionalDependencyError, match=r"soprosynth\[audio\]"):
        load_reference_audio(
            (np.ones(8, dtype=np.float32), 48000),
            expected_sample_rate=24000,
            resampler="audiosig",
        )


def test_invalid_resampler_name_is_rejected():
    with pytest.raises(InvalidVoiceError, match="resampler"):
        invalid_resampler: Any = "unknown"
        load_reference_audio((np.ones(4, dtype=np.float32), 24000), resampler=invalid_resampler)
