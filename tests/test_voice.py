import wave

import numpy as np
import pytest

from soprosynth import PreparedVoice
from soprosynth.errors import InvalidVoiceError
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
