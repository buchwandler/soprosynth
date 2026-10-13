import wave

import numpy as np
import pytest

from soprosynth import SynthesisResult
from soprosynth.audio import load_pcm_wav
from soprosynth.errors import InvalidVoiceError


def test_save_wav_is_mono_pcm16_clips_only_the_copy(tmp_path):
    source = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = SynthesisResult("a", source, 24000, "text", "en")
    output = result.save_wav(tmp_path / "nested" / "out.wav")
    assert np.array_equal(result.audio, source)
    with wave.open(str(output), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 24000
        assert reader.getnframes() == 5
        pcm = np.frombuffer(reader.readframes(5), dtype="<i2")
    assert pcm.tolist() == [-32767, -32767, 0, 32767, 32767]


def test_pcm_wav_load_mixes_stereo_and_preserves_sample_rate(tmp_path):
    path = tmp_path / "stereo.wav"
    stereo = np.array([[32767, -32768], [16384, 16384]], dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(stereo.tobytes())
    audio, sample_rate = load_pcm_wav(path)
    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.shape == (2,)
    assert audio[0] == pytest.approx(-1 / 65536, abs=1e-6)
    assert audio[1] == pytest.approx(0.5, abs=1e-6)


def test_rejects_compressed_and_malformed_wav(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wave")
    with pytest.raises(InvalidVoiceError):
        load_pcm_wav(path)
