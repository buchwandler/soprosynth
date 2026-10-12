import sys
import types
import wave
import numpy as np
import pytest
from soprosynth import AudioResult, GenerationConfig, SynthesisRequest, SoproRuntime
from soprosynth.errors import OnnxPipelineNotImplementedError


def test_validation():
    with pytest.raises(ValueError):
        SynthesisRequest(" ")
    with pytest.raises(ValueError):
        SynthesisRequest("hello", language="ru")
    with pytest.raises(ValueError):
        GenerationConfig(top_p=2)


def test_save(tmp_path):
    file = AudioResult(np.array([0.0, 0.5, -0.5], dtype="f4"), 24000).save_wav(tmp_path / "out.wav")
    with wave.open(str(file)) as f:
        assert f.getnframes() == 3 and f.getframerate() == 24000


def test_native_backend_wiring(monkeypatch, tmp_path):
    path = tmp_path / "reference.wav"
    path.touch()

    class FakeNative:
        sample_rate = 24000

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def prepare_reference(self, **kw):
            return {"voice": 1}

        def synthesize(self, text, **kwargs):
            assert text == "hello" and kwargs["lang"] == "de" and kwargs["ref"] == {"voice": 1}
            return np.array([0.25, -0.25], dtype=np.float32)

        def stream(self, text, **kwargs):
            yield np.array([0.5], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sopro", types.SimpleNamespace(SoproTTS=FakeNative))
    with SoproRuntime.from_pretrained() as tts:
        v = tts.prepare_voice(path)
        out = tts.synthesize(SynthesisRequest("hello", language="de", id="a"), voice=v)
        assert out.audio.tolist() == pytest.approx([0.25, -0.25])
        assert out.request_id == "a"
        assert len(list(tts.stream("hello", voice=v))) == 1
    with pytest.raises(RuntimeError):
        tts.prepare_voice(path)


def test_onnx_not_claimed():
    class Fake:
        sample_rate = 24000

        def close(self):
            pass

    with SoproRuntime(Fake(), backend="onnx") as tts:
        with pytest.raises(OnnxPipelineNotImplementedError):
            tts.synthesize("test", voice="x.wav")
