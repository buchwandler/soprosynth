import sys
import types
import wave

import numpy as np
import pytest

from soprosynth._native import NativeBackend
from soprosynth.errors import OptionalDependencyError, SessionCreationError
from soprosynth.voice import ReferenceAudio


def test_native_boundary_imports_lazily_and_forwards_options(monkeypatch):
    calls = {}

    class Engine:
        sample_rate = 24000

        def prepare_reference(self, **kwargs):
            calls["prepare"] = kwargs
            return "prepared-state"

        def synthesize(self, text, **kwargs):
            calls["synthesize"] = (text, kwargs)
            return np.array([0.25], dtype=np.float32)

        def stream(self, text, **kwargs):
            yield text, kwargs

        def close(self):
            calls["closed"] = True

    class FakeTTS:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls["load"] = (model, kwargs)
            return Engine()

    monkeypatch.setitem(sys.modules, "sopro", types.SimpleNamespace(SoproTTS=FakeTTS))
    backend = NativeBackend.from_pretrained(
        "m", device="cpu", revision="rev", cache_dir="cache", quantization=None
    )
    assert calls["load"] == (
        "m",
        {"device": "cpu", "revision": "rev", "cache_dir": "cache", "quantization": None},
    )
    ref = ReferenceAudio(
        np.array([0.0, 0.5], dtype=np.float32),
        24000,
        source="/tmp/source.wav",
        native_source="/tmp/source.wav",
    )
    assert backend.prepare_reference(ref) == "prepared-state"
    assert calls["prepare"]["ref_audio_path"].as_posix() == "/tmp/source.wav"
    assert backend.synthesize("hello", temperature=0.7).tolist() == [0.25]
    assert next(backend.stream("hello"))[0] == "hello"
    backend.close()
    assert calls["closed"] is True


def test_reference_array_uses_temporary_pcm_wav():
    captured = {}

    class Engine:
        sample_rate = 24000

        def prepare_reference(self, *, ref_audio_path, stream):
            with wave.open(str(ref_audio_path), "rb") as reader:
                captured["rate"] = reader.getframerate()
                captured["width"] = reader.getsampwidth()
                captured["frames"] = reader.getnframes()
            captured["path"] = str(ref_audio_path)
            return "state"

    backend = NativeBackend(Engine())
    reference = ReferenceAudio(np.array([0.0, 0.5], dtype=np.float32), 24000)
    assert backend.prepare_reference(reference) == "state"
    assert captured["rate"] == 24000
    assert captured["width"] == 2
    assert captured["frames"] == 2
    assert not __import__("pathlib").Path(captured["path"]).exists()


def test_native_dependency_and_model_errors_are_typed(monkeypatch):
    monkeypatch.setitem(sys.modules, "sopro", None)
    with pytest.raises(OptionalDependencyError):
        NativeBackend.from_pretrained(
            "m", device="cpu", revision=None, cache_dir=None, quantization=None
        )

    class BrokenTTS:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("load failed")

    monkeypatch.setitem(sys.modules, "sopro", types.SimpleNamespace(SoproTTS=BrokenTTS))
    with pytest.raises(SessionCreationError):
        NativeBackend.from_pretrained(
            "m", device="cpu", revision=None, cache_dir=None, quantization=None
        )
