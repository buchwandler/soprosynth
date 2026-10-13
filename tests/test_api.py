import sys
import types
import wave
from typing import Any, cast

import numpy as np
import pytest

from soprosynth import (
    GenerationConfig,
    InvalidGenerationConfigError,
    SoproRuntime,
    SynthesisRequest,
    SynthesisResult,
    UnsupportedFeatureError,
)
from soprosynth.errors import (
    EmptyTextError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    RuntimeClosedError,
)


def test_validation():
    with pytest.raises(InvalidRequestError):
        SynthesisRequest(id="", text="hello")
    with pytest.raises(InvalidRequestError):
        SynthesisRequest(id="speech", text=cast(Any, 3))
    with pytest.raises(InvalidLanguageError):
        SynthesisRequest(id="speech", text="hello", language="ru")
    with pytest.raises(EmptyTextError):
        SoproRuntime(object()).synthesize_text(" \t", voice="unused.wav")
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(top_p=2)
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(top_k=True)


def test_result_validation_and_float32_normalization():
    result = SynthesisResult(
        id="speech",
        audio=cast(Any, [0.0, 0.5, -0.5]),
        sample_rate=24000,
        text="hello",
        language="en",
    )
    assert result.audio.dtype == np.float32
    assert result.audio.tolist() == pytest.approx([0.0, 0.5, -0.5])
    with pytest.raises(InvalidRequestError):
        SynthesisResult("speech", cast(Any, [float("nan")]), 24000, "hello", "en")


def test_native_backend_wiring(monkeypatch, tmp_path):
    path = tmp_path / "reference.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00")

    prepared_calls = []

    class FakeNative:
        sample_rate = 24000

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def prepare_reference(self, **kwargs):
            prepared_calls.append(kwargs)
            assert kwargs["ref_audio_path"] == path
            return {"voice": 1}

        def synthesize(self, text, **kwargs):
            assert text == "hello"
            assert kwargs["lang"] == "de"
            assert kwargs["ref"] == {"voice": 1}
            return np.array([0.25, -0.25], dtype=np.float32)

        def stream(self, text, **kwargs):
            yield np.array([0.5], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sopro", types.SimpleNamespace(SoproTTS=FakeNative))
    with SoproRuntime.from_pretrained() as runtime:
        voice = runtime.prepare_voice(path)
        assert runtime.prepare_voice(voice) is voice
        assert len(prepared_calls) == 1
        request = SynthesisRequest(id="a", text="hello", language="de")
        result = runtime.synthesize(request, voice=voice)
        assert result.audio.tolist() == pytest.approx([0.25, -0.25])
        assert result.id == "a"
        assert result.metadata["backend"] == "native"
        with pytest.raises(UnsupportedFeatureError):
            runtime.measure_request(request)
        runtime.synthesize(SynthesisRequest(id="b", text="hello", language="de"), voice=voice)
        assert len(prepared_calls) == 1
        assert next(iter(runtime.stream(request, voice=voice))).id == "a"
    with pytest.raises(RuntimeClosedError):
        runtime.prepare_voice(path)

    other = SoproRuntime.from_pretrained()
    with pytest.raises(InvalidVoiceError):
        other.synthesize(request, voice=voice)
    other.close()


def test_unsupported_backend_is_explicit():
    from soprosynth import RuntimeCapabilityError

    with pytest.raises(RuntimeCapabilityError):
        SoproRuntime.from_pretrained(backend="auto")
