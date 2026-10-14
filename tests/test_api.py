from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from soprosynth import (
    GenerationConfig,
    InvalidGenerationConfigError,
    SoproRuntime,
    SynthesisRequest,
    SynthesisResult,
)
from soprosynth.errors import (
    EmptyTextError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    RuntimeClosedError,
)
from soprosynth.voice import fingerprint_audio


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


class FakeEngine:
    sample_rate = 24000
    profile = "wasm-uint8"
    model_id = "sopro:v2-turbo"
    source_revision = "revision-1"

    fixed_steps = 32
    max_segment_chars = 300
    token_samples = 1024
    generation_defaults = SimpleNamespace(
        temperature=0.8,
        top_p=0.9,
        top_k=25,
        steps=32,
        max_seconds=30.0,
        min_seconds=0.0,
        ref_seconds=10.0,
    )

    def __init__(self):
        self.prepared = []
        self.prepared_seconds = []
        self.measured = []
        self.closed = False
        self.run_args = None
        self.run_closed = False

    def prepare_reference(self, reference, *, seconds=None):
        self.prepared.append(reference)
        self.prepared_seconds.append(seconds)
        return {"voice": len(self.prepared)}

    def reference_level_db(self, state):
        return -23.0

    def measure_text(self, text, language=None):
        self.measured.append((text, language))
        return 2

    def begin_run(self, state, *, language=None, **generation):
        self.run_args = (state, language, generation)
        engine = self

        class Run:
            def synthesize_segment(self, text):
                return engine.synthesize(text, lang=language, ref=state, **generation)

            def close(self):
                engine.run_closed = True

        return Run()

    def synthesize(self, text, **kwargs):
        assert text == "hello"
        assert kwargs["lang"] == "de"
        assert kwargs["ref"] == {"voice": 1}
        return np.array([0.25, -0.25], dtype=np.float32)

    def close(self):
        self.closed = True


def test_from_pretrained_routes_to_onnxvoice_with_canonical_defaults(monkeypatch):
    import soprosynth._onnxvoice as boundary

    calls = []
    engine = FakeEngine()

    def install_and_open(*args, **kwargs):
        calls.append((args, kwargs))
        return engine

    monkeypatch.setattr(boundary, "install_and_open", install_and_open)
    runtime = SoproRuntime.from_pretrained()

    assert calls == [
        (
            ("sopro:v2-turbo",),
            {
                "profile": "wasm-uint8",
                "providers": "cpu",
                "provider_options": None,
                "session_options": None,
                "force": False,
                "refresh": False,
                "progress": None,
            },
        )
    ]
    assert runtime.backend == "onnx"
    assert runtime.model_id == "sopro:v2-turbo"
    assert runtime.capabilities.supports_request_measurement is True
    assert runtime.capabilities.supports_streaming is False

    voice = runtime.prepare_voice((np.array([0.1, -0.1], dtype=np.float32), 24000))
    assert runtime.prepare_voice(voice) is voice
    request = SynthesisRequest("request-1", "hello", language="de")
    result = runtime.synthesize(request, voice=voice)
    assert result.audio.tolist() == pytest.approx([0.25, -0.25])
    assert engine.run_args == (voice.state, "de", {})
    assert engine.run_closed is True
    assert result.metadata["backend"] == "onnx"
    measured = runtime.measure_request(request, maximum=2)
    assert measured.amount == 2 and measured.fits is True
    assert engine.measured == [("hello", "de")]
    runtime.close()
    assert engine.closed
    with pytest.raises(RuntimeClosedError):
        runtime.prepare_voice((np.array([0.1], dtype=np.float32), 24000))


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_prepare_voice_preserves_source_rate_and_runtime_identity(sample_rate):
    engine = FakeEngine()
    runtime = SoproRuntime(engine)
    audio = np.array([0.1, -0.1], dtype=np.float32)

    voice = runtime.prepare_voice((audio, sample_rate))
    reference = engine.prepared[0]

    assert reference.sample_rate == sample_rate
    assert reference.audio.dtype == np.float32 and reference.audio.flags.c_contiguous
    assert np.array_equal(reference.audio, audio)
    assert voice.sample_rate == runtime.sample_rate == 24000
    assert voice.identity["sample_rate"] == 24000
    assert voice.fingerprint == fingerprint_audio(audio, sample_rate)
    assert voice.metadata == {
        "reference_sample_rate": sample_rate,
        "prepared_reference_sample_rate": sample_rate,
        "reference_seconds": 10.0,
        "reference_level_db": -23.0,
    }
    result = runtime.synthesize(
        SynthesisRequest("source-rate", "hello", language="de"), voice=voice
    )
    assert result.sample_rate == 24000
    runtime.close()


def test_explicit_legacy_resampling_preserves_source_identity():
    engine = FakeEngine()
    runtime = SoproRuntime(engine)
    audio = np.array([0.1, -0.1, 0.25, -0.25, 0.4, -0.4, 0.5, -0.5], dtype=np.float32)

    voice = runtime.prepare_voice((audio, 48000), resampler="linear")
    reference = engine.prepared[0]

    assert reference.sample_rate == 24000
    assert reference.audio.size == 4
    assert voice.sample_rate == 24000
    assert voice.fingerprint == fingerprint_audio(audio, 48000)
    assert voice.metadata == {
        "reference_sample_rate": 48000,
        "prepared_reference_sample_rate": 24000,
        "reference_seconds": 10.0,
        "reference_level_db": -23.0,
    }
    runtime.close()


def test_prepare_voice_forwards_and_validates_reference_seconds():
    engine = FakeEngine()
    runtime = SoproRuntime(engine)
    voice = runtime.prepare_voice(
        (np.array([0.1, -0.1], dtype=np.float32), 24000), reference_seconds=3.5
    )
    assert engine.prepared_seconds == [3.5]
    assert voice.metadata["reference_seconds"] == 3.5

    for invalid in (0, -1, float("nan"), float("inf"), True, "3"):
        with pytest.raises(InvalidGenerationConfigError, match="reference_seconds"):
            runtime.prepare_voice(
                (np.array([0.1, -0.1], dtype=np.float32), 24000),
                reference_seconds=invalid,
            )
    assert engine.prepared_seconds == [3.5]
    runtime.close()


def test_public_api_has_only_onnxvoice_runtime_controls():
    runtime_params = inspect.signature(SoproRuntime.from_pretrained).parameters
    assert set(runtime_params) == {
        "model",
        "profile",
        "providers",
        "provider_options",
        "session_options",
        "force",
        "refresh",
        "progress",
    }
    voice_params = inspect.signature(SoproRuntime.prepare_voice).parameters
    assert "stream" not in voice_params
    runtime = SoproRuntime(FakeEngine())
    assert callable(runtime.stream)
    assert callable(runtime.stream_text)
    assert runtime.capabilities.supports_streaming is False


def test_prepared_voice_remains_runtime_bound():
    first = SoproRuntime(FakeEngine())
    second = SoproRuntime(FakeEngine())
    voice = first.prepare_voice((np.array([0.1], dtype=np.float32), 24000))
    with pytest.raises(InvalidVoiceError, match="owned by another runtime"):
        second.synthesize(SynthesisRequest("request-1", "hello", language="de"), voice=voice)
    first.close()
    second.close()
