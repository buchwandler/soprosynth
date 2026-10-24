from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from soprosynth import (
    GenerationConfig,
    ModelInferenceError,
    RuntimeCapabilityError,
    SoproRuntime,
    SynthesisRequest,
    VoiceLevelConfig,
)
from soprosynth.config import SoproOutputConfig


class SegmentRun:
    def __init__(self, engine, state, language, generation):
        self.engine = engine
        self.state = state
        self.language = language
        self.generation = generation
        self.closed = False
        self.segment_calls = []
        engine.runs.append(self)

    def synthesize_segment(self, text):
        self.segment_calls.append(text)
        self.engine.calls.append((text, self.language))
        self.engine.generation_kwargs.append(dict(self.generation))
        if self.engine.fail_on_segment == text:
            raise RuntimeError("fixture inference failure")
        output = self.engine.outputs.pop(0)
        return np.asarray(output["audio"], dtype=np.float32)

    def close(self):
        self.closed = True
        self.engine.closed_runs += 1


class SegmentEngine:
    sample_rate = 1_000
    profile = "wasm-uint8"
    model_id = "sopro:v2-turbo"
    source_revision = "fixture-revision"
    max_segment_chars = 6
    token_samples = 32
    fixed_steps = 8
    generation_defaults = SimpleNamespace(
        temperature=0.8,
        top_p=0.9,
        top_k=25,
        steps=8,
        max_seconds=30.0,
        min_seconds=0.0,
        ref_seconds=10.0,
    )

    def __init__(self, outputs, *, reference_level=-23.0):
        self.outputs = list(outputs)
        self.reference_level = reference_level
        self.calls = []
        self.generation_kwargs = []
        self.reference_seconds = []
        self.runs = []
        self.closed_runs = 0
        self.fail_on_segment = None
        self.measured = []

    def measure_text(self, text, *, language=None):
        self.measured.append((text, language))
        return 13

    def prepare_reference(self, reference, *, seconds=None):
        self.reference_seconds.append(seconds)
        return {"level_db": self.reference_level}

    def reference_level_db(self, state):
        return state.get("level_db")

    def begin_run(self, state, *, language=None, **generation):
        self.run_arguments = (state, language, generation)
        return SegmentRun(self, state, language, generation)

    def close(self):
        pass


def _voice(runtime):
    return runtime.prepare_voice((np.array([0.1, -0.1], dtype=np.float32), 1_000))


def test_long_text_splits_and_uses_one_onnxvoice_run():
    engine = SegmentEngine(
        [
            {"audio": np.array([0.1, 0.1], dtype=np.float32)},
            {"audio": np.array([0.2, 0.2], dtype=np.float32)},
            {"audio": np.array([0.3, 0.3], dtype=np.float32)},
        ]
    )
    runtime = SoproRuntime(engine)
    result = runtime.synthesize_text(
        "One.\nTwo.  Three.",
        voice=_voice(runtime),
        output=SoproOutputConfig(mode="raw"),
    )
    assert [text for text, _ in engine.calls] == ["One.", "Two.", "Three."]
    assert len(engine.runs) == 1
    assert engine.runs[0].segment_calls == ["One.", "Two.", "Three."]
    assert engine.runs[0].closed is True
    assert all("prompt_tokens" not in run.generation for run in engine.runs)
    np.testing.assert_allclose(result.audio, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    assert result.metadata["sopro_output"]["mode"] == "raw"
    assert result.metadata["sopro_output"]["text_segment_count"] == 3
    runtime.close()


def test_text_chunk_override_seed_and_effective_generation():
    engine = SegmentEngine(
        [
            {"audio": np.array([0.1], dtype=np.float32)},
            {"audio": np.array([0.2], dtype=np.float32)},
        ]
    )
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)
    result = runtime.synthesize_text(
        "One. Two. Three.",
        voice=voice,
        config=GenerationConfig(temperature=0.5, seed=123, text_chunk_size=10),
        output=SoproOutputConfig(mode="raw"),
    )
    assert [text for text, _ in engine.calls] == ["One. Two.", "Three."]
    assert [kwargs["seed"] for kwargs in engine.generation_kwargs] == [123, 123]
    assert len(engine.runs) == 1
    assert "text_chunk_size" not in engine.runs[0].generation
    effective = result.metadata["effective_generation"]
    assert effective == {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 25,
        "steps": 8,
        "max_seconds": 30.0,
        "min_seconds": 0.0,
        "seed": 123,
        "text_chunk_size": 10,
        "reference_seconds": 10.0,
    }
    runtime.close()


def test_wrong_fixed_steps_fail_before_run_creation():
    engine = SegmentEngine([{"audio": np.array([0.1], dtype=np.float32)}])
    runtime = SoproRuntime(engine)
    with pytest.raises(RuntimeCapabilityError, match="fixed at 8"):
        runtime.synthesize_text(
            "Hello.",
            voice=_voice(runtime),
            config=GenerationConfig(steps=9),
            output=SoproOutputConfig(mode="raw"),
        )
    assert engine.runs == []
    runtime.close()


def test_run_closes_when_middle_segment_fails():
    engine = SegmentEngine(
        [{"audio": np.array([0.1], dtype=np.float32)}],
    )
    engine.fail_on_segment = "Two."
    runtime = SoproRuntime(engine)
    with pytest.raises(ModelInferenceError, match="Sopro synthesis failed"):
        runtime.synthesize_text(
            "One. Two.",
            voice=_voice(runtime),
            output=SoproOutputConfig(mode="raw"),
        )
    assert len(engine.runs) == 1
    assert engine.runs[0].closed is True
    assert engine.closed_runs == 1
    runtime.close()


def test_prepared_voice_is_reusable_across_independent_runs():
    engine = SegmentEngine(
        [
            {"audio": np.array([0.1], dtype=np.float32)},
            {"audio": np.array([0.2], dtype=np.float32)},
        ]
    )
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)
    for request_id in ("first", "second"):
        runtime.synthesize(
            SynthesisRequest(request_id, "Hello."),
            voice=voice,
            output=SoproOutputConfig(mode="raw"),
        )
    assert len(engine.runs) == 2
    assert engine.runs[0] is not engine.runs[1]
    assert all(run.state is voice.state and run.closed for run in engine.runs)
    runtime.close()


def test_native_conditioning_precedes_separate_voice_calibration():
    engine = SegmentEngine(
        [{"audio": np.full(200, 1.2, dtype=np.float32), "tokens": (), "limit": 2}],
        reference_level=-23.0,
    )
    runtime = SoproRuntime(engine)
    result = runtime.synthesize_text(
        "Hello.",
        voice=_voice(runtime),
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=6.0),
    )
    output = result.metadata["sopro_output"]
    assert output["mode"] == "native"
    assert output["target_level_db"] == -23.0
    assert output["gain_db"] == 0.0
    assert output["limited"] is True
    assert result.metadata["voice_level_application"]["source"] == "override"
    assert result.audio[0] > 1.9
    runtime.close()


def test_raw_mode_keeps_waveform_and_applies_calibration_as_a_distinct_layer():
    source = np.array([0.75, -0.75], dtype=np.float32)
    engine = SegmentEngine([{"audio": source, "tokens": (), "limit": 2}])
    runtime = SoproRuntime(engine)
    result = runtime.synthesize_text(
        "Hello.",
        voice=_voice(runtime),
        output=SoproOutputConfig(mode="raw"),
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=6.0),
    )
    np.testing.assert_allclose(result.audio, source * 10 ** (6.0 / 20.0), rtol=1e-6)
    assert result.metadata["sopro_output"]["mode"] == "raw"
    assert result.metadata["sopro_output"]["gain_db"] is None
    assert result.metadata["voice_level_application"]["source"] == "override"
    runtime.close()


def test_native_mode_requires_reference_level_before_inference():
    engine = SegmentEngine([{"audio": np.array([0.1], dtype=np.float32)}], reference_level=None)
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)
    assert "reference_level_db" not in voice.metadata
    with pytest.raises(RuntimeCapabilityError, match="reference_level_db"):
        runtime.synthesize(SynthesisRequest("native", "Hello."), voice=voice)
    assert engine.calls == []
    raw_result = runtime.synthesize(
        SynthesisRequest("raw", "Hello."),
        voice=voice,
        output=SoproOutputConfig(mode="raw"),
    )
    assert raw_result.audio.tolist() == pytest.approx([0.1])
    runtime.close()


def test_long_prose_measures_as_fitting_and_synthesizes_multiple_bounded_segments():
    from soprosynth.postprocess import split_text

    text = ("This is a normal sentence with several words. " * 3).strip()
    engine = SegmentEngine(
        [{"audio": np.array([0.1], dtype=np.float32)} for _ in split_text(text, 6)]
    )
    runtime = SoproRuntime(engine)
    request = SynthesisRequest("long-prose", text)

    measured = runtime.measure_request(request)
    assert len(text) > engine.max_segment_chars
    assert measured.amount == 13
    assert measured.maximum is None
    assert measured.fits is True
    assert measured.unit == "tokens"
    assert engine.measured == [(text, None)]
    assert engine.runs == []
    assert engine.calls == []
    assert engine.closed_runs == 0
    assert engine.reference_seconds == []

    voice = _voice(runtime)
    runtime.synthesize(request, voice=voice, output=SoproOutputConfig(mode="raw"))
    segments = engine.runs[0].segment_calls
    assert len(segments) > 1
    assert all(len(segment) <= engine.max_segment_chars for segment in segments)
    assert engine.closed_runs == 1
    runtime.close()
