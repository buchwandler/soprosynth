from types import SimpleNamespace

import numpy as np
import pytest

from soprosynth import (
    GenerationConfig,
    ModelInferenceError,
    RuntimeCapabilityError,
    RuntimeClosedError,
    SoproRuntime,
    SynthesisChunk,
    SynthesisRequest,
    VoiceLevelConfig,
)
from soprosynth.config import SoproOutputConfig


def _chunk(audio: np.ndarray, *, final: bool = False):
    return (audio, 1_000, final, {"provider": "fixture"})


class StreamingRun:
    def __init__(self, engine, state, language, generation):
        self.engine = engine
        self.state = state
        self.language = language
        self.generation = generation
        self.closed = False
        self.segment_calls = []
        engine.runs.append(self)

    def stream_segment(self, text, *, language=None):
        self.segment_calls.append((text, language))
        if self.engine.fail_on_segment == text:
            yield _chunk(np.array([0.1], dtype=np.float32))
            raise RuntimeError("fixture stream failure")
        yield from self.engine.outputs.pop(0)

    def close(self):
        self.closed = True
        self.engine.closed_runs += 1


class StreamingEngine:
    sample_rate = 1_000
    profile = "wasm-uint8"
    model_id = "sopro:v2-turbo"
    source_revision = "fixture-revision"
    max_segment_chars = 6
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
        self.runs = []
        self.closed_runs = 0
        self.fail_on_segment = None
        self.prepare_calls = []

    def prepare_reference(self, reference, *, seconds=None):
        return {"level_db": self.reference_level}

    def reference_level_db(self, state):
        return state.get("level_db")

    def prepare_streaming(self, state, **controls):
        self.prepare_calls.append((state, dict(controls)))

    def begin_run(self, state, *, language=None, **generation):
        return StreamingRun(self, state, language, generation)

    def close(self):
        pass


def _voice(runtime):
    return runtime.prepare_voice((np.array([0.1, -0.1], dtype=np.float32), 1_000))


def test_stream_text_emits_monotonic_chunks_and_one_final_chunk_across_segments():
    engine = StreamingEngine(
        [
            [
                _chunk(np.array([0.1, 0.2], dtype=np.float32)),
                _chunk(np.array([0.3], dtype=np.float32), final=True),
            ],
            [
                _chunk(np.array([-0.1, -0.2], dtype=np.float32)),
                _chunk(np.array([-0.3], dtype=np.float32), final=True),
            ],
            [_chunk(np.array([0.4], dtype=np.float32), final=True)],
        ]
    )
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)
    chunks = list(
        runtime.stream_text(
            "First. Second.",
            id="request-1",
            voice=voice,
            output=SoproOutputConfig(mode="raw"),
        )
    )

    assert all(isinstance(chunk, SynthesisChunk) for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(4))
    assert [chunk.segment_index for chunk in chunks] == [0, 0, 1, 1]
    assert [chunk.is_final for chunk in chunks] == [False, False, False, True]
    assert sum(chunk.is_final for chunk in chunks) == 1
    assert all(chunk.sample_rate == 1_000 for chunk in chunks)
    assert all(
        chunk.audio.dtype == np.float32 and chunk.audio.flags.c_contiguous for chunk in chunks
    )
    assert [call[0] for call in engine.runs[0].segment_calls] == ["First.", "Second."]
    assert engine.runs[0].closed
    summary = chunks[-1].metadata
    assert summary["stream_final"] is True
    assert summary["chunk_count"] == len(chunks)
    assert summary["segment_count"] == 2
    assert summary["audio_samples_total"] == 6
    assert summary["first_chunk_ms"] >= 0
    assert summary["total_stream_ms"] >= summary["first_chunk_ms"]

    list(runtime.stream_text("Again.", voice=voice, output=SoproOutputConfig(mode="raw")))
    assert len(engine.runs) == 2
    assert engine.runs[1].state is voice.state
    assert engine.runs[1].closed
    runtime.close()


def test_raw_stream_applies_resolved_voice_level_gain_after_backend_chunks():
    source = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    engine = StreamingEngine([[_chunk(source[:1]), _chunk(source[1:], final=True)]])
    runtime = SoproRuntime(engine)
    chunks = list(
        runtime.stream(
            SynthesisRequest("calibrated", "Hello."),
            voice=_voice(runtime),
            output=SoproOutputConfig(mode="raw"),
            voice_level=VoiceLevelConfig(mode="calibrated", gain_db=6.0),
        )
    )

    actual = np.concatenate([chunk.audio for chunk in chunks])
    np.testing.assert_allclose(actual, source * 10 ** (6.0 / 20.0), rtol=1e-6)
    assert chunks[-1].metadata["voice_level_application"]["source"] == "override"
    assert chunks[-1].is_final
    runtime.close()


def test_native_stream_gates_across_chunks_then_applies_calibration_and_final_fade():
    source = np.concatenate(
        (
            np.zeros(250, dtype=np.float32),
            np.full(500, 0.2, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    engine = StreamingEngine(
        [
            [
                _chunk(source[:275]),
                _chunk(source[275:600]),
                _chunk(source[600:], final=True),
            ]
        ]
    )
    runtime = SoproRuntime(engine)
    chunks = list(
        runtime.stream_text(
            "Hello.",
            voice=_voice(runtime),
            voice_level=VoiceLevelConfig(mode="calibrated", gain_db=6.0),
        )
    )
    audio = np.concatenate([chunk.audio for chunk in chunks])

    assert chunks
    assert chunks[-1].is_final
    assert np.count_nonzero(audio[:80]) == 0
    assert audio[100] == pytest.approx(0.2 * 10 ** (6.0 / 20.0), abs=1e-6)
    assert audio[-1] == 0.0
    assert all(np.isfinite(chunk.audio).all() and chunk.audio.ndim == 1 for chunk in chunks)
    runtime.close()


def test_stream_failure_maps_error_and_closes_run():
    engine = StreamingEngine([[]])
    engine.fail_on_segment = "Hello."
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)

    with pytest.raises(ModelInferenceError, match="Sopro streaming failed"):
        list(runtime.stream_text("Hello.", voice=voice, output=SoproOutputConfig(mode="raw")))

    assert len(engine.runs) == 1
    assert engine.runs[0].closed
    runtime.close()


def test_early_stream_close_releases_segment_and_run():
    engine = StreamingEngine(
        [
            [
                _chunk(np.array([0.1], dtype=np.float32)),
                _chunk(np.array([0.2], dtype=np.float32)),
                _chunk(np.array([0.3], dtype=np.float32), final=True),
            ]
        ]
    )
    runtime = SoproRuntime(engine)
    iterator = runtime.stream_text(
        "Hello.", voice=_voice(runtime), output=SoproOutputConfig(mode="raw")
    )

    first = next(iterator)
    assert first.index == 0 and not first.is_final
    iterator.close()

    assert engine.runs[0].closed
    assert engine.closed_runs == 1
    runtime.close()


def test_native_stream_requires_prepared_reference_level_before_opening_run():
    engine = StreamingEngine([[]], reference_level=None)
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)

    with pytest.raises(RuntimeCapabilityError, match="reference_level_db"):
        list(runtime.stream_text("Hello.", voice=voice))

    assert engine.runs == []
    runtime.close()


def test_onnxvoice_run_adapter_validates_raw_chunks_and_closes_segment_stream():
    from soprosynth._onnxvoice import OnnxVoiceRun

    class UpstreamRun:
        def __init__(self):
            self.stream_closed = False
            self.closed = False

        def stream_segment(self, text, *, language=None):
            try:
                yield SimpleNamespace(
                    audio=np.array([0.25], dtype=np.float64),
                    sample_rate=1_000,
                    is_final=True,
                    metadata={"provider": "fixture"},
                )
            finally:
                self.stream_closed = True

        def close(self):
            self.closed = True

    upstream = UpstreamRun()
    run = OnnxVoiceRun(upstream)
    chunks = run.stream_segment("Hello.")
    audio, sample_rate, is_final, metadata = next(chunks)

    assert audio.dtype == np.float32 and audio.flags.c_contiguous
    assert sample_rate == 1_000 and is_final is True
    assert metadata["provider"] == "fixture"
    close_stream = getattr(chunks, "close", None)
    assert callable(close_stream)
    close_stream()
    assert upstream.stream_closed
    run.close()
    assert upstream.closed


def test_prepare_streaming_is_opaque_forwards_only_state_controls_and_preserves_equivalence():
    source = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    engine = StreamingEngine(
        [
            [_chunk(source[:1]), _chunk(source[1:], final=True)],
            [_chunk(source[:1]), _chunk(source[1:], final=True)],
        ]
    )
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)
    metadata_before = dict(voice.metadata)
    config = GenerationConfig(temperature=0.4, seed=123, steps=8, text_chunk_size=10)
    output = SoproOutputConfig(mode="raw")

    cold = list(runtime.stream_text("Hello.", voice=voice, config=config, output=output))
    assert runtime.prepare_streaming(voice, config=config) is None
    assert engine.prepare_calls == [(voice.state, {"seed": 123, "steps": 8})]
    assert dict(voice.metadata) == metadata_before
    warm = list(runtime.stream_text("Hello.", voice=voice, config=config, output=output))

    np.testing.assert_array_equal(
        np.concatenate([chunk.audio for chunk in cold]),
        np.concatenate([chunk.audio for chunk in warm]),
    )
    assert warm[-1].metadata["chunk_count"] == len(warm)
    assert warm[-1].metadata["audio_samples_total"] == source.size
    assert len(engine.runs) == 2 and all(run.closed for run in engine.runs)
    assert not any("cache" in key.lower() for key in voice.metadata)

    runtime.close()
    with pytest.raises(RuntimeClosedError):
        runtime.prepare_streaming(voice, config=config)


def test_prepare_streaming_rejects_wrong_fixed_steps_before_backend_call():
    engine = StreamingEngine([])
    runtime = SoproRuntime(engine)
    voice = _voice(runtime)

    with pytest.raises(RuntimeCapabilityError, match="fixed at 8"):
        runtime.prepare_streaming(voice, config=GenerationConfig(steps=7))

    assert engine.prepare_calls == []
    runtime.close()


def test_onnxvoice_backend_forwards_only_non_none_streaming_controls():
    from soprosynth._onnxvoice import OnnxVoiceBackend

    class CacheEngine:
        def __init__(self):
            self.calls = []

        def prepare_streaming(self, state, **controls):
            self.calls.append((state, controls))

    engine = CacheEngine()
    backend = OnnxVoiceBackend(engine, profile="wasm-uint8", requested_providers="cpu")
    state = object()

    backend.prepare_streaming(state, seed=42, steps=8)
    backend.prepare_streaming(state)

    assert engine.calls == [(state, {"seed": 42, "steps": 8}), (state, {})]
