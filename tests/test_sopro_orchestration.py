from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from soprosynth import RuntimeCapabilityError, SoproRuntime, SynthesisRequest, VoiceLevelConfig
from soprosynth.config import SoproOutputConfig


class SegmentEngine:
    sample_rate = 1_000
    profile = "wasm-uint8"
    model_id = "sopro:v2-turbo"
    source_revision = "fixture-revision"
    max_segment_chars = 6
    token_samples = 32

    def __init__(self, outputs, *, reference_level=-23.0):
        self.outputs = list(outputs)
        self.reference_level = reference_level
        self.calls = []

    def prepare_reference(self, reference):
        return {"level_db": self.reference_level}

    def reference_level_db(self, state):
        return state.get("level_db")

    def synthesize_segment(self, text, **kwargs):
        self.calls.append((text, kwargs.get("prompt_tokens")))
        output = self.outputs.pop(0)
        return SimpleNamespace(
            audio=output["audio"],
            semantic_tokens=output.get("tokens"),
            prompt_token_limit=output.get("limit"),
        )

    def close(self):
        pass


def _voice(runtime):
    return runtime.prepare_voice((np.array([0.1, -0.1], dtype=np.float32), 1_000))


def test_long_text_splits_normalizes_and_carries_last_prompt_window():
    engine = SegmentEngine(
        [
            {"audio": np.array([0.1, 0.1], dtype=np.float32), "tokens": (10, 11, 12), "limit": 2},
            {"audio": np.array([0.2, 0.2], dtype=np.float32), "tokens": (), "limit": 2},
            {"audio": np.array([0.3, 0.3], dtype=np.float32), "tokens": (20, 21), "limit": 2},
        ]
    )
    runtime = SoproRuntime(engine)
    result = runtime.synthesize_text(
        "One.\nTwo.  Three.",
        voice=_voice(runtime),
        output=SoproOutputConfig(mode="raw"),
    )
    assert [text for text, _ in engine.calls] == ["One.", "Two.", "Three."]
    assert [prompt for _, prompt in engine.calls] == [None, (11, 12), (11, 12)]
    np.testing.assert_allclose(result.audio, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    assert result.metadata["sopro_output"]["mode"] == "raw"
    assert result.metadata["sopro_output"]["text_segment_count"] == 3
    runtime.close()


def test_long_text_without_semantic_carry_contract_fails_before_inference():
    class LegacyEngine:
        sample_rate = 1_000
        profile = "wasm-uint8"
        max_segment_chars = 6

        def __init__(self):
            self.calls = 0

        def prepare_reference(self, reference):
            return object()

        def synthesize(self, text, **kwargs):
            self.calls += 1
            return np.array([0.25], dtype=np.float32)

        def close(self):
            pass

    engine = LegacyEngine()
    runtime = SoproRuntime(engine)
    with pytest.raises(RuntimeCapabilityError, match="semantic prompt carry"):
        runtime.synthesize_text(
            "One. Two. Three.",
            voice=_voice(runtime),
            output=SoproOutputConfig(mode="raw"),
        )
    assert engine.calls == 0
    runtime.close()


def test_missing_semantic_metadata_fails_before_the_next_segment():
    engine = SegmentEngine([{"audio": np.array([0.1], dtype=np.float32)}])
    runtime = SoproRuntime(engine)
    with pytest.raises(RuntimeCapabilityError, match="generated semantic tokens"):
        runtime.synthesize_text(
            "One. Two.",
            voice=_voice(runtime),
            output=SoproOutputConfig(mode="raw"),
        )
    assert engine.calls == [("One.", None)]
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
