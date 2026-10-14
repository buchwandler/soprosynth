from __future__ import annotations

import numpy as np
import pytest

from soprosynth import SoproOutputConfig
from soprosynth.errors import InvalidGenerationConfigError, RuntimeCapabilityError
from soprosynth.postprocess import condition_sopro_output, split_text


def test_split_text_matches_upstream_sentence_clause_and_word_packing():
    assert split_text("First one. Second two. Third.", 12) == (
        "First one.",
        "Second two.",
        "Third.",
    )
    assert split_text("one, two, three, four", 9) == ("one, two,", "three,", "four")
    assert split_text("  First.\n  Second. ", 50) == ("First. Second.",)
    assert split_text("one two extraordinarilylongword", 5)[-1] == "extraordinarilylongword"


def test_output_config_defaults_native_and_rejects_unknown_modes():
    assert SoproOutputConfig().mode == "native"
    assert SoproOutputConfig(mode="raw").mode == "raw"
    with pytest.raises(InvalidGenerationConfigError, match="native.*raw"):
        SoproOutputConfig(mode="unsupported")  # type: ignore[arg-type]


def test_raw_output_is_unchanged_and_does_not_require_reference_level():
    parts = (
        np.array([0.25, -0.5], dtype=np.float32),
        np.array([0.75], dtype=np.float32),
    )
    audio, metadata = condition_sopro_output(
        parts,
        sample_rate=24_000,
        mode="raw",
        reference_level_db=None,
    )
    np.testing.assert_array_equal(audio, [0.25, -0.5, 0.75])
    assert metadata["gain_db"] is None
    assert metadata["trimmed_lead_samples"] == metadata["trimmed_trail_samples"] == 0
    assert metadata["limiter_applied"] is False


def test_native_gain_trim_and_final_fade_match_sopro_constants():
    source = np.zeros(1_200, dtype=np.float32)
    source[300:700] = 0.2
    audio, metadata = condition_sopro_output(
        [source],
        sample_rate=1_000,
        mode="native",
        reference_level_db=-19.8,
    )
    assert audio.shape == (790,)
    assert metadata["gain_db"] == pytest.approx(-3.2)
    assert metadata["trimmed_lead_samples"] == 210
    assert metadata["trimmed_trail_samples"] == 200
    assert metadata["limited"] is False
    assert audio[100] == pytest.approx(0.2 * 10 ** (-3.2 / 20), abs=1e-7)
    assert audio[-1] == 0.0


def test_native_segments_get_join_fades_then_soft_limit_and_final_fade():
    first = np.full(100, 0.5, dtype=np.float32)
    second = np.full(100, 1.2, dtype=np.float32)
    audio, metadata = condition_sopro_output(
        [first, second],
        sample_rate=1_000,
        mode="native",
        reference_level_db=-23.0,
    )
    assert audio.shape == (200,)
    assert metadata["segment_count"] == 2
    assert metadata["limiter_applied"] is True
    assert metadata["limited"] is True
    assert audio[99] == 0.0 and audio[100] == 0.0
    assert float(np.max(np.abs(audio))) < 1.0
    assert audio[-1] == 0.0


def test_native_mode_requires_prepared_reference_level_and_handles_no_audio():
    with pytest.raises(RuntimeCapabilityError, match="reference_level_db"):
        condition_sopro_output([], sample_rate=24_000, mode="native", reference_level_db=None)
    audio, metadata = condition_sopro_output(
        [],
        sample_rate=24_000,
        mode="native",
        reference_level_db=-23.0,
        empty_audio_samples=1_024,
    )
    assert audio.shape == (1_024,) and not np.any(audio)
    assert metadata["limiter_applied"] is False
