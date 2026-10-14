import numpy as np
import pytest

from soprosynth import SynthesisChunk
from soprosynth.errors import InvalidRequestError, RuntimeCapabilityError
from soprosynth.postprocess import SoproStreamConditioner


def _collect(*groups: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    return [chunk for group in groups for chunk in group]


def test_synthesis_chunk_validates_and_freezes_public_metadata():
    chunk = SynthesisChunk(
        id="request-1",
        audio=np.array([0.1, -0.2], dtype=np.float32),
        sample_rate=24_000,
        index=0,
        segment_index=0,
        is_final=False,
        metadata={"detail": {"values": [1, 2]}},
    )

    assert chunk.audio.dtype == np.float32
    assert chunk.audio.flags.c_contiguous
    assert chunk.metadata["detail"]["values"] == (1, 2)
    assert type(chunk.metadata).__name__ == "mappingproxy"
    with pytest.raises(InvalidRequestError, match="index"):
        SynthesisChunk("x", np.array([0.1]), 24_000, True, 0, False)


def test_native_conditioner_carries_onset_across_chunks_and_trims_tail():
    sample_rate = 1_000
    source = np.concatenate(
        (
            np.zeros(250, dtype=np.float32),
            np.full(500, 0.3, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    conditioner = SoproStreamConditioner(
        sample_rate=sample_rate,
        mode="native",
        reference_level_db=-23.0,
    )
    conditioner.begin_segment(first=True, last=True)

    first = conditioner.push(source[:275])
    second = conditioner.push(source[275:600])
    third = conditioner.push(source[600:])
    final = conditioner.finish_segment()
    chunks = _collect(first, second, third, final)
    result = np.concatenate(chunks)

    assert first == ()  # onset is not stable until audio from the next chunk arrives
    assert chunks
    assert all(chunk.size > 0 for chunk in chunks)
    assert all(chunk.ndim == 1 and chunk.dtype == np.float32 for chunk in chunks)
    assert all(chunk.flags.c_contiguous and np.isfinite(chunk).all() for chunk in chunks)
    assert np.count_nonzero(result[:80]) == 0
    assert 870 <= result.size <= 900
    assert result[-1] == 0.0


def test_later_native_segment_uses_skip_gate_fade_and_limiter():
    sample_rate = 1_000
    source = np.concatenate(
        (
            np.zeros(250, dtype=np.float32),
            np.full(500, 1.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    conditioner = SoproStreamConditioner(
        sample_rate=sample_rate,
        mode="native",
        reference_level_db=-23.0,
    )
    conditioner.begin_segment(first=False, last=False)
    chunks = _collect(conditioner.push(source), conditioner.finish_segment())
    result = np.concatenate(chunks)

    first_active = int(np.flatnonzero(result)[0])
    assert 140 <= first_active <= 160  # retain the 100 ms skip and onset safety margin
    assert float(np.max(np.abs(result))) < 1.0
    assert result[-1] == 0.0  # inter-segment join fade


def test_raw_conditioner_passes_chunks_through_without_native_processing():
    conditioner = SoproStreamConditioner(sample_rate=24_000, mode="raw", reference_level_db=None)
    conditioner.begin_segment(first=True, last=True)
    source = np.array([0.2, -0.3], dtype=np.float64)

    output = conditioner.push(source)

    assert len(output) == 1
    np.testing.assert_array_equal(output[0], source.astype(np.float32))
    assert output[0].flags.c_contiguous
    assert conditioner.finish_segment() == ()
    assert conditioner.finish() == ()


def test_native_conditioner_requires_reference_calibration_and_never_yields_empty():
    with pytest.raises(RuntimeCapabilityError, match="reference_level_db"):
        SoproStreamConditioner(sample_rate=24_000, mode="native", reference_level_db=None)

    conditioner = SoproStreamConditioner(
        sample_rate=1_000,
        mode="native",
        reference_level_db=-23.0,
    )
    conditioner.begin_segment(first=True, last=True)
    assert conditioner.push(np.empty(0, dtype=np.float32)) == ()
    assert conditioner.finish_segment() == ()
