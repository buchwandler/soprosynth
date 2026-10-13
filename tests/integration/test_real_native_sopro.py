"""Opt-in real native integration test; never downloads model assets in normal CI."""

from __future__ import annotations

import importlib.metadata
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from soprosynth import SoproRuntime, SynthesisRequest, __version__

_RUN_REAL_NATIVE = os.environ.get("SOPROSYNTH_RUN_REAL_NATIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN_REAL_NATIVE,
    reason="set SOPROSYNTH_RUN_REAL_NATIVE_TESTS=1 with licensed model assets and reference audio",
)


def test_real_native_load_prepare_synthesize_reuse_and_write(tmp_path):
    reference = os.environ.get("SOPROSYNTH_REAL_NATIVE_REFERENCE")
    if not reference or not Path(reference).is_file():
        pytest.fail("SOPROSYNTH_REAL_NATIVE_REFERENCE must name an existing licensed PCM WAV")
    model = os.environ.get("SOPROSYNTH_REAL_NATIVE_MODEL", "samuel-vitorino/sopro-v2-turbo")
    device = os.environ.get("SOPROSYNTH_REAL_NATIVE_DEVICE", "cpu")

    with SoproRuntime.from_pretrained(model=model, backend="native", device=device) as runtime:
        voice = runtime.prepare_voice(reference)
        first = runtime.synthesize(
            SynthesisRequest(id="native-en", text="Hello from SoproSynth.", language="en"),
            voice=voice,
        )
        second = runtime.synthesize(
            SynthesisRequest(id="native-de", text="Guten Tag.", language="de"),
            voice=voice,
        )
        assert first.audio.ndim == second.audio.ndim == 1
        assert first.audio.size > 0 and second.audio.size > 0
        assert np.isfinite(first.audio).all() and np.isfinite(second.audio).all()
        assert first.sample_rate == second.sample_rate == 24000
        output = first.save_wav(tmp_path / "real-native.wav")
        with wave.open(str(output), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 24000
            assert wav.getnframes() > 0
        diagnostics = runtime.diagnostics

    print(
        {
            "soprosynth_version": __version__,
            "sopro_version": importlib.metadata.version("sopro"),
            "model": diagnostics.model_id,
            "revision": diagnostics.source_revision,
            "device": diagnostics.device,
            "duration_seconds": first.audio.size / first.sample_rate,
        }
    )
