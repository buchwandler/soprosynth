"""Opt-in real ONNX integration gate requiring released ONNXVoice and model assets."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from soprosynth import (
    GenerationConfig,
    SoproOutputConfig,
    SoproRuntime,
    SynthesisRequest,
    __version__,
)

_RUN_REAL_ONNX = os.environ.get("SOPROSYNTH_RUN_REAL_ONNX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN_REAL_ONNX,
    reason=(
        "set SOPROSYNTH_RUN_REAL_ONNX_TESTS=1 with released ONNXVoice >=0.2.7,<0.3, "
        "compatible model assets and licensed reference audio"
    ),
)


def _require_released_onnxvoice() -> str:
    try:
        distribution = importlib.metadata.distribution("onnxvoice")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("ONNX integration gate requires a released ONNXVoice installation")
    version = distribution.version
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            origin = json.loads(direct_url)
        except json.JSONDecodeError:
            pytest.fail("ONNXVoice direct_url.json is invalid")
        if origin.get("dir_info", {}).get("editable") is True:
            pytest.fail("ONNX integration gate must use a released wheel, not an editable checkout")
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?", version)
    parsed = tuple(int(component) for component in match.groups()) if match else None
    if parsed is None or not (0, 2, 7) <= parsed < (0, 3, 0):
        pytest.fail(f"ONNX integration requires released ONNXVoice >=0.2.7,<0.3; found {version}")
    return version


def _check_audio(result) -> None:
    assert result.audio.ndim == 1
    assert result.audio.size > 0
    assert np.isfinite(result.audio).all()
    assert result.sample_rate == 24000


def _check_stream(chunks) -> np.ndarray:
    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert sum(chunk.is_final for chunk in chunks) == 1
    assert chunks[-1].is_final
    assert all(chunk.sample_rate == 24_000 for chunk in chunks)
    assert all(
        chunk.audio.ndim == 1
        and chunk.audio.dtype == np.float32
        and chunk.audio.flags.c_contiguous
        and np.isfinite(chunk.audio).all()
        for chunk in chunks
    )
    summary = chunks[-1].metadata
    assert summary["chunk_count"] == len(chunks)
    assert summary["audio_samples_total"] == sum(chunk.audio.size for chunk in chunks)
    return np.concatenate([chunk.audio for chunk in chunks])


def test_real_onnx_managed_install_offline_reopen_reference_and_synthesis(tmp_path, monkeypatch):
    onnxvoice_version = _require_released_onnxvoice()
    reference = os.environ.get("SOPROSYNTH_REAL_ONNX_REFERENCE")
    if not reference or not Path(reference).is_file():
        pytest.fail("SOPROSYNTH_REAL_ONNX_REFERENCE must name an existing licensed PCM WAV")
    model = os.environ.get("SOPROSYNTH_REAL_ONNX_MODEL", "sopro:v2-turbo")
    profile = os.environ.get("SOPROSYNTH_REAL_ONNX_PROFILE", "wasm-uint8")

    stream_text = (
        "Streaming integration exercises multiple model chunks while preserving one prepared voice. "
        "Audio is consumed incrementally rather than collected by SoproSynth."
    )
    stream_config = GenerationConfig(seed=1234)
    raw_output = SoproOutputConfig(mode="raw")
    cold_audio: np.ndarray | None = None
    with SoproRuntime.from_pretrained(
        model=model,
        profile=profile,
        providers="cpu",
    ) as runtime:
        voice = runtime.prepare_voice(reference)
        installed_result = runtime.synthesize(
            SynthesisRequest(id="onnx-installed", text="Hello from SoproSynth.", language="en"),
            voice=voice,
            config=GenerationConfig(temperature=0.8, top_p=0.9),
        )
        _check_audio(installed_result)
        installed_diagnostics = runtime.diagnostics
        assert installed_diagnostics.backend == "onnx"
        assert "CPUExecutionProvider" in installed_diagnostics.providers_active
        cold_chunks = list(
            runtime.stream_text(
                stream_text,
                id="onnx-cold-stream",
                language="en",
                voice=voice,
                config=stream_config,
                output=raw_output,
            )
        )
        cold_audio = _check_stream(cold_chunks)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    # load() opens an installed ONNXVoice asset without catalog refresh or acquisition.
    with SoproRuntime.load(model, profile=profile, providers="cpu") as runtime:
        voice = runtime.prepare_voice(reference)
        reopened_result = runtime.synthesize(
            SynthesisRequest(id="onnx-reopened", text="Bonjour de SoproSynth.", language="fr"),
            voice=voice,
            config=GenerationConfig(temperature=0.8),
        )
        _check_audio(reopened_result)
        reopened_diagnostics = runtime.diagnostics
        output = reopened_result.save_wav(tmp_path / "real-onnx.wav")
        assert reopened_diagnostics.backend == "onnx"
        assert "CPUExecutionProvider" in reopened_diagnostics.providers_active
        runtime.prepare_streaming(voice, config=stream_config)
        warm_chunks = list(
            runtime.stream_text(
                stream_text,
                id="onnx-prewarmed-stream",
                language="en",
                voice=voice,
                config=stream_config,
                output=raw_output,
            )
        )
        warm_audio = _check_stream(warm_chunks)
        assert cold_audio is not None
        np.testing.assert_allclose(warm_audio, cold_audio, rtol=1e-5, atol=1e-6)

        native_chunks = list(
            runtime.stream_text(
                "Native streaming conditioning should produce finite audible output.",
                id="onnx-native-stream",
                language="en",
                voice=voice,
                config=stream_config,
            )
        )
        native_audio = _check_stream(native_chunks)
        assert float(np.max(np.abs(native_audio))) > 0.0
        with wave.open(str(output), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 24000
            assert wav.getnframes() > 0

    print(
        {
            "soprosynth_version": __version__,
            "onnxvoice_version": onnxvoice_version,
            "model": reopened_diagnostics.model_id,
            "revision": reopened_diagnostics.source_revision,
            "profile": profile,
            "providers": reopened_diagnostics.providers_active,
            "duration_seconds": reopened_result.audio.size / reopened_result.sample_rate,
            "installed_revision": installed_diagnostics.source_revision,
        }
    )
