"""Opt-in real native integration gate; never downloads model assets in normal CI."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from soprosynth import (
    GenerationConfig,
    RuntimeClosedError,
    SoproRuntime,
    SynthesisRequest,
    __version__,
)
from soprosynth.audio import save_wav

_RUN_REAL_NATIVE = os.environ.get("SOPROSYNTH_RUN_REAL_NATIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN_REAL_NATIVE,
    reason="set SOPROSYNTH_RUN_REAL_NATIVE_TESTS=1 with pinned model assets and a catalogued reference WAV",
)

_DEFAULT_REFERENCE_SHA256 = "9c35277376fd34d8217672cedd8fa098e2321f243007d190838ac6a93cf4dbaf"


def _require_sopro_version() -> str:
    try:
        version = importlib.metadata.version("sopro")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("native integration gate requires the released sopro>=2.2.0,<3 package")
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?", version)
    parsed = tuple(int(component) for component in match.groups()) if match else None
    if parsed is None or not (2, 2, 0) <= parsed < (3, 0, 0):
        pytest.fail(f"native integration requires sopro>=2.2.0,<3; found {version}")
    return version


def _check_audio(audio: np.ndarray, sample_rate: int) -> None:
    assert audio.ndim == 1
    assert audio.size > 0
    assert np.isfinite(audio).all()
    assert sample_rate == 24000


def _check_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() > 0


def test_real_native_multilingual_reuse_stream_and_write(tmp_path):
    sopro_version = _require_sopro_version()
    reference_value = os.environ.get("SOPROSYNTH_REAL_NATIVE_REFERENCE")
    if not reference_value:
        pytest.fail("SOPROSYNTH_REAL_NATIVE_REFERENCE must name the catalogued CC0 PCM WAV")
    reference = Path(reference_value).expanduser().resolve()
    if not reference.is_file():
        pytest.fail(f"SOPROSYNTH_REAL_NATIVE_REFERENCE does not exist: {reference}")
    reference_bytes = reference.read_bytes()
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    expected_reference_sha256 = os.environ.get(
        "SOPROSYNTH_REAL_NATIVE_REFERENCE_SHA256", _DEFAULT_REFERENCE_SHA256
    )
    if reference_sha256 != expected_reference_sha256:
        pytest.fail(
            "native integration reference SHA-256 mismatch: "
            f"expected {expected_reference_sha256}, got {reference_sha256}"
        )

    model = os.environ.get("SOPROSYNTH_REAL_NATIVE_MODEL", "samuel-vitorino/sopro-v2-turbo")
    revision = os.environ.get("SOPROSYNTH_REAL_NATIVE_REVISION")
    if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        pytest.fail("SOPROSYNTH_REAL_NATIVE_REVISION must be the exact 40-character model commit")
    device = os.environ.get("SOPROSYNTH_REAL_NATIVE_DEVICE", "cpu")
    reference_id = os.environ.get(
        "SOPROSYNTH_REAL_NATIVE_REFERENCE_ID", "kyutai/tts-voices:voice-donations/f9cf"
    )
    config = GenerationConfig(temperature=0.8, top_p=0.9, top_k=40, max_seconds=4.0)
    samples: dict[str, np.ndarray] = {}
    first_result = None

    runtime = SoproRuntime.from_pretrained(
        model=model,
        backend="native",
        device=device,
        revision=revision,
    )
    with runtime:
        voice = runtime.prepare_voice(reference)
        for request_id, text, language in (
            ("native-en", "Hello.", "en"),
            ("native-pt", "Olá.", "pt"),
            ("native-fr", "Bonjour.", "fr"),
            ("native-de", "Guten Tag.", "de"),
        ):
            result = runtime.synthesize(
                SynthesisRequest(id=request_id, text=text, language=language),
                voice=voice,
                config=config,
            )
            _check_audio(result.audio, result.sample_rate)
            samples[language] = result.audio
            if language == "en":
                first_result = result

        assert first_result is not None
        first_path = tmp_path / "real-native.wav"
        first_result.save_wav(first_path)
        _check_wav(first_path)

        stream_voice = runtime.prepare_voice(reference, stream=True)
        chunks = list(
            runtime.stream(
                SynthesisRequest(id="native-stream", text="Hello from native streaming.", language="en"),
                voice=stream_voice,
                config=config,
                chunk_frames=64,
            )
        )
        assert chunks
        for index, chunk in enumerate(chunks):
            assert chunk.index == index
            _check_audio(chunk.audio, chunk.sample_rate)
        streamed = np.concatenate([chunk.audio for chunk in chunks])
        stream_path = tmp_path / "real-native-stream.wav"

        save_wav(streamed, chunks[0].sample_rate, stream_path)
        _check_wav(stream_path)
        diagnostics = runtime.diagnostics
        assert diagnostics.backend == "native"
        assert diagnostics.source_revision == revision

    runtime.close()
    with pytest.raises(RuntimeClosedError):
        runtime.prepare_voice(reference)

    print(
        {
            "soprosynth_version": __version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "sopro_version": sopro_version,
            "model": diagnostics.model_id,
            "model_revision": diagnostics.source_revision,
            "device": diagnostics.device,
            "reference_id": reference_id,
            "reference_sha256": reference_sha256,
            "languages": tuple(samples),
            "duration_seconds": {
                language: audio.size / 24000 for language, audio in samples.items()
            },
            "stream_chunks": len(chunks),
            "stream_duration_seconds": streamed.size / 24000,
        }
    )
