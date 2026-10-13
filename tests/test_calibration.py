from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from soprosynth import (
    SoproRuntime,
    SynthesisRequest,
    UnsupportedBundleError,
    VoiceLevelConfig,
)
from soprosynth.calibration import (
    apply_voice_level_calibration,
    calibration_key,
    load_calibration_catalog,
)


def _record(gain_db: float = 6.0, revision: str = "rev-1") -> dict[str, Any]:
    return {
        "gain_db": gain_db,
        "corpus": "licensed-fixture-corpus",
        "measurements": 12,
        "revision": revision,
    }


def test_packaged_calibration_catalog_has_no_unmeasured_records():
    assert load_calibration_catalog() == {}


def test_calibration_keys_require_exact_backend_revision_and_onnx_profile():
    native = {
        "backend": "native",
        "model_id": "samuel-vitorino/sopro-v2-turbo",
        "source_revision": "native-rev",
    }
    onnx = {
        "backend": "onnx",
        "model_id": "sopro:v2-turbo",
        "source_revision": "onnx-rev",
        "profile": "wasm-uint8",
    }
    assert calibration_key(native) == "native:sopro-v2-turbo:native-rev"
    assert calibration_key(onnx) == "onnx:sopro:v2-turbo:onnx-rev:wasm-uint8"
    assert calibration_key({**native, "source_revision": None}) is None
    assert calibration_key({**onnx, "profile": None}) is None


def test_off_mode_and_missing_calibration_leave_audio_unchanged():
    audio = np.array([0.25, -0.5], dtype=np.float32)
    identity = {"backend": "native", "model_id": "sopro-v2-turbo", "source_revision": "rev-1"}
    off_audio, off = apply_voice_level_calibration(
        audio, VoiceLevelConfig(), identity=identity, catalog={}
    )
    missing_audio, missing = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated"),
        identity=identity,
        catalog={},
    )
    assert off_audio.dtype == np.float32 and np.array_equal(off_audio, audio)
    assert off.source == "off" and off.applied is False
    assert np.array_equal(missing_audio, audio)
    assert missing.source == "missing_calibration" and missing.gain_db == 0.0


def test_explicit_static_gain_overrides_missing_catalog_without_clipping():
    audio = np.array([0.75, -0.75], dtype=np.float32)
    result, applied = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated", gain_db=6.0),
        identity=None,
        catalog={},
    )
    assert result.dtype == np.float32
    assert result[0] > 1.0 and result[1] < -1.0
    assert applied.source == "override" and applied.applied is True


def test_exact_calibration_record_applies_but_revision_or_profile_mismatch_does_not():
    audio = np.array([0.5], dtype=np.float32)
    native_identity = {
        "backend": "native",
        "model_id": "sopro-v2-turbo",
        "source_revision": "rev-1",
    }
    native_key = calibration_key(native_identity)
    assert native_key is not None
    catalog = {native_key: _record()}
    calibrated, application = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated"),
        identity=native_identity,
        catalog=catalog,
    )
    assert calibrated[0] == pytest.approx(0.5 * 10 ** (6.0 / 20.0))
    assert application.source == "catalog" and application.applied is True

    changed_revision, missing_revision = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated"),
        identity={**native_identity, "source_revision": "rev-2"},
        catalog=catalog,
    )
    assert np.array_equal(changed_revision, audio)
    assert missing_revision.source == "missing_calibration"

    onnx_identity = {
        "backend": "onnx",
        "model_id": "sopro:v2-turbo",
        "source_revision": "rev-1",
        "profile": "wasm-uint8",
    }
    onnx_key = calibration_key(onnx_identity)
    assert onnx_key is not None
    _, profile_mismatch = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated"),
        identity={**onnx_identity, "profile": "webgpu-fp16"},
        catalog={onnx_key: _record()},
    )
    assert profile_mismatch.source == "missing_calibration"


def test_calibration_catalog_rejects_nonfinite_gain_and_bad_identity():
    key = "native:sopro-v2-turbo:rev-1"
    with pytest.raises(UnsupportedBundleError, match="gain"):
        load_calibration_catalog({"schema": 1, "records": {key: _record(float("nan"))}})
    with pytest.raises(UnsupportedBundleError, match="exact runtime revision"):
        apply_voice_level_calibration(
            np.array([0.5], dtype=np.float32),
            VoiceLevelConfig(mode="calibrated"),
            identity={
                "backend": "native",
                "model_id": "sopro-v2-turbo",
                "source_revision": "rev-1",
            },
            catalog={key: _record(revision="rev-2")},
        )


def test_voice_level_rejects_off_override_and_unrepresentable_gain():
    from soprosynth.errors import InvalidGenerationConfigError

    with pytest.raises(InvalidGenerationConfigError, match="gain_db requires"):
        VoiceLevelConfig(mode="off", gain_db=6.0)

    audio = np.array([0.25], dtype=np.float32)
    with pytest.raises(InvalidGenerationConfigError, match="unrepresentable scale"):
        apply_voice_level_calibration(
            audio,
            VoiceLevelConfig(mode="calibrated", gain_db=1000.0),
            identity=None,
            catalog={},
        )

    invalid_config = VoiceLevelConfig(mode="calibrated", gain_db=6.0)
    object.__setattr__(invalid_config, "mode", "off")
    with pytest.raises(InvalidGenerationConfigError, match="gain_db requires"):
        apply_voice_level_calibration(audio, invalid_config, identity=None, catalog={})


class _NativeEngine:
    sample_rate = 24000

    def prepare_reference(self, reference, *, stream=False):
        return {"prepared": True}

    def synthesize(self, text, **kwargs):
        return np.array([0.75, -0.75], dtype=np.float32)


def test_runtime_applies_voice_level_and_reports_backend_neutral_metadata(monkeypatch):
    from soprosynth import calibration

    monkeypatch.setattr(
        calibration,
        "load_calibration_catalog",
        lambda: {"native:sopro-v2-turbo:rev-1": _record()},
    )
    runtime = SoproRuntime(
        _NativeEngine(),
        backend="native",
        model_id="sopro-v2-turbo",
        source_revision="rev-1",
    )
    voice = runtime.prepare_voice((np.array([0.1, -0.1], dtype=np.float32), 24000))
    result = runtime.synthesize(
        SynthesisRequest("calibrated", "Hello"),
        voice=voice,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    assert result.audio[0] > 1.0
    assert isinstance(result.metadata["voice_identity"], Mapping)
    assert result.metadata["voice_level_application"]["source"] == "catalog"
    assert result.metadata["runtime_diagnostics"]["backend"] == "native"
    assert result.metadata["model_revision"] == "rev-1"
    runtime.close()
