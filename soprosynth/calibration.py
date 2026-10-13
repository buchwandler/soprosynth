"""Exact-identity static voice-level calibration; no loudness analysis is performed."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib import resources
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .config import VoiceLevelApplication, VoiceLevelConfig
from .errors import InvalidGenerationConfigError, UnsupportedBundleError

_CALIBRATION_RESOURCE = "data/voice_level_calibration.json"


def _validate_catalog(raw: Any) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(raw, Mapping) or type(raw.get("schema")) is not int or raw.get("schema") != 1:
        raise UnsupportedBundleError("Voice-level calibration catalog must use schema 1")
    records = raw.get("records")
    if not isinstance(records, Mapping):
        raise UnsupportedBundleError("Voice-level calibration records must be an object")
    validated: dict[str, Mapping[str, Any]] = {}
    for key, record in records.items():
        if not isinstance(key, str) or not key or not isinstance(record, Mapping):
            raise UnsupportedBundleError("Voice-level calibration entries must be keyed objects")
        if set(record) != {"gain_db", "corpus", "measurements", "revision"}:
            raise UnsupportedBundleError(f"Calibration record {key!r} has an unsupported schema")
        gain = record.get("gain_db")
        if isinstance(gain, bool) or not isinstance(gain, Real) or not math.isfinite(float(gain)):
            raise UnsupportedBundleError(f"Invalid voice-level calibration gain for {key!r}")
        revision = record.get("revision")
        corpus = record.get("corpus")
        measurements = record.get("measurements")
        if not isinstance(revision, str) or not revision:
            raise UnsupportedBundleError(f"Calibration record {key!r} must declare its revision")
        if not isinstance(corpus, str) or not corpus:
            raise UnsupportedBundleError(f"Calibration record {key!r} must declare its corpus")
        if type(measurements) is not int or measurements <= 0:
            raise UnsupportedBundleError(
                f"Calibration record {key!r} must declare a positive measurement count"
            )
        validated[key] = MappingProxyType(
            {
                "gain_db": float(gain),
                "corpus": corpus,
                "measurements": measurements,
                "revision": revision,
            }
        )
    return MappingProxyType(validated)


def load_calibration_catalog(
    source: str | Path | Mapping[str, Any] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Load validated records from the packaged catalog or an explicit JSON file/object."""
    if isinstance(source, Mapping):
        raw = source
    else:
        if source is None:
            resource = resources.files("soprosynth").joinpath(_CALIBRATION_RESOURCE)
            if not resource.is_file():
                return MappingProxyType({})
            try:
                text = resource.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise UnsupportedBundleError(
                    "Could not read packaged voice-level calibration"
                ) from exc
        else:
            try:
                text = Path(source).expanduser().read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise UnsupportedBundleError(
                    "Could not read voice-level calibration catalog"
                ) from exc
        try:
            raw = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UnsupportedBundleError("Invalid JSON in voice-level calibration catalog") from exc
    return _validate_catalog(raw)


def calibration_key(identity: Mapping[str, Any] | None) -> str | None:
    """Build a stable key only when the backend, model, exact revision, and profile are known."""
    if not isinstance(identity, Mapping):
        return None
    backend = identity.get("backend")
    model_id = identity.get("model_id")
    revision = identity.get("source_revision") or identity.get("revision")
    if (
        not isinstance(backend, str)
        or backend not in {"native", "onnx"}
        or not isinstance(model_id, str)
        or not model_id.strip()
        or not isinstance(revision, str)
        or not revision.strip()
    ):
        return None
    canonical_model = model_id.strip()
    if canonical_model in {"samuel-vitorino/sopro-v2-turbo", "sopro-v2-turbo", "v2-turbo"}:
        canonical_model = "sopro-v2-turbo"
    elif canonical_model.startswith("sopro:"):
        canonical_model = canonical_model.removeprefix("sopro:")
    if backend == "native":
        return f"native:{canonical_model}:{revision}"
    profile = identity.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        return None
    return f"onnx:sopro:{canonical_model}:{revision}:{profile}"


def apply_voice_level_calibration(
    audio: Any,
    config: VoiceLevelConfig,
    *,
    identity: Mapping[str, Any] | None,
    catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, VoiceLevelApplication]:
    """Apply an explicit or exact-identity static gain without clipping or measuring audio."""
    if not isinstance(config, VoiceLevelConfig):
        raise InvalidGenerationConfigError("config must be a VoiceLevelConfig")
    source = "off"
    gain_db = 0.0
    if config.gain_db is not None:
        gain_db = float(config.gain_db)
        source = "override"
    elif config.mode == "calibrated":
        key = calibration_key(identity)
        record = catalog.get(key) if key is not None else None
        if record is None:
            source = "missing_calibration"
        else:
            if not isinstance(record, Mapping):
                raise UnsupportedBundleError(f"Invalid voice-level calibration record for {key!r}")
            if not isinstance(identity, Mapping):
                raise UnsupportedBundleError(
                    "Calibration identity is unavailable for the selected record"
                )
            revision = identity.get("source_revision") or identity.get("revision")
            if record.get("revision") != revision:
                raise UnsupportedBundleError(
                    f"Calibration record {key!r} is not bound to the exact runtime revision"
                )
            record_gain = record.get("gain_db")
            if (
                isinstance(record_gain, bool)
                or not isinstance(record_gain, Real)
                or not math.isfinite(float(record_gain))
            ):
                raise UnsupportedBundleError(f"Invalid voice-level calibration gain for {key!r}")
            gain_db = float(record_gain)
            source = "catalog"

    try:
        scale = np.float32(10.0 ** (gain_db / 20.0))
    except OverflowError as exc:
        raise InvalidGenerationConfigError(
            "voice-level gain produces an unrepresentable scale"
        ) from exc
    if not np.isfinite(scale):
        raise InvalidGenerationConfigError("voice-level gain produces an unrepresentable scale")
    output = np.asarray(audio, dtype=np.float32) * scale
    if not np.isfinite(output).all():
        raise InvalidGenerationConfigError("voice-level gain produces non-finite audio")
    application = VoiceLevelApplication(
        mode=config.mode,
        gain_db=gain_db,
        source=source,
        applied=source in {"override", "catalog"} and gain_db != 0.0,
    )
    return output, application
