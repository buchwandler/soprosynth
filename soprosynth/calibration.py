"""Exact-identity static voice-level calibration; no loudness analysis is performed."""

from __future__ import annotations

import json
import math
import statistics
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
    if not isinstance(raw, Mapping) or type(raw.get("schema")) is not int or raw.get("schema") not in (1, 2):
        raise UnsupportedBundleError("Voice-level calibration catalog must use schema 1 or 2")
    schema = raw["schema"]
    records = raw.get("records")
    if not isinstance(records, Mapping):
        raise UnsupportedBundleError("Voice-level calibration records must be an object")
    legacy_fields = {"gain_db", "corpus", "measurements", "revision"}
    measured_fields = {
        "gain_db",
        "corpus",
        "license",
        "provenance",
        "measurements",
        "revision",
        "target_lufs",
        "max_true_peak_dbtp",
        "median_measured_lufs",
        "max_measured_sample_peak_dbfs",
        "max_measured_true_peak_dbtp",
        "predicted_median_lufs",
        "predicted_max_sample_peak_dbfs",
        "predicted_max_true_peak_dbtp",
        "max_abs_gain_db",
        "measurement_tool",
        "measurement_tool_version",
        "measurement_method",
        "sample_rates",
        "durations_seconds",
        "measured_lufs",
        "measured_sample_peak_dbfs",
        "measured_true_peak_dbtp",
        "sample_sha256s",
    }
    validated: dict[str, Mapping[str, Any]] = {}
    for key, record in records.items():
        if not isinstance(key, str) or not key or not isinstance(record, Mapping):
            raise UnsupportedBundleError("Calibration entries must be keyed objects")
        expected_fields = legacy_fields if schema == 1 else measured_fields
        if set(record) != expected_fields:
            raise UnsupportedBundleError(f"Calibration record {key!r} has an unsupported schema")
        gain = record.get("gain_db")
        if isinstance(gain, bool) or not isinstance(gain, Real) or not math.isfinite(float(gain)):
            raise UnsupportedBundleError(f"Invalid voice-level calibration gain for {key!r}")
        revision = record.get("revision")
        corpus = record.get("corpus")
        measurements = record.get("measurements")
        if not isinstance(revision, str) or not revision.strip():
            raise UnsupportedBundleError(f"Calibration record {key!r} must declare its revision")
        if not isinstance(corpus, str) or not corpus.strip():
            raise UnsupportedBundleError(f"Calibration record {key!r} must declare its corpus")
        if type(measurements) is not int or measurements <= 0:
            raise UnsupportedBundleError(
                f"Calibration record {key!r} must declare a positive measurement count"
            )
        if schema == 1:
            validated[key] = MappingProxyType(
                {
                    "gain_db": float(gain),
                    "corpus": corpus,
                    "measurements": measurements,
                    "revision": revision,
                }
            )
            continue

        for field in ("license", "provenance", "measurement_tool_version", "measurement_method"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise UnsupportedBundleError(
                    f"Calibration record {key!r} must declare non-empty {field} metadata"
                )
        if record.get("measurement_tool") != "audiosig":
            raise UnsupportedBundleError(
                f"Calibration record {key!r} must name audiosig as its measurement tool"
            )
        sample_hashes = record.get("sample_sha256s")
        if (
            not isinstance(sample_hashes, (list, tuple))
            or len(sample_hashes) != measurements
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in sample_hashes
            )
            or len(set(sample_hashes)) != len(sample_hashes)
        ):
            raise UnsupportedBundleError(
                f"Calibration record {key!r} must carry one distinct lowercase SHA-256 per measurement"
            )
        sample_rates = record.get("sample_rates")
        if (
            not isinstance(sample_rates, (list, tuple))
            or len(sample_rates) != measurements
            or any(type(rate) is not int or rate <= 0 for rate in sample_rates)
        ):
            raise UnsupportedBundleError(
                f"Calibration record {key!r} must record a positive sample rate per measurement"
            )
        numeric_arrays: dict[str, tuple[float, ...]] = {}
        for field in (
            "durations_seconds",
            "measured_lufs",
            "measured_sample_peak_dbfs",
            "measured_true_peak_dbtp",
        ):
            values = record.get(field)
            if not isinstance(values, (list, tuple)) or len(values) != measurements:
                raise UnsupportedBundleError(
                    f"Calibration record {key!r} must record {field} per measurement"
                )
            converted: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise UnsupportedBundleError(f"Calibration record {key!r} has invalid {field}")
                converted.append(float(value))
            if field == "durations_seconds" and any(value <= 0 for value in converted):
                raise UnsupportedBundleError(f"Calibration record {key!r} has non-positive durations")
            numeric_arrays[field] = tuple(converted)
        numeric_fields = (
            "target_lufs",
            "max_true_peak_dbtp",
            "median_measured_lufs",
            "max_measured_sample_peak_dbfs",
            "max_measured_true_peak_dbtp",
            "predicted_median_lufs",
            "predicted_max_sample_peak_dbfs",
            "predicted_max_true_peak_dbtp",
            "max_abs_gain_db",
        )
        numeric: dict[str, float] = {}
        for field in numeric_fields:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise UnsupportedBundleError(f"Calibration record {key!r} has invalid {field}")
            numeric[field] = float(value)
        if numeric["max_abs_gain_db"] <= 0 or abs(float(gain)) > numeric["max_abs_gain_db"] + 1e-9:
            raise UnsupportedBundleError(f"Calibration record {key!r} exceeds its safe gain bound")
        median_lufs = float(statistics.median(numeric_arrays["measured_lufs"]))
        max_sample_peak = max(numeric_arrays["measured_sample_peak_dbfs"])
        max_true_peak = max(numeric_arrays["measured_true_peak_dbtp"])
        summary = (
            ("median_measured_lufs", median_lufs),
            ("max_measured_sample_peak_dbfs", max_sample_peak),
            ("max_measured_true_peak_dbtp", max_true_peak),
        )
        if any(not math.isclose(numeric[field], value, rel_tol=0.0, abs_tol=1e-6) for field, value in summary):
            raise UnsupportedBundleError(f"Calibration record {key!r} has inconsistent measurement summaries")
        policy_gain = min(
            numeric["target_lufs"] - median_lufs,
            numeric["max_true_peak_dbtp"] - max_true_peak,
        )
        if not math.isclose(float(gain), policy_gain, rel_tol=0.0, abs_tol=1e-6):
            raise UnsupportedBundleError(f"Calibration record {key!r} does not follow its target/peak policy")
        predictions = (
            ("predicted_median_lufs", median_lufs + float(gain)),
            ("predicted_max_sample_peak_dbfs", max_sample_peak + float(gain)),
            ("predicted_max_true_peak_dbtp", max_true_peak + float(gain)),
        )
        if any(
            not math.isclose(numeric[field], value, rel_tol=0.0, abs_tol=1e-6)
            for field, value in predictions
        ):
            raise UnsupportedBundleError(f"Calibration record {key!r} has inconsistent predicted measurements")
        if numeric["predicted_max_true_peak_dbtp"] > numeric["max_true_peak_dbtp"] + 1e-6:
            raise UnsupportedBundleError(f"Calibration record {key!r} exceeds its true-peak ceiling")
        frozen = dict(record)
        frozen.update(numeric)
        frozen["gain_db"] = float(gain)
        frozen["sample_rates"] = tuple(sample_rates)
        frozen.update(numeric_arrays)
        frozen["sample_sha256s"] = tuple(sample_hashes)
        validated[key] = MappingProxyType(frozen)
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
    if config.mode == "off" and config.gain_db is not None:
        raise InvalidGenerationConfigError(
            "gain_db requires voice-level mode 'calibrated'"
        )
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
        scale_value = 10.0 ** (gain_db / 20.0)
    except OverflowError as exc:
        raise InvalidGenerationConfigError(
            "voice-level gain produces an unrepresentable scale"
        ) from exc
    float32 = np.finfo(np.float32)
    if (
        not math.isfinite(scale_value)
        or scale_value > float(float32.max)
        or scale_value < float(float32.smallest_subnormal)
    ):
        raise InvalidGenerationConfigError("voice-level gain produces an unrepresentable scale")
    scale = np.float32(scale_value)
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
