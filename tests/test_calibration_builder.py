import json
import wave

import numpy as np
import pytest

from soprosynth import UnsupportedBundleError
from soprosynth.calibration import load_calibration_catalog
from soprosynth.calibration_builder import build_calibration_record, main, measure_audio_files
from soprosynth.errors import OptionalDependencyError


def _identity():
    return {
        "backend": "onnx",
        "model_id": "sopro:v2-turbo",
        "source_revision": "a" * 40,
        "profile": "wasm-uint8",
    }


def _record(**overrides):
    values = {
        "identity": _identity(),
        "corpus": "licensed-fixture-corpus",
        "license_id": "CC0-1.0",
        "provenance": "fixture release fixture-1",
        "target_lufs": -16.0,
        "max_true_peak_dbtp": -1.0,
        "max_abs_gain_db": 12.0,
        "measured_lufs": [-24.0, -20.0, -18.0],
        "measured_sample_peaks_dbfs": [-9.0, -8.0, -7.0],
        "sample_rates": [24000, 24000, 48000],
        "durations_seconds": [1.0, 2.0, 1.5],
        "measured_true_peaks_dbtp": [-3.0, -2.0, -1.5],
        "sample_sha256s": ["a" * 64, "b" * 64, "c" * 64],
        "measurement_tool_version": "0.1.7",
    }
    values.update(overrides)
    return build_calibration_record(**values)


def test_calibration_solver_targets_median_and_honors_true_peak_ceiling():
    key, record = _record()

    assert key == f"onnx:sopro:v2-turbo:{'a' * 40}:wasm-uint8"
    assert record["gain_db"] == pytest.approx(0.5)
    assert record["median_measured_lufs"] == pytest.approx(-20.0)
    assert record["predicted_median_lufs"] == pytest.approx(-19.5)
    assert record["predicted_max_true_peak_dbtp"] == pytest.approx(-1.0)
    catalog = load_calibration_catalog({"schema": 2, "records": {key: record}})
    assert catalog[key]["measurement_tool"] == "audiosig"
    assert catalog[key]["sample_sha256s"] == tuple(record["sample_sha256s"])


def test_calibration_solver_rejects_gain_outside_safe_bound():
    with pytest.raises(ValueError, match="safe gain"):
        _record(
            target_lufs=-12.0,
            measured_lufs=[-40.0, -40.0, -40.0],
            measured_true_peaks_dbtp=[-35.0, -35.0, -35.0],
        )


def test_schema_two_requires_consistent_measurement_provenance():
    key, record = _record()
    missing_license = dict(record)
    del missing_license["license"]
    with pytest.raises(UnsupportedBundleError, match="unsupported schema"):
        load_calibration_catalog({"schema": 2, "records": {key: missing_license}})

    bad_peak_policy = dict(record)
    bad_peak_policy["predicted_max_true_peak_dbtp"] = -0.5
    with pytest.raises(UnsupportedBundleError, match="inconsistent predicted"):
        load_calibration_catalog({"schema": 2, "records": {key: bad_peak_policy}})


def test_audiosig_measurements_are_deterministic_for_scaled_fixture_wavs(tmp_path):
    pytest.importorskip("audiosig")
    rate = 48000
    times = np.arange(rate, dtype=np.float64) / rate
    base = np.sin(2 * np.pi * 1000 * times)
    paths = [tmp_path / "full.wav", tmp_path / "half.wav"]
    for path, scale in zip(paths, (0.5, 0.25), strict=True):
        samples = (base * scale).astype(np.float32)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
            output.writeframes(pcm.tobytes())

    (
        loudness,
        sample_peaks,
        true_peaks,
        sample_rates,
        durations,
        hashes,
        tool_version,
    ) = measure_audio_files(paths)

    assert len(loudness) == len(sample_peaks) == len(true_peaks) == len(hashes) == 2
    assert loudness[0] - loudness[1] == pytest.approx(20 * np.log10(2), abs=0.1)
    assert sample_peaks[0] - sample_peaks[1] == pytest.approx(20 * np.log10(2), abs=0.1)
    assert true_peaks[0] - true_peaks[1] == pytest.approx(20 * np.log10(2), abs=0.1)
    assert sample_rates == [rate, rate]
    assert durations == [1.0, 1.0]
    assert len(set(hashes)) == 2
    assert tool_version
    assert load_calibration_catalog() == {}


def test_calibration_builder_import_error_is_actionable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "audiosig", None)
    with pytest.raises(OptionalDependencyError, match=r"declared AudioSig dependency"):
        measure_audio_files(["not-read-without-audiosig.wav"])


def test_calibration_cli_writes_reviewable_catalog_outside_production(tmp_path):
    pytest.importorskip("audiosig")
    rate = 48000
    times = np.arange(rate, dtype=np.float64) / rate
    samples = (0.25 * np.sin(2 * np.pi * 1000 * times)).astype(np.float32)
    source = tmp_path / "licensed-output.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((samples * 32767).astype("<i2").tobytes())
    destination = tmp_path / "reviewable-calibration.json"

    assert (
        main(
            [
                "--model-id",
                "sopro:v2-turbo",
                "--revision",
                "a" * 40,
                "--profile",
                "wasm-uint8",
                "--audio",
                str(source),
                "--corpus",
                "fixture corpus",
                "--license",
                "CC0-1.0",
                "--provenance",
                "fixture source and run id",
                "--target-lufs",
                "-16",
                "--max-true-peak-dbtp",
                "-1",
                "--output",
                str(destination),
                "--i-have-rights",
            ]
        )
        == 0
    )
    raw = json.loads(destination.read_text(encoding="utf-8"))
    assert raw["schema"] == 2
    assert len(raw["records"]) == 1
    load_calibration_catalog(raw)
    assert load_calibration_catalog() == {}
