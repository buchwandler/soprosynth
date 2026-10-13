"""Offline, licensed-corpus calibration catalog builder; audiosig loads on demand."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

from .audio import load_pcm_wav
from .calibration import calibration_key, load_calibration_catalog
from .errors import OptionalDependencyError

_MEASUREMENT_METHOD = "audiosig gated BS.1770-style LUFS (400 ms blocks/100 ms hop), sample peak, and 4x true peak; median LUFS and maximum peaks"
_PRODUCTION_CATALOG = Path(__file__).resolve().parent / "data" / "voice_level_calibration.json"


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite real number")
    return number


def build_calibration_record(
    *,
    identity: dict[str, object],
    corpus: str,
    license_id: str,
    provenance: str,
    target_lufs: float,
    max_true_peak_dbtp: float,
    max_abs_gain_db: float,
    measured_lufs: Sequence[float],
    measured_sample_peaks_dbfs: Sequence[float],
    measured_true_peaks_dbtp: Sequence[float],
    sample_rates: Sequence[int],
    durations_seconds: Sequence[float],
    sample_sha256s: Sequence[str],
    measurement_tool_version: str,
) -> tuple[str, dict[str, object]]:
    """Solve a bounded uniform gain and return a provenance-rich catalog record."""
    key = calibration_key(identity)
    if key is None:
        raise ValueError(
            "identity must include a supported backend, model, exact revision, and ONNX profile"
        )
    revision = identity.get("source_revision") or identity.get("revision")
    assert isinstance(revision, str)
    if revision != revision.strip():
        raise ValueError("revision must not contain leading or trailing whitespace")
    for label, value in (
        ("corpus", corpus),
        ("license", license_id),
        ("provenance", provenance),
        ("measurement_tool_version", measurement_tool_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")

    target = _finite_number(target_lufs, "target_lufs")
    peak_ceiling = _finite_number(max_true_peak_dbtp, "max_true_peak_dbtp")
    gain_bound = _finite_number(max_abs_gain_db, "max_abs_gain_db")
    if gain_bound <= 0:
        raise ValueError("max_abs_gain_db must be positive")
    measurements = len(measured_lufs)
    sample_arrays = (
        measured_sample_peaks_dbfs,
        measured_true_peaks_dbtp,
        sample_rates,
        durations_seconds,
        sample_sha256s,
    )
    if measurements == 0 or any(len(values) != measurements for values in sample_arrays):
        raise ValueError("each calibration sample must have complete measurements and provenance")
    hashes = list(sample_sha256s)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("sample_sha256s must contain lowercase SHA-256 digests")
    if len(set(hashes)) != len(hashes):
        raise ValueError("calibration samples must have distinct SHA-256 digests")

    loudness = [_finite_number(value, "measured_lufs") for value in measured_lufs]
    sample_peaks = [
        _finite_number(value, "measured_sample_peak_dbfs") for value in measured_sample_peaks_dbfs
    ]
    true_peaks = [
        _finite_number(value, "measured_true_peak_dbtp") for value in measured_true_peaks_dbtp
    ]
    rates: list[int] = []
    for rate in sample_rates:
        if type(rate) is not int or rate <= 0:
            raise ValueError("sample_rates must contain positive integers")
        rates.append(rate)
    durations = [_finite_number(value, "duration_seconds") for value in durations_seconds]
    if any(duration <= 0 for duration in durations):
        raise ValueError("duration_seconds must be positive")

    median_lufs = float(statistics.median(loudness))
    worst_sample_peak = max(sample_peaks)
    worst_true_peak = max(true_peaks)
    desired_gain = target - median_lufs
    peak_limited_gain = peak_ceiling - worst_true_peak
    if not math.isfinite(desired_gain) or not math.isfinite(peak_limited_gain):
        raise ValueError("target/peak policy produces an unrepresentable gain")
    gain_db = min(desired_gain, peak_limited_gain)
    if abs(gain_db) > gain_bound:
        raise ValueError(
            f"safe gain {gain_db:.3f} dB exceeds the configured +/-{gain_bound:.3f} dB bound"
        )
    try:
        scale = 10.0 ** (gain_db / 20.0)
    except OverflowError as exc:
        raise ValueError("solved gain has an unrepresentable float32 scale") from exc
    if (
        not math.isfinite(scale)
        or scale > float.fromhex("0x1.fffffep+127")
        or scale < float.fromhex("0x1.0p-149")
    ):
        raise ValueError("solved gain has an unrepresentable float32 scale")

    predicted_median_lufs = median_lufs + gain_db
    predicted_worst_sample_peak = worst_sample_peak + gain_db
    predicted_worst_true_peak = worst_true_peak + gain_db
    if not all(
        math.isfinite(value)
        for value in (predicted_median_lufs, predicted_worst_sample_peak, predicted_worst_true_peak)
    ):
        raise ValueError("predicted calibration measurements are not finite")
    if predicted_worst_true_peak > peak_ceiling + 1e-9:
        raise ValueError("solved gain violates the true-peak ceiling")
    record: dict[str, object] = {
        "gain_db": gain_db,
        "corpus": corpus.strip(),
        "license": license_id.strip(),
        "provenance": provenance.strip(),
        "measurements": measurements,
        "revision": revision,
        "target_lufs": target,
        "max_true_peak_dbtp": peak_ceiling,
        "median_measured_lufs": median_lufs,
        "max_measured_sample_peak_dbfs": worst_sample_peak,
        "max_measured_true_peak_dbtp": worst_true_peak,
        "predicted_median_lufs": predicted_median_lufs,
        "predicted_max_sample_peak_dbfs": predicted_worst_sample_peak,
        "predicted_max_true_peak_dbtp": predicted_worst_true_peak,
        "max_abs_gain_db": gain_bound,
        "measurement_tool": "audiosig",
        "measurement_tool_version": measurement_tool_version,
        "measurement_method": _MEASUREMENT_METHOD,
        "sample_rates": rates,
        "durations_seconds": durations,
        "measured_lufs": loudness,
        "measured_sample_peak_dbfs": sample_peaks,
        "measured_true_peak_dbtp": true_peaks,
        "sample_sha256s": hashes,
    }
    load_calibration_catalog({"schema": 2, "records": {key: record}})
    return key, record


def measure_audio_files(
    paths: Sequence[str | Path],
) -> tuple[list[float], list[float], list[float], list[int], list[float], list[str], str]:
    """Measure licensed PCM WAVs with audiosig, importing the optional extra lazily."""
    if len(paths) == 0:
        raise ValueError("at least one licensed PCM WAV is required")
    try:
        import audiosig
    except ImportError as exc:
        raise OptionalDependencyError(
            "Measured calibration requires audiosig; install 'soprosynth[audio]'"
        ) from exc

    loudnesses: list[float] = []
    sample_peaks: list[float] = []
    true_peaks: list[float] = []
    sample_rates: list[int] = []
    durations: list[float] = []
    hashes: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        audio, sample_rate = load_pcm_wav(path)
        analysis = audiosig.analyze_loudness(
            audio,
            sample_rate=sample_rate,
            integrated=True,
            sample_peak=True,
            true_peak=True,
            true_peak_oversample=4,
        )
        if (
            analysis.integrated_lufs is None
            or analysis.sample_peak_dbfs is None
            or analysis.true_peak_dbtp is None
        ):
            raise ValueError(f"audiosig returned incomplete measurements for {path}")
        loudnesses.append(
            _finite_number(analysis.integrated_lufs, f"integrated loudness for {path}")
        )
        sample_peaks.append(_finite_number(analysis.sample_peak_dbfs, f"sample peak for {path}"))
        true_peaks.append(_finite_number(analysis.true_peak_dbtp, f"true peak for {path}"))
        sample_rates.append(sample_rate)
        durations.append(audio.size / sample_rate)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return (
        loudnesses,
        sample_peaks,
        true_peaks,
        sample_rates,
        durations,
        hashes,
        str(audiosig.__version__),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one offline voice-level calibration record from licensed PCM WAV measurements."
    )
    parser.add_argument("--backend", required=True, choices=("native", "onnx"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="exact immutable model revision")
    parser.add_argument("--profile", help="required for ONNX calibration identity")
    parser.add_argument(
        "--audio", action="append", required=True, help="licensed output WAV; repeat per sample"
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--license", dest="license_id", required=True)
    parser.add_argument(
        "--provenance", required=True, help="corpus revision, source URL, or run record"
    )
    parser.add_argument("--target-lufs", type=float, required=True)
    parser.add_argument("--max-true-peak-dbtp", type=float, required=True)
    parser.add_argument("--max-abs-gain-db", type=float, default=24.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    parser.add_argument(
        "--i-have-rights",
        action="store_true",
        required=True,
        help="confirm that the audio corpus permits this offline measurement and provenance record",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    identity: dict[str, object] = {
        "backend": args.backend,
        "model_id": args.model_id,
        "source_revision": args.revision,
    }
    if args.profile is not None:
        identity["profile"] = args.profile
    key = calibration_key(identity)
    if key is None:
        parser.error("the identity must include an exact supported revision and an ONNX profile")

    output = args.output.expanduser().resolve()
    if output == _PRODUCTION_CATALOG.resolve():
        parser.error(
            "write a reviewable calibration file; the packaged production catalog is not an output target"
        )
    if output.exists() and not args.force:
        parser.error(f"output already exists (pass --force to replace it): {output}")

    try:
        (
            measured_lufs,
            measured_sample_peaks,
            measured_true_peaks,
            sample_rates,
            durations_seconds,
            sample_hashes,
            tool_version,
        ) = measure_audio_files(args.audio)
        record_key, record = build_calibration_record(
            identity=identity,
            corpus=args.corpus,
            license_id=args.license_id,
            provenance=args.provenance,
            target_lufs=args.target_lufs,
            max_true_peak_dbtp=args.max_true_peak_dbtp,
            max_abs_gain_db=args.max_abs_gain_db,
            measured_lufs=measured_lufs,
            measured_sample_peaks_dbfs=measured_sample_peaks,
            measured_true_peaks_dbtp=measured_true_peaks,
            sample_rates=sample_rates,
            durations_seconds=durations_seconds,
            sample_sha256s=sample_hashes,
            measurement_tool_version=tool_version,
        )
        catalog = {"schema": 2, "records": {record_key: record}}
        load_calibration_catalog(catalog)
    except (OSError, ValueError, OptionalDependencyError) as exc:
        parser.error(str(exc))

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(catalog, indent=2, sort_keys=True))
            stream.write("\n")
        temporary.replace(output)
    except OSError as exc:
        parser.error(f"could not write calibration output {output}: {exc}")
    print(f"Wrote {len(measured_lufs)} licensed measurements for {record_key} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
