"""SoproSynth command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__, discover_bundles, request_api_contract, runtime_identity
from .api import SoproRuntime
from .config import GenerationConfig, SoproOutputConfig, VoiceLevelConfig
from .serialization import to_json_value


def _print_json(value: Any) -> None:
    print(
        json.dumps(
            to_json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _contract_summary() -> dict[str, Any]:
    contract = request_api_contract()
    return {
        "version": contract.version,
        "request_type": contract.request_type.__name__,
        "result_type": contract.result_type.__name__,
        "synthesis_method": contract.synthesis_method,
        "measurement_method": contract.measurement_method,
        "supports_reference_voice": contract.supports_reference_voice,
        "supports_voice_level": contract.supports_voice_level,
        "supports_sopro_output_policy": contract.supports_sopro_output_policy,
        "caller_owns_text_boundaries": contract.caller_owns_text_boundaries,
        "supports_request_measurement": contract.supports_request_measurement,
        "supports_streaming": contract.supports_streaming,
        "supports_word_timings": contract.supports_word_timings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soprosynth",
        description="SoproSynth speech synthesis using ONNXVoice's Sopro runtime",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    speak = subparsers.add_parser("speak", help="synthesize speech")
    speak.add_argument("text", help="text to synthesize")
    speak.add_argument("--ref", required=True, help="reference voice WAV path")
    speak.add_argument("--out", default="out.wav", help="output mono PCM16 WAV path")
    speak.add_argument("--language", choices=("en", "pt", "fr", "de", "auto"))
    speak.add_argument("--model", default="sopro:v2-turbo")
    speak.add_argument("--profile", default="wasm-uint8")
    speak.add_argument("--provider", choices=("cpu",), default="cpu")
    speak.add_argument(
        "--output-mode",
        choices=("native", "raw"),
        default="native",
        help="Sopro-compatible output conditioning or raw ONNX waveform",
    )
    speak.add_argument("--temperature", type=float)
    speak.add_argument("--top-p", type=float)
    speak.add_argument("--top-k", type=int)
    speak.add_argument("--max-seconds", type=float)
    speak.add_argument("--min-seconds", type=float)
    speak.add_argument("--voice-level", choices=("off", "calibrated"), default="off")
    speak.add_argument("--gain-db", type=float, help="explicit static gain override")

    subparsers.add_parser(
        "info", help="show API contract and runtime provenance without loading a model"
    )

    discover = subparsers.add_parser("discover", help="list ONNXVoice Sopro bundle metadata")
    discover.add_argument(
        "--offline", action="store_true", help="use catalog data available offline"
    )
    discover.add_argument(
        "--refresh", action="store_true", help="refresh ONNXVoice catalog metadata"
    )
    discover.add_argument("--cache-dir", type=Path)

    inspect = subparsers.add_parser(
        "inspect",
        help="open a local ONNXVoice session and print diagnostics; does not synthesize",
    )
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("--profile", default="wasm-uint8")
    return parser


def _speak(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.gain_db is not None and args.voice_level != "calibrated":
        parser.error("--gain-db requires --voice-level calibrated")
    config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_seconds=args.max_seconds,
        min_seconds=args.min_seconds,
    )
    voice_level = VoiceLevelConfig(mode=args.voice_level, gain_db=args.gain_db)
    output_config = SoproOutputConfig(mode=args.output_mode)
    with SoproRuntime.from_pretrained(
        model=args.model,
        profile=args.profile,
        providers=args.provider,
    ) as runtime:
        result = runtime.synthesize_text(
            args.text,
            id="cli-speak",
            voice=args.ref,
            language=args.language,
            config=config,
            voice_level=voice_level,
            output=output_config,
        )
        output_path = result.save_wav(args.out)
        print(output_path)
        _print_json(
            {
                "sopro_output": result.metadata.get("sopro_output"),
                "voice_level_application": result.metadata.get("voice_level_application"),
            }
        )
    return 0


def _discover(args: argparse.Namespace) -> int:
    bundles = discover_bundles(
        offline=args.offline,
        refresh=args.refresh,
        cache_dir=args.cache_dir,
    )
    _print_json(bundles)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    print("Developer inspection only; this command does not run synthesis.")
    with SoproRuntime.from_onnx_local(args.directory, profile=args.profile) as runtime:
        _print_json(runtime.diagnostics)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "speak":
        return _speak(args, parser)
    if args.command == "info":
        _print_json(
            {"version": __version__, "api": _contract_summary(), "runtime": runtime_identity()}
        )
        return 0
    if args.command == "discover":
        return _discover(args)
    if args.command == "inspect":
        return _inspect(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
