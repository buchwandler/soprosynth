from __future__ import annotations
import argparse
from . import __version__
from .api import SoproRuntime, SynthesisRequest


def main() -> None:
    p = argparse.ArgumentParser(
        prog="soprosynth", description="SoproSynth API and ONNX graph inspector"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    ins = sub.add_parser("inspect", help="inspect local exported graph names (loads no sessions)")
    ins.add_argument("directory")
    ins.add_argument("--profile", default="wasm-uint8")
    sp = sub.add_parser("speak", help="synthesize using upstream native Sopro backend")
    sp.add_argument("text")
    sp.add_argument("--ref", required=True)
    sp.add_argument("--out", default="out.wav")
    sp.add_argument("--language", default=None, choices=["en", "pt", "fr", "de", "auto"])
    args = p.parse_args()
    if args.command == "inspect":
        with SoproRuntime.from_onnx_local(args.directory, profile=args.profile) as runtime:
            print("ONNX graphs:")
            for graph in runtime.onnx_adapter.graphs:
                print("  " + graph)
    else:
        with SoproRuntime.from_pretrained() as runtime:
            out = runtime.synthesize(
                SynthesisRequest(args.text, language=args.language), voice=args.ref
            )
            print(out.save_wav(args.out))


if __name__ == "__main__":
    main()
