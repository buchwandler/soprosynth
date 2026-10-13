"""Inspect a local ONNXVoice session; this developer tool does not synthesize.

Requires ``soprosynth[cpu]`` and a compatible local export. Opening the runtime
loads its session and prints diagnostics; it does not establish inference
quality or parity.
"""

from soprosynth import SoproRuntime

with SoproRuntime.from_onnx_local("./onnx-model", profile="wasm-uint8") as runtime:
    print(runtime.diagnostics)
