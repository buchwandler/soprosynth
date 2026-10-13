"""Experimental developer-only local ONNX bundle inspection; no synthesis is run."""

from soprosynth import SoproRuntime

# Requires an ONNXVoice-compatible local export and the optional ONNX dependencies.
# This validates/open metadata only; it does not establish model inference parity.
with SoproRuntime.from_onnx_local("./onnx-model", profile="wasm-uint8") as runtime:
    print(runtime.diagnostics)
