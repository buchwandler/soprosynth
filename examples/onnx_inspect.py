"""Experimental local ONNXVoice session inspection; no synthesis is run.

Requires ``soprosynth[cpu]`` and a compatible local export. Opening the runtime
loads its session; this does not establish inference or quality parity.
"""

from soprosynth import SoproRuntime

# Requires an ONNXVoice-compatible local export and the optional ONNX dependencies.
# This opens a local runtime/session; it does not run inference or establish parity.
with SoproRuntime.from_onnx_local("./onnx-model", profile="wasm-uint8") as runtime:
    print(runtime.diagnostics)
