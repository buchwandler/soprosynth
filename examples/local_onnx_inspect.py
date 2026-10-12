from soprosynth import SoproRuntime

# Requires a real upstream ONNX export under ./onnx-model/ and onnxruntime.
with SoproRuntime.from_onnx_local("./onnx-model", profile="wasm-uint8") as runtime:
    for name in runtime.onnx_adapter.graphs:
        print(name)
        # Print tensor shapes only if you want to initialize an ONNX Runtime session:
        # print(runtime.onnx_adapter.graph_specs(name))
