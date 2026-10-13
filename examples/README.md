# Example prerequisites and limits

Examples are runnable scripts, not offline tests. Do not run a synthesis example unless you have the required runtime, model assets, and permission to use the reference recording. Output files are written relative to the current working directory.

## Native examples

Install `soprosynth[native]`, provide an authorized `reference.wav`, and let upstream Sopro access its model assets. The first uncached model open may download weights. These examples use the native backend explicitly:

- `quickstart.py` — one request; it does not detect language.
- `strict_request.py` — the structured API request/result path.
- `prepared_voice.py` — reuses a prepared reference only in the same open runtime; prepared state is not portable.
- `multilingual.py` — submits explicit `en`, `pt`, `fr`, and `de` labels; it does not promise language detection or perceptual quality.
- `streaming.py` — native-only. It collects all chunks in memory for the demo and makes no real-time guarantee. ONNX streaming is unsupported.
- `calibrated_voice_level.py` — the packaged calibration catalog is empty, so this example's default calibrated request is unchanged (`missing_calibration`). It does not demonstrate measured loudness normalization.

## Dependency-light example

`discovery.py` works with the base installation. It queries the API contract, static native bundle metadata, and runtime identity without loading a model or importing optional runtimes. The v1 request contract is a producer-level union; inspect an opened runtime's `capabilities` for backend-specific support.

## Experimental ONNX example

`onnx_inspect.py` requires `soprosynth[cpu]` and a compatible local ONNXVoice export. It opens an ONNXVoice runtime/session to print diagnostics but does not synthesize. It is not a metadata-only/no-model check and does not establish inference quality or parity. The tested Sopro boundary is `wasm-uint8` with CPU execution; other profiles/providers and ONNX streaming remain unsupported.
