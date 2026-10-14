# Example prerequisites and support boundary

These scripts are runnable examples, not offline tests. Install the ONNXVoice CPU runtime and provide a reference recording you are authorized to use before running synthesis examples. Output files are written relative to the current working directory.

```bash
python -m pip install "soprosynth[cpu]"
```

The first uncached runtime open may acquire the ONNXVoice Sopro bundle. The supported Python inference boundary is:

```text
model: sopro:v2-turbo
profile: wasm-uint8
provider: CPUExecutionProvider (cpu)
languages: en, pt, fr, de, auto
output: mono float32 PCM at 24 kHz
reference voice: required
output mode: native-compatible conditioning by default; raw is opt-in
reference samples: forwarded at their original rate by default
model-quality claim: `wasm-uint8` is quantized; no native-PyTorch parity claim
streaming: not supported
```

SoproSynth has no runtime dependency on the native `sopro` Python package. It segments long text and carries semantic tokens between segments when the installed ONNXVoice runtime exposes that contract. The `native` output name describes DSP conditioning only; it does not describe the model's quality tier.
Use only reference recordings whose rights and license permit your intended use. SoproSynth does not grant rights to recordings or generated speech.

## Synthesis examples

- `quickstart.py` — one request using the default ONNXVoice runtime settings.
- `strict_request.py` — the structured request/result API and generation configuration.
- `prepared_voice.py` — reuses prepared reference state within one runtime only.
- `multilingual.py` — explicit `en`, `pt`, `fr`, and `de` language labels; no language detection or quality guarantee.
- `calibrated_voice_level.py` — shows the unchanged `missing_calibration` fallback; the packaged calibration catalog is empty.

## Catalog and diagnostics examples

- `discovery.py` — queries ONNXVoice catalog metadata and prints API/runtime identity; it does not open a synthesis runtime. Install `soprosynth[cpu]` to provide ONNXVoice.
- `onnx_inspect.py` — developer diagnostics for a compatible local ONNXVoice export. It opens a session but does not synthesize or establish inference quality/parity.
