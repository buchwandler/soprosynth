# SoproSynth

SoproSynth provides a stable Python request/result API around ONNXVoice's built-in Sopro V2 Turbo implementation. SoproSynth owns request validation, reference-voice lifecycle, loudness calibration policy, result metadata, and WAV output. ONNXVoice owns Sopro model assets, tokenization, graph/session execution, and inference.

SoproSynth supports one synthesis path: ONNXVoice's Sopro adapter. It does not vendor model weights or implement ONNX inference itself. Python 3.10 or newer is required.

## Support boundary

| Capability           | v0.1.0 support                                            |
| -------------------- | --------------------------------------------------------- |
| Model                | `sopro:v2-turbo`                                          |
| Profile              | `wasm-uint8`                                              |
| Provider             | CPU (`CPUExecutionProvider`)                              |
| Languages            | `en`, `pt`, `fr`, `de`, and `auto`                        |
| Input                | One text request and a prepared reference voice           |
| Output               | Mono float32 PCM at 24 kHz; `save_wav()` writes PCM16 WAV |
| Prepared voice reuse | Supported within the same open runtime                    |
| Request measurement  | Supported through ONNXVoice text preparation              |
| Streaming            | Not supported                                             |

Unsupported profiles and providers fail explicitly. SoproSynth does not perform language detection, text segmentation, pronunciation overrides, linguistic-token processing, word alignment, or final mastering.

## Install

The base package keeps inference dependencies optional. Install the CPU extra for synthesis:

```bash
python -m pip install "soprosynth[cpu]"
```

For development:

```bash
python -m pip install -e ".[cpu,dev]"
```

Optional audio tools remain separate:

```bash
python -m pip install "soprosynth[audio]"
```

The audio extra provides opt-in anti-aliased reference resampling and offline loudness/peak measurement for calibration records. It is not required for synthesis.

## Quickstart

Provide a reference recording you are authorized to use. The first uncached run may acquire the ONNXVoice Sopro bundle.

```python
from soprosynth import SoproRuntime

with SoproRuntime.from_pretrained() as runtime:
    result = runtime.synthesize_text(
        "Hello from SoproSynth.",
        id="hello",
        language="en",
        voice="reference.wav",
    )
    result.save_wav("hello.wav")
```

The default model is `sopro:v2-turbo`; defaults are profile `wasm-uint8` and CPU execution. See [`examples/quickstart.py`](examples/quickstart.py).

## Structured API

```python
from soprosynth import GenerationConfig, SoproRuntime, SynthesisRequest

with SoproRuntime.from_pretrained(
    model="sopro:v2-turbo",
    profile="wasm-uint8",
    providers="cpu",
) as runtime:
    voice = runtime.prepare_voice("reference.wav")
    request = SynthesisRequest(
        id="speech-001",
        text="Hallo von SoproSynth.",
        language="de",
    )
    result = runtime.synthesize(
        request,
        voice=voice,
        config=GenerationConfig(temperature=0.8, top_p=0.9, top_k=40),
    )
```

`SynthesisRequest` requires a stable ID and a non-empty text value. Supported language IDs are `en`, `pt`, `fr`, `de`, and `auto` (or `None` where allowed). The caller owns text preparation and segment boundaries.

`runtime.prepare_voice(...)` normalizes and prepares a reference recording for the open runtime. Reuse the resulting `PreparedVoice` only with that same runtime; prepared voice state is not portable across runtimes, model revisions, or processes. `runtime.measure_request(request)` uses ONNXVoice text preparation to report the token count.

`request_api_contract().supports_streaming` and `runtime.capabilities.supports_streaming` are both `False`. Runtime capabilities report request measurement and prepared-voice reuse as supported.

## CLI

```bash
soprosynth speak "Hello from SoproSynth." \
  --ref reference.wav \
  --out speech.wav \
  --model sopro:v2-turbo \
  --profile wasm-uint8 \
  --provider cpu

soprosynth discover [--offline] [--refresh] [--cache-dir PATH]
soprosynth info
soprosynth inspect EXPORT_DIRECTORY --profile wasm-uint8
```

`speak` accepts language, generation controls, and optional voice-level calibration controls. `discover` queries ONNXVoice catalog metadata. `inspect` is a developer diagnostic for a compatible local export; it opens a runtime session but does not synthesize.

## Voice-level calibration

Voice-level processing is opt-in through `VoiceLevelConfig`. The packaged calibration catalog is empty because no measured corpus was supplied. In calibrated mode, a gain is used only for an exact ONNX model, immutable source revision, and profile. If there is no exact record, the result reports `source="missing_calibration"` and leaves the audio unchanged. A caller may instead provide an explicit static `gain_db` override; this applies linear gain without measuring or clipping the in-memory result.

Build a reviewable calibration record from licensed PCM WAVs with the optional `soprosynth[audio]` extra and `soprosynth-build-calibration`. The command requires `--model-id`, immutable `--revision`, `--profile`, corpus/license/provenance metadata, explicit loudness/true-peak policy, and `--i-have-rights`. It writes an external reviewable catalog and does not modify the packaged catalog.

Reference-rate conversion uses dependency-free linear interpolation by default. To opt into anti-aliased resampling, install the audio extra and call `runtime.prepare_voice(reference, resampler="audiosig")`.

## Discovery and runtime identity

`discover_bundles()` queries ONNXVoice's `sopro` catalog and does not open a synthesis model. Catalog access requires ONNXVoice, provided by the `cpu` extra. `runtime_identity()` returns SoproSynth/ONNXVoice version information and exact catalog/bundle revisions where available; missing revisions remain unknown rather than being guessed.

## Examples

Runnable scripts and their prerequisites are documented in [`examples/README.md`](examples/README.md):

- [`examples/quickstart.py`](examples/quickstart.py)
- [`examples/strict_request.py`](examples/strict_request.py)
- [`examples/prepared_voice.py`](examples/prepared_voice.py)
- [`examples/multilingual.py`](examples/multilingual.py)
- [`examples/calibrated_voice_level.py`](examples/calibrated_voice_level.py)
- [`examples/discovery.py`](examples/discovery.py)
- [`examples/onnx_inspect.py`](examples/onnx_inspect.py)

Do not run synthesis examples unless you have model access and the rights to use the reference recording. SoproSynth does not grant rights to user-provided recordings or generated speech.

## Development

```bash
python -m pip install -e ".[cpu,dev]"
python -m pytest -q
ruff check .
python -m build
```

SoproSynth source code is licensed under Apache-2.0. Pretrained model assets and reference recordings are separate resources governed by their own licenses and usage terms.
