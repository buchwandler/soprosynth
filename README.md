# SoproSynth

SoproSynth provides a stable Python request/result and synchronous streaming API around ONNXVoice's built-in Sopro V2 Turbo implementation. It owns request validation, original-rate reference forwarding, text segmentation, native/raw output conditioning through AudioSig, optional application-level voice calibration, chunking, result metadata, and WAV output. ONNXVoice owns model assets, tokenization, semantic carry, ONNX graph/session execution, and neural inference.

SoproSynth executes Sopro through ONNXVoice and Python ONNX Runtime. There is no native Sopro inference backend. It has no runtime dependency on the native `sopro` Python package and does not copy Sopro neural modules or model weights. Python 3.10 or newer is required.

## Support boundary

| Capability           | Current support                                                                     |
| -------------------- | ----------------------------------------------------------------------------------- |
| Model                | `sopro:v2-turbo`                                                                    |
| Profile              | `wasm-uint8` (portable, quantized)                                                  |
| Provider             | CPU (`CPUExecutionProvider`)                                                        |
| Languages            | `en`, `pt`, `fr`, `de`, and `auto`                                                  |
| Input                | Text request and prepared reference voice                                           |
| Long text            | Upstream-compatible splitting; ONNXVoice owns semantic carry                        |
| Output               | Native-compatible conditioning by default; raw mode optional                        |
| Audio                | Mono float32 PCM at 24 kHz; `save_wav()` writes PCM16 WAV                           |
| Prepared voice reuse | Supported within the same open runtime                                              |
| Request measurement  | Supported through ONNXVoice text preparation                                        |
| Streaming            | Synchronous API implemented; capability remains gated on released-wheel integration |

Unsupported profiles and providers fail explicitly. `SoproOutputConfig(mode="native")` selects only Sopro-compatible DSP conditioning; it does not select native PyTorch inference or imply a model-quality tier. Language detection, pronunciation overrides, linguistic-token processing, and word alignment are not supported. Streaming is synchronous, but the capability flag remains false until the opt-in released-wheel real-model integration gate passes.

## Install

The base package keeps inference dependencies optional. Install the CPU extra for synthesis:

```bash
python -m pip install "soprosynth[cpu]"
```

For development:

```bash
python -m pip install -e ".[cpu,dev]"
```

AudioSig `>=0.1.8,<0.2` is a required runtime dependency for reference-rate handling and output conditioning. ONNXVoice remains optional until inference is requested; install the `cpu` extra to synthesize.

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

The default model is `sopro:v2-turbo`; defaults are profile `wasm-uint8`, CPU execution, and native-compatible output conditioning. The `native` output mode is DSP policy, not native PyTorch inference or a quality-tier claim. See [`examples/quickstart.py`](examples/quickstart.py).

## Structured API

```python
from soprosynth import GenerationConfig, SoproOutputConfig, SoproRuntime, SynthesisRequest

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
        output=SoproOutputConfig(mode="native"),
    )
```

`SynthesisRequest` requires a stable ID and non-empty text. SoproSynth applies upstream-compatible character/sentence segmentation; ONNXVoice's `SoproRun` owns semantic-token carry between segments. Requests use one ONNXVoice run, which is closed on success, failure, or early streaming cancellation.

`runtime.prepare_voice(...)` forwards clean mono float32 reference samples at their original sample rate by default; ONNXVoice owns Sopro reference preprocessing. Explicit `resampler="linear"` or `resampler="audiosig"` selects a legacy SoproSynth-side conversion override. Reuse a `PreparedVoice` only with its creating runtime; its state is not portable across runtimes, model revisions, or processes. `runtime.measure_request(request)` uses ONNXVoice text preparation to report the token count.

`request_api_contract().supports_streaming` and `runtime.capabilities.supports_streaming` are both `False` until the opt-in released-wheel real-model integration gate passes. The synchronous `stream()` and `stream_text()` APIs are implemented, but streaming is not yet advertised as a supported capability. The request contract reports automatic text boundaries (`caller_owns_text_boundaries=False`) and support for the native/raw output policy.

## Streaming and prewarm

Streaming is synchronous and yields validated `SynthesisChunk` values. The generator provides backpressure; close it explicitly if abandoning iteration early. Native/raw output conditioning and optional static voice-level gain are applied incrementally. Prewarm state remains inside ONNXVoice; `PreparedVoice` metadata does not expose caches.

```python
from soprosynth import GenerationConfig, SoproRuntime

with SoproRuntime.from_pretrained() as runtime:
    voice = runtime.prepare_voice("reference.wav")
    config = GenerationConfig(seed=1234)
    runtime.prepare_streaming(voice, config=config)  # optional prewarm
    # audio_sink is supplied by the application's playback integration.

    for chunk in runtime.stream_text("A streamed request.", voice=voice, config=config):
        audio_sink.write(chunk.audio, sample_rate=chunk.sample_rate)
        if chunk.is_final:
            print(chunk.metadata["chunk_count"], chunk.metadata["total_stream_ms"])
```

The API is implemented, but both streaming capability flags remain false until the opt-in integration gate passes with a released ONNXVoice wheel and compatible licensed model/reference assets. The real-model gate is enabled with `SOPROSYNTH_RUN_REAL_ONNX_TESTS=1`; it rejects editable ONNXVoice checkouts.

## CLI

````bash
soprosynth speak "Hello from SoproSynth." \
  --ref reference.wav \
  --out speech.wav \
  --model sopro:v2-turbo \
  --profile wasm-uint8 \
  --provider cpu \
  --output-mode native

soprosynth speak "Hello from SoproSynth." --ref reference.wav --output-mode raw
soprosynth discover [--offline] [--refresh] [--cache-dir PATH]
soprosynth info
soprosynth inspect EXPORT_DIRECTORY --profile wasm-uint8

`speak` accepts language, generation controls, `--output-mode native|raw`, and optional voice-level calibration controls. `native` is the default. `discover` queries ONNXVoice catalog metadata. `inspect` is a developer diagnostic for a compatible local export; it opens a runtime session but does not synthesize.

## Native/raw output policy

`SoproOutputConfig()` defaults to `mode="native"`. It derives gain from ONNXVoice's prepared reference-level metadata (target `-23 dB`; no integrated-LUFS estimate). Trimming keeps 80 ms of first-segment lead, uses 300 ms of later lead with a 100 ms skip, and keeps 300 ms of trailing audio per segment. Joins use 10 ms edge fades, followed by the `0.9` soft limiter and an 80 ms final fade. Native mode requires finite `PreparedVoice.metadata["reference_level_db"]`; missing metadata raises a clear capability error.

`SoproOutputConfig(mode="raw")` concatenates unconditioned ONNX vocoder segment waveforms without native gain, trimming, fades, or limiting. Long requests use Sopro-compatible segmentation; ONNXVoice owns semantic-token carry. Optional voice-level calibration can still be applied afterward. Output-policy measurements are reported under `metadata["sopro_output"]`.

Optional `VoiceLevelConfig(mode="calibrated")` is a separate layer applied after the selected output mode. It defaults off, is reported under `metadata["voice_level_application"]`, and is not part of Sopro's reference-relative gain policy.

## Voice-level calibration

Voice-level processing is opt-in through `VoiceLevelConfig`. The packaged calibration catalog is empty because no measured corpus was supplied. In calibrated mode, a gain is used only for an exact ONNX model, immutable source revision, and profile. If there is no exact record, the result reports `source="missing_calibration"` and leaves the audio unchanged. A caller may instead provide an explicit static `gain_db` override; this applies linear gain without measuring or clipping the in-memory result.

Build a reviewable calibration record from licensed PCM WAVs with the declared AudioSig dependency and `soprosynth-build-calibration`. The command requires `--model-id`, immutable `--revision`, `--profile`, corpus/license/provenance metadata, explicit loudness/true-peak policy, and `--i-have-rights`. It writes an external reviewable catalog and does not modify the packaged catalog.

AudioSig is a required core dependency and supplies the Sopro-compatible gain, boundary, fade, join, and limiter operations. The default reference path does not pre-resample; explicitly select legacy model-rate conversion with `runtime.prepare_voice(reference, resampler="linear")` or `resampler="audiosig"`.

## Reference-rate and model-quality boundary

Only the portable `wasm-uint8` profile is currently accepted by the Python runtime; it is quantized and is not native-PyTorch-equivalent. `cpu-fp32` remains unsupported until upstream exports, pinned graph artifacts, and direct fp32 oracle outputs are available. A licensed native-versus-ONNX listening corpus is also an external release dependency. Unit tests and portable graph goldens do not satisfy those quality gates; SoproSynth makes no native-quality or full-parity claim.

## Discovery and runtime identity

`discover_bundles()` queries ONNXVoice's `sopro` catalog and does not open a synthesis model. Catalog access requires ONNXVoice, provided by the `cpu` extra. `runtime_identity()` returns SoproSynth/ONNXVoice version information and exact catalog/bundle revisions where available; missing revisions remain unknown rather than being guessed.

## Immutable objects and JSON

Public discovery metadata is intentionally immutable, including nested mappings. Avoid using `dataclasses.asdict()` as a generic transport serializer: immutable mappings such as `MappingProxyType` are not compatible with its deep-copy behavior. Use SoproSynth's supported conversion helpers instead:

```python
import json
from soprosynth import discover_bundles, to_json_value

payload = to_json_value(discover_bundles(offline=True))
print(json.dumps(payload, indent=2, allow_nan=False))
```

For one discovery result, `discovered_bundle_to_dict(bundle)` returns the same detached JSON-compatible field representation. Mappings become ordinary dictionaries, tuples become lists, and `Path` values become strings. Unsupported values and non-finite floats raise errors rather than being silently stringified.

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
````

SoproSynth source code is licensed under Apache-2.0. Pretrained model assets and reference recordings are separate resources governed by their own licenses and usage terms.
