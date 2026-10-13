# SoproSynth

SoproSynth is a flat-layout Python library providing a backend-neutral request/result API for the upstream Sopro V2 Turbo text-to-speech system. The native upstream Sopro implementation is the default and reference backend. The stable public request API is version 1; optional ONNXVoice integration is a separate, explicitly selected backend.

SoproSynth accepts a single text request at a time. Callers retain ownership of text preparation, segment boundaries, composition, and final mastering. Python >=3.10 is required.

## Support status

| Area                              | Status                                                                 |
| --------------------------------- | ---------------------------------------------------------------------- |
| Native Sopro synthesis            | Primary backend; requires the `native` extra and upstream model assets |
| Strict request/result API         | Stable SoproSynth API v1                                               |
| Reference voice preparation/reuse | Supported within one open runtime                                      |
| Native streaming                  | Supported by the native backend only                                   |
| Static voice-level calibration    | Opt-in; no calibration records are bundled                             |
| ONNXVoice Sopro integration       | Optional and experimental; see [ONNX status](#onnx-status)             |
| ONNX streaming                    | Not supported                                                          |

Offline unit and boundary tests use fake backends. They do not establish quality or model-level parity for a real Sopro model.

## Install

Install the native backend and run the test suite:

```bash
python -m pip install 'soprosynth[native]'
python -m pip install 'soprosynth[dev]'
python -m pytest -q
```

The base package depends on NumPy only. `soprosynth[native]` installs upstream `sopro>=2.2.0,<3`; the native backend downloads or reads its own model weights through upstream Sopro. ONNX extras are optional and must not be installed unless you intend to experiment with that backend:

```bash
python -m pip install 'soprosynth[cpu]'
# The gpu extra forwards ONNXVoice's GPU dependencies; it does not imply
# that GPU execution is supported by the current Sopro parity boundary.
```

## Quickstart

Provide a reference WAV you are authorized to use. The first native run may download model assets.

```python
from soprosynth import SoproRuntime

with SoproRuntime.from_pretrained(backend="native") as runtime:
    result = runtime.synthesize_text(
        "Hello from SoproSynth.",
        id="hello",
        language="en",
        voice="reference.wav",
    )
    result.save_wav("hello.wav")
```

The native backend is selected by default, but examples pass `backend="native"` to make the choice explicit. Results contain normalized mono floating-point audio; `save_wav()` writes mono PCM16 WAV and clips only during PCM conversion.

## Strict request API

Use `SynthesisRequest` for downstream integrations that need stable IDs, explicit language, and validated generation policy:

```python
from soprosynth import GenerationConfig, SoproRuntime, SynthesisRequest

request = SynthesisRequest(
    id="segment-001",
    text="Guten Morgen.",
    language="de",
)
with SoproRuntime.from_pretrained(backend="native") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize(
        request,
        voice=voice,
        config=GenerationConfig(temperature=0.8, top_p=0.9),
    )
```

Supported language IDs are `en`, `pt`, `fr`, `de`, and `auto` (or `None` where allowed). SoproSynth does not run language detection. It does not implement text segmentation, pronunciation overrides, linguistic tokens, word timings, or final mastering. Request API capabilities are available via `request_api_contract()` without opening a model.

`GenerationConfig` provides validated controls such as `temperature`, `top_p`, `top_k`, `steps`, `max_seconds`, and `min_seconds`. A backend must reject unsupported controls rather than silently ignore them.

## Reference voice preparation and reuse

`runtime.prepare_voice(...)` validates and prepares a reference for the selected runtime. Reuse the returned `PreparedVoice` for multiple requests on that same runtime. Prepared voices are runtime-bound and are not portable to another runtime, backend, model revision, or process. The library retains a fingerprint for compatibility checks; it does not put raw reference samples in public identity metadata.

## Multilingual usage

Pass an explicit language ID for every request when the language is known. See [`examples/multilingual.py`](examples/multilingual.py) for `en`, `pt`, `fr`, and `de`. The API does not guess the language.

## Streaming

Native streaming is available through `runtime.stream(request, voice=prepared_voice)`. It yields `StreamingChunk` values. ONNX streaming is not supported and is never implied by the common contract. See [`examples/streaming.py`](examples/streaming.py).

## Voice-level calibration

Voice-level processing is opt-in through `VoiceLevelConfig`. The packaged schema-1 calibration catalog is empty because no measured corpus was supplied. In calibrated mode, a record is used only for an exact backend, canonical model, runtime revision, and (for ONNX) profile identity. If there is no exact record, the result reports `source="missing_calibration"` and the audio is unchanged. No gain or target loudness has been invented.

A caller may provide an explicit static `gain_db` override. This applies linear gain without measuring loudness or clipping the in-memory result; PCM WAV conversion may clip. Example: [`examples/calibrated_voice_level.py`](examples/calibrated_voice_level.py).

## Discovery and runtime identity

`request_api_contract()`, native `discover_bundles()`, and `runtime_identity()` do not load a speech model or import the optional runtimes. ONNX discovery may read or refresh ONNXVoice catalog metadata, but it does not open a model. Runtime identity reports exact package/model metadata where available; missing revisions stay unknown rather than being guessed. See [`examples/discovery.py`](examples/discovery.py).

## Backend architecture

SoproSynth owns the public request/result types, input validation, reference-voice lifecycle, static calibration policy, metadata normalization, and WAV output. The native backend delegates synthesis to upstream Sopro. The ONNX boundary delegates catalog, bundle, profile/provider, tokenizer, DSP, graph/session, and inference behavior to ONNXVoice; SoproSynth does not register an ONNX adapter or execute graphs itself.

## ONNX status

The optional integration targets `onnxvoice>=0.2.6,<0.3` and uses ONNXVoice's built-in Sopro API. CPU and GPU extras forward ONNXVoice's corresponding extras. ONNX is opt-in; install `soprosynth[cpu]` only with a compatible ONNXVoice release. This implementation was exercised against a modified sibling checkout that reported version 0.2.5; it has **not** been verified against a published ONNXVoice 0.2.6 wheel. Do not treat that local smoke check as released-wheel validation.

The current Sopro parity boundary is limited to the explicitly supported `wasm-uint8` profile with CPU execution. Other profiles/providers fail explicitly; installing a GPU extra does not change this support statement. ONNX streaming is unsupported. Real-bundle inference and quality parity remain unverified here. For local export metadata inspection only, use [`examples/onnx_inspect.py`](examples/onnx_inspect.py) or `soprosynth inspect`; this experimental developer command does not synthesize speech.

Managed ONNX acquisition uses ONNXVoice catalogs and its asset manager. To open an already installed ONNXVoice model without acquiring assets, use `SoproRuntime.load(...)`. To inspect a local compatible export, use `SoproRuntime.from_onnx_local(...)`. The optional ONNX path is not the default and should not be represented as validated against a published 0.2.6 distribution until that check has actually passed.

## Offline behavior

Importing `soprosynth`, querying the API contract, reading native runtime identity, and native static discovery do not load a model or require a network connection. Native `from_pretrained()` may download upstream Sopro weights unless they are already cached. ONNX `from_pretrained(backend="onnx")` may refresh catalogs or acquire bundles; use `discover_bundles(backend="onnx", offline=True)` for offline catalog access and `SoproRuntime.load(...)` or `from_onnx_local(...)` for assets already present locally. Offline mode does not create missing assets.

## CLI

```text
soprosynth speak TEXT --ref reference.wav [--out speech.wav]
soprosynth info
soprosynth discover [--backend native|onnx|all] [--offline]
soprosynth inspect EXPORT_DIRECTORY  # experimental metadata inspection only
```

`speak` supports `--backend`, `--model`, `--device`, `--language`, `--temperature`, `--top-p`, `--top-k`, `--max-seconds`, `--min-seconds`, `--voice-level`, and `--gain-db`. `--gain-db` requires calibrated voice-level mode. `discover` also accepts `--refresh` and `--cache-dir`; `inspect` accepts `--profile`. ONNX-specific inspection is kept separate from stable speech commands. Run `soprosynth speak --help` for option details.

## Migration from the pre-release scaffold

The durable request type requires a stable ID; use keyword arguments instead of constructing a request from text alone:

```python
# Durable API
SynthesisRequest(
    id="speech-001",
    text="Hello",
    language="en",
)
```

`SynthesisRequest("Hello", language="en")` is not the supported request form. No misleading `AudioResult` alias is exported; use `SynthesisResult`. The former producer-owned ONNX adapter/graph execution surface is not part of the normal public API. Use `SoproRuntime` and ONNXVoice's supported built-in Sopro integration instead.

## Versioning, provenance, and license

Package versions are derived from Git tags by `setuptools_scm`; the `v0.1.0` release tag has not been created by this implementation task. SoproSynth source code is Apache-2.0. SoproSynth does not redistribute pretrained weights. Native weights, ONNX exports, and reference recordings are separate assets: obtain them from their respective sources and follow their licenses and usage terms. SoproSynth does not claim rights to user-provided reference audio.

## Examples

- [`examples/quickstart.py`](examples/quickstart.py)
- [`examples/strict_request.py`](examples/strict_request.py)
- [`examples/prepared_voice.py`](examples/prepared_voice.py)
- [`examples/multilingual.py`](examples/multilingual.py)
- [`examples/streaming.py`](examples/streaming.py)
- [`examples/calibrated_voice_level.py`](examples/calibrated_voice_level.py)
- [`examples/discovery.py`](examples/discovery.py)
- [`examples/onnx_inspect.py`](examples/onnx_inspect.py) (experimental, inspection only)
