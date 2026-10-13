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

Supported language IDs are `en`, `pt`, `fr`, `de`, and `auto` (or `None` where allowed). SoproSynth does not run language detection. It does not implement text segmentation, pronunciation overrides, linguistic tokens, word timings, or final mastering. `request_api_contract()` is the API-v1 producer-level union: a true feature may belong to another backend. Inspect `runtime.capabilities` for the selected backend without changing API v1 or loading another runtime.

`GenerationConfig` provides validated controls such as `temperature`, `top_p`, `top_k`, `steps`, `max_seconds`, and `min_seconds`. A backend must reject unsupported controls rather than silently ignore them.

## Reference voice preparation and reuse

`runtime.prepare_voice(...)` validates and prepares a reference for the selected runtime. Reuse the returned `PreparedVoice` for multiple requests on that same runtime. Prepared voices are runtime-bound and are not portable to another runtime, backend, model revision, or process. The library retains a fingerprint for compatibility checks; it does not put raw reference samples in public identity metadata.

## Multilingual usage

Pass an explicit language ID for every request when the language is known. See [`examples/multilingual.py`](examples/multilingual.py) for `en`, `pt`, `fr`, and `de`. The API does not guess the language.

## Streaming

Native streaming is available through `runtime.stream(request, voice=prepared_voice)`. It yields `StreamingChunk` values; check `runtime.capabilities.supports_streaming` before using it. Exact request measurement is currently available only on ONNXVoice runtimes (`supports_request_measurement`). The v1 contract reports the producer-level union, not per-backend support. ONNX streaming is unsupported. See [`examples/streaming.py`](examples/streaming.py).

## Voice-level calibration

Voice-level processing is opt-in through `VoiceLevelConfig`. The packaged schema-2 calibration catalog is empty because no measured corpus was supplied. In calibrated mode, a record is used only for an exact backend, canonical model, runtime revision, and (for ONNX) profile identity. If there is no exact record, the result reports `source="missing_calibration"` and the audio is unchanged. No gain or target loudness has been invented; legacy schema-1 records remain readable.

A caller may provide an explicit static `gain_db` override. This applies linear gain without measuring loudness or clipping the in-memory result; PCM WAV conversion may clip. Example: [`examples/calibrated_voice_level.py`](examples/calibrated_voice_level.py).

For an offline measured catalog, install `soprosynth[audio]` and run `soprosynth-build-calibration` on licensed PCM WAV outputs. The command requires corpus/license/provenance metadata, an exact model revision, explicit target LUFS and true-peak ceiling, and `--i-have-rights`; it records the `audiosig` version, per-sample rates/durations/loudness/sample/true peaks, and source-file SHA-256 values. It targets the median per-file integrated LUFS while limiting the worst 4x true peak, and rejects gains outside ±24 dB by default. The command writes a reviewable external catalog and refuses to write into the packaged catalog. Keep that production catalog empty until real licensed measurements have been reviewed; never commit reference audio or generated speech.

Reference-rate conversion remains dependency-free linear interpolation by default. To opt into anti-aliased resampling, install `soprosynth[audio]` and call `runtime.prepare_voice(reference, resampler="audiosig")`; ONNX Runtime and audiosig remain optional.

## Discovery and runtime identity

`request_api_contract()`, native `discover_bundles()`, and `runtime_identity()` do not load a speech model or import optional runtimes. ONNX discovery may read or refresh ONNXVoice catalog metadata, but does not open a model. Runtime identity reports exact package/model metadata where available; missing revisions stay unknown rather than being guessed. See [`examples/discovery.py`](examples/discovery.py), which works with the NumPy-only base install.

## Backend architecture

SoproSynth owns the public request/result types, input validation, reference-voice lifecycle, static calibration policy, metadata normalization, and WAV output. The native backend delegates synthesis to upstream Sopro. The ONNX boundary delegates catalog, bundle, profile/provider, tokenizer, DSP, graph/session, and inference behavior to ONNXVoice; SoproSynth does not register an ONNX adapter or execute graphs itself.

## ONNX status

The optional integration targets `onnxvoice>=0.2.6,<0.3`, which includes published 0.2.6 and its built-in Sopro API. CPU and GPU extras forward ONNXVoice's corresponding runtime dependencies. The opt-in real gate rejects editable ONNXVoice checkouts and covers managed install, offline reopen, reference preparation/inference, WAV output, and diagnostics. The wheel-based smoke passed in this implementation run; details and limits are recorded below.

The tested Sopro parity boundary remains `wasm-uint8` with CPU execution only; other profiles/providers fail explicitly, and installing a GPU extra does not change that boundary. ONNX streaming is unsupported. The real smoke passed with the published ONNXVoice 0.2.6 wheel, Python 3.14, ONNX Runtime 1.29.0, `CPUExecutionProvider`, and manifest revision `7a645ef2aaecfea9` (catalog source commit `9fe97bddf52ed785af3634db86a032bb7e07bf18`). Its reference was `kyutai/tts-voices` `voice-donations/f9cf.wav`, pinned at `323332d33f997de8394f24a193e1a76df720e01a`, marked CC0-1.0, SHA-256 `9c35277376fd34d8217672cedd8fa098e2321f243007d190838ac6a93cf4dbaf`. ONNX Runtime warned that Android is unsupported; this is a smoke result for that environment, not a platform-support, quality, or broad parity claim. For experimental local inspection, [`examples/onnx_inspect.py`](examples/onnx_inspect.py) and `soprosynth inspect` open an ONNXVoice runtime/session and print diagnostics but do not synthesize; they require optional ONNX dependencies and a compatible local export.

Managed ONNX acquisition uses ONNXVoice catalogs and its asset manager. To open an already installed model without acquiring assets, use `SoproRuntime.load(...)`; for a local compatible export, use `SoproRuntime.from_onnx_local(...)`. Neither API implies support for untested profiles/providers or quality parity.

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

## PyPI release ownership

The publish job uses PyPI Trusted Publishing (OIDC), not a stored API token. Configure the PyPI Trusted Publisher with GitHub owner `buchwandler`, repository `soprosynth`, workflow `publish.yml`, and GitHub environment `pypi`; the publish job grants `id-token: write`. For the first publication, leave the repository variable `PYPI_PROJECT_CLAIMED` unset/false so the workflow requires the name to be unclaimed. After a successful first publish and verification that the Trusted Publisher is correctly bound, set `PYPI_PROJECT_CLAIMED=true`. Later tagged releases skip the unclaimed-name check and rely on PyPI's Trusted Publishing authorization. Do not set the variable early to bypass the first-publication guard.

## Versioning, provenance, and license

Package versions are derived from Git tags by `setuptools_scm`; the `v0.1.0` release tag has not been created by this implementation task. SoproSynth source code is Apache-2.0. SoproSynth does not redistribute pretrained weights. Native weights, ONNX exports, and reference recordings are separate assets: obtain them from their respective sources and follow their licenses and usage terms. SoproSynth does not claim rights to user-provided reference audio.

## Examples

For prerequisites and capability limits, see [`examples/README.md`](examples/README.md).

- [`examples/quickstart.py`](examples/quickstart.py)
- [`examples/strict_request.py`](examples/strict_request.py)
- [`examples/prepared_voice.py`](examples/prepared_voice.py)
- [`examples/multilingual.py`](examples/multilingual.py)
- [`examples/streaming.py`](examples/streaming.py)
- [`examples/calibrated_voice_level.py`](examples/calibrated_voice_level.py)
- [`examples/discovery.py`](examples/discovery.py)
- [`examples/onnx_inspect.py`](examples/onnx_inspect.py) (experimental; opens a local session, no synthesis)
