# SoproSynth

Minimal, flat-layout Python API for Samuel Vitorino's **Sopro V2 Turbo**, inspired by
PocketSynth. Python >=3.10. Packaging uses setuptools_scm Git-tag versioning
(`git tag v0.1.0`, `python -m build`); there is **no src/ directory**.

## Install

```bash
python -m pip install -e '.[native,dev]'
pytest -q
```

The **native** backend works through the upstream `sopro>=2.2.0` PyTorch implementation,
using Apache-2.0 upstream model weights. It will download real weights on first run.

```python
from soprosynth import SoproRuntime, SynthesisRequest
with SoproRuntime.from_pretrained() as tts:
    voice = tts.prepare_voice('reference.wav')
    result = tts.synthesize(SynthesisRequest('Hello!',language='en'), voice=voice)
    result.save_wav('hello.wav')
```

## Embedded OnnxVoice ONNX graph runtime (partial)

```bash
pip install -e '.[cpu]'
soprosynth inspect /path/to/sopro-v2-turbo-onnx --profile wasm-uint8
```

```python
from soprosynth import SoproRuntime
with SoproRuntime.from_onnx_local('/path/to/export', profile='wasm-uint8') as tts:
    adapter = tts.onnx_adapter
    print(adapter.graphs)
    print(adapter.graph_specs('reference'))  # initializes ONNX Runtime session
    # adapter.run_graph('reference', mapping_of_exact_onnx_input_tensors)
```

This uses public OnnxVoice `register_adapter` + `OnnxVoice.open_local` and `OnnxSession`.
ONNX artifacts must be downloaded separately in **original export folder layout**:
`manifest.json`, `tokenizer.json`, `common/`, `wasm-uint8/`, `webgpu-fp16/`.
The `sopro-onnx-bundles` companion repo provides the pinned-revision catalog builder
and a development-only downloader for local ONNX graph testing.

**Important limitation:** ONNX `synthesize`, `prepare_voice` and streaming are not yet
implemented. They raise `OnnxPipelineNotImplementedError` rather than returning fake
speech. The model has a reference encoder, autoregressive semantic decoder, acoustic
conditioning + ODE solver and vocoder. An ONNX port must reproduce the upstream JS
host-side DSP, token sampling and generation loop. `wasm-uint8` is exported for
ONNX Runtime Web/WASM; support in native Python ONNX Runtime requires actual graph tests.

## Scope and responsibilities

- **SoproSynth:** API, request validation, WAV output, Sopro-specific voice lifecycle,
  optional working upstream native backend.
- **OnnxVoice:** registered Sopro ONNX adapter, ONNX sessions, execution providers,
  graph diagnostics (with this MVP's in-package adapter registration).
- **sopro-onnx-bundles:** manifest ingest, revision pinning, catalog generation,
  provenance; not a Python package and not a model mirror.

## TODO for real ONNX synthesis

1. Validate all graphs + shards on ONNX Runtime CPU and compare to official web outputs.
2. Port `web/src/dsp.js` reference frontend and tokenizer JSON from official runtime.
3. Port `semantic_prefix` + `semantic_core` cached autoregressive token loop.
4. Port `acoustic_condition` + two-step offline acoustic sampling / rotary inputs.
5. Port vocoder output normalization, append audio parity and streaming state.
6. Move Sopro catalog parsing, install and the adapter into an upstream OnnxVoice PR;
   then simplify this in-package adapter to a thin compatibility layer.

Upstream: https://github.com/samuel-vitorino/sopro
ONNX weights: https://huggingface.co/samuel-vitorino/sopro-v2-turbo-onnx

Software is Apache-2.0. No pretrained weights are redistributed.
