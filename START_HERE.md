# Getting started with SoproSynth

SoproSynth provides a stable request/result API around ONNXVoice's built-in Sopro V2 Turbo runtime. Install the optional CPU inference extra to synthesize:

```bash
python -m pip install "soprosynth[cpu]"
```

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

The supported Python inference boundary is model `sopro:v2-turbo`, profile `wasm-uint8`, CPU provider, English/Portuguese/French/German (or `auto`), a required reference voice, and mono float32 24 kHz output. Streaming is not supported. SoproSynth owns request validation and result/WAV handling; ONNXVoice owns model assets and inference.

For development, install `python -m pip install -e ".[cpu,dev]"`. See [README.md](README.md) for the structured API, discovery, calibration, CLI, and examples.

Use only recordings whose rights and license permit your intended use. SoproSynth does not grant rights to reference recordings or generated speech.
