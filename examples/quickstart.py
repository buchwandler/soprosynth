"""One-request SoproSynth quickstart with optional streaming prewarm.

Requires ``soprosynth[cpu]``, a released ONNXVoice >=0.2.7,<0.3 runtime, model
assets, and an authorized reference.wav. Streaming capability reporting remains
gated until the opt-in released-wheel real-model integration test passes.
"""

from soprosynth import GenerationConfig, SoproRuntime

with SoproRuntime.from_pretrained() as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize_text(
        "Hello from SoproSynth.",
        id="hello",
        language="en",
        voice=voice,
    )
    result.save_wav("hello.wav")

    config = GenerationConfig(seed=1234)
    runtime.prepare_streaming(voice, config=config)  # optional; ONNXVoice owns the cache
    for chunk in runtime.stream_text(
        "A streamed follow-up.",
        id="streamed-hello",
        voice=voice,
        config=config,
    ):
        # Replace this with an application playback or output sink.
        print(f"chunk {chunk.index}: {chunk.audio.size} samples")
        if chunk.is_final:
            print(chunk.metadata)
