"""Smallest native SoproSynth example (requires weights and reference.wav)."""

from soprosynth import SoproRuntime

with SoproRuntime.from_pretrained(backend="native") as runtime:
    result = runtime.synthesize_text(
        "Hello from SoproSynth.",
        id="hello",
        language="en",
        voice="reference.wav",
    )
    result.save_wav("hello.wav")
