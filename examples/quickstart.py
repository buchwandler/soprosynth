"""Native one-request quickstart.

Requires ``soprosynth[native]``, model weights, and an authorized reference.wav;
opening an uncached model may download weights. No language detection is run.
"""

from soprosynth import SoproRuntime

with SoproRuntime.from_pretrained(backend="native") as runtime:
    result = runtime.synthesize_text(
        "Hello from SoproSynth.",
        id="hello",
        language="en",
        voice="reference.wav",
    )
    result.save_wav("hello.wav")
