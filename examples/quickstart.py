"""One-request SoproSynth quickstart.

Requires ``soprosynth[cpu]``, compatible model assets, and an authorized
reference.wav. The first uncached run may acquire the ONNXVoice Sopro bundle.
"""

from soprosynth import SoproRuntime

with SoproRuntime.from_pretrained() as runtime:
    result = runtime.synthesize_text(
        "Hello from SoproSynth.",
        id="hello",
        language="en",
        voice="reference.wav",
    )
    result.save_wav("hello.wav")
