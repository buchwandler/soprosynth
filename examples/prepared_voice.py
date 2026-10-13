"""Prepare one reference voice and reuse it within the same runtime.

Requires ``soprosynth[cpu]``, compatible model assets, and an authorized
reference.wav. Prepared voice state is runtime-bound and is not portable across
runtimes or processes.
"""

from soprosynth import SoproRuntime, SynthesisRequest

with SoproRuntime.from_pretrained() as runtime:
    voice = runtime.prepare_voice("reference.wav")
    for index, text in enumerate(("First request.", "Second request."), start=1):
        request = SynthesisRequest(
            id=f"segment-{index:03d}",
            text=text,
            language="en",
        )
        runtime.synthesize(request, voice=voice).save_wav(f"segment-{index:03d}.wav")
