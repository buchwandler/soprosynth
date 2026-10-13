"""Prepare one reference voice and reuse it on the same runtime."""

from soprosynth import SoproRuntime, SynthesisRequest

with SoproRuntime.from_pretrained(backend="native") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    for index, text in enumerate(("First request.", "Second request."), start=1):
        request = SynthesisRequest(
            id=f"segment-{index:03d}",
            text=text,
            language="en",
        )
        runtime.synthesize(request, voice=voice).save_wav(f"segment-{index:03d}.wav")
