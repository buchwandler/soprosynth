"""Demonstrate the stable request/result API."""

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
        config=GenerationConfig(),
    )
    result.save_wav("guten-morgen.wav")
