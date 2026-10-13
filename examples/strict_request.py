"""Demonstrate the stable SoproSynth request/result API.

Requires ``soprosynth[cpu]``, compatible model assets, and an authorized
reference.wav.
"""

from soprosynth import GenerationConfig, SoproRuntime, SynthesisRequest

request = SynthesisRequest(
    id="segment-001",
    text="Guten Morgen.",
    language="de",
)

with SoproRuntime.from_pretrained(
    model="sopro:v2-turbo",
    profile="wasm-uint8",
    providers="cpu",
) as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize(
        request,
        voice=voice,
        config=GenerationConfig(temperature=0.8, top_p=0.9),
    )
    result.save_wav("guten-morgen.wav")
