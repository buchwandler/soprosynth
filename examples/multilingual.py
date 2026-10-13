"""Submit explicit-language examples; no language detection is performed.

Requires ``soprosynth[cpu]``, compatible model assets, and an authorized
reference.wav. Explicit language labels do not guarantee perceptual quality.
"""

from soprosynth import SoproRuntime, SynthesisRequest

samples = {
    "en": "Hello from SoproSynth.",
    "pt": "Olá do SoproSynth.",
    "fr": "Bonjour de SoproSynth.",
    "de": "Hallo von SoproSynth.",
}

with SoproRuntime.from_pretrained() as runtime:
    voice = runtime.prepare_voice("reference.wav")
    for language, text in samples.items():
        request = SynthesisRequest(
            id=f"greeting-{language}",
            text=text,
            language=language,
        )
        runtime.synthesize(request, voice=voice).save_wav(f"greeting-{language}.wav")
