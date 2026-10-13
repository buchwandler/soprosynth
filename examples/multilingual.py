"""Use explicit IDs for four native requests; no language detection is run.

Requires ``soprosynth[native]``, model weights, and an authorized reference.wav.
Explicit IDs do not promise language quality or automatic language support.
"""

from soprosynth import SoproRuntime, SynthesisRequest

samples = {
    "en": "Hello from SoproSynth.",
    "pt": "Olá do SoproSynth.",
    "fr": "Bonjour de SoproSynth.",
    "de": "Hallo von SoproSynth.",
}

with SoproRuntime.from_pretrained(backend="native") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    for language, text in samples.items():
        request = SynthesisRequest(
            id=f"greeting-{language}",
            text=text,
            language=language,
        )
        runtime.synthesize(request, voice=voice).save_wav(f"greeting-{language}.wav")
