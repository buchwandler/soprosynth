from soprosynth import SoproRuntime, SynthesisRequest

# Requires pip install -e '.[native]' and your own licensed reference WAV.
with SoproRuntime.from_pretrained() as tts:
    voice = tts.prepare_voice("reference.wav")
    result = tts.synthesize(SynthesisRequest("Hello from SoproSynth.", language="en"), voice=voice)
    result.save_wav("hello.wav")
