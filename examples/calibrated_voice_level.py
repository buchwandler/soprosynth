"""Demonstrate calibrated mode without inventing a catalog calibration value."""

from soprosynth import SoproRuntime, SynthesisRequest, VoiceLevelConfig

request = SynthesisRequest(id="calibrated-001", text="A short sample.", language="en")
with SoproRuntime.from_pretrained(backend="native") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize(
        request,
        voice=voice,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    print(result.metadata["voice_level_application"])
    result.save_wav("calibrated.wav")

# If the exact backend/model/revision has no measured catalog record, the application
# reports source="missing_calibration" and leaves samples unchanged. A caller may
# explicitly choose a static override with VoiceLevelConfig(mode="calibrated", gain_db=2.0).
