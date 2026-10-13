"""Show calibrated-mode fallback without inventing a calibration value.

Requires ``soprosynth[cpu]``, compatible model assets, and an authorized
reference.wav. The packaged catalog is empty, so this default request is unchanged.
"""

from soprosynth import SoproRuntime, SynthesisRequest, VoiceLevelConfig

request = SynthesisRequest(id="calibrated-001", text="A short sample.", language="en")
with SoproRuntime.from_pretrained() as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize(
        request,
        voice=voice,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    print(result.metadata["voice_level_application"])
    result.save_wav("calibrated.wav")

# When no exact model/revision/profile record exists, the application reports
# source="missing_calibration" and leaves samples unchanged. A caller may choose
# a static override with VoiceLevelConfig(mode="calibrated", gain_db=2.0).
