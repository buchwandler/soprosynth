"""Native streaming only; ONNX streaming is not implied or supported."""

import numpy as np

from soprosynth import SoproRuntime, SynthesisRequest
from soprosynth.audio import save_wav

request = SynthesisRequest(id="stream-001", text="A short streamed sample.", language="en")
with SoproRuntime.from_pretrained(backend="native") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    chunks = list(runtime.stream(request, voice=voice))
    if not chunks:
        raise RuntimeError("native stream produced no audio chunks")
    audio = np.concatenate([chunk.audio for chunk in chunks])
    save_wav(audio, chunks[0].sample_rate, "streamed.wav")
