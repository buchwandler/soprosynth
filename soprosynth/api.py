"""Public Python API; SoproSynth owns orchestration, not ONNX session implementation."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import numpy as np

from .errors import BackendUnavailableError, OnnxPipelineNotImplementedError

LANGUAGES = frozenset(("en", "pt", "fr", "de"))


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    steps: int | None = None
    max_seconds: float | None = None
    min_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.steps is not None and self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")

    def as_native_kwargs(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("top_k", self.top_k),
                ("steps", self.steps),
                ("max_seconds", self.max_seconds),
                ("min_seconds", self.min_seconds),
            )
            if v is not None
        }


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    language: str | None = None
    config: GenerationConfig = GenerationConfig()
    id: str | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text must not be empty")
        if self.language not in (None, "auto", *LANGUAGES):
            raise ValueError("language must be auto, en, pt, fr, de or None")


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    """Precomputed upstream reference state tied to a particular runtime instance."""

    source: Path
    _state: Any
    _owner_id: int


@dataclass(frozen=True, slots=True)
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    request_id: str | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.audio)
        if data.ndim != 1 or not np.issubdtype(data.dtype, np.floating):
            raise ValueError("audio must be mono 1-D floating-point samples")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "audio", np.ascontiguousarray(data, dtype=np.float32))

    def save_wav(self, path: str | Path) -> Path:
        """Write 16-bit PCM WAV without an audio I/O dependency."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(self.audio, -1, 1) * 32767).astype("<i2")
        with wave.open(str(out), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(pcm.tobytes())
        return out


class SoproRuntime:
    """Context-managed API backed by native Sopro or embedded OnnxVoice sessions.

    backend='native' supports synthesis via upstream Sopro >=2.2.0 (PyTorch).
    backend='onnx' loads and runs individual exported ONNX graphs through OnnxVoice;
    the Sopro-specific graph pipeline is intentionally not yet implemented.
    """

    def __init__(self, engine: Any, *, backend: str) -> None:
        self._engine = engine
        self.backend = backend
        self.sample_rate = int(getattr(engine, "sample_rate", 24000))
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model: str = "samuel-vitorino/sopro-v2-turbo",
        *,
        device: str = "cpu",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        quantization: str | None = None,
    ) -> SoproRuntime:
        """Create functioning native PyTorch runtime (requires soprosynth[native])."""
        try:
            from sopro import SoproTTS
        except ImportError as exc:
            raise BackendUnavailableError("Install soprosynth[native] to run synthesis. ") from exc
        engine = SoproTTS.from_pretrained(
            model,
            device=device,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            quantization=quantization,
        )
        return cls(engine, backend="native")

    @classmethod
    def from_onnx_local(
        cls,
        directory: str | Path,
        *,
        profile: str = "wasm-uint8",
        providers: str | tuple[str, ...] = "cpu",
        verify: bool = True,
    ) -> SoproRuntime:
        """Open local multi-graph ONNX bundle via OnnxVoice's registered adapter."""
        from .onnx import open_onnx_bundle

        return cls(
            open_onnx_bundle(directory, profile=profile, providers=providers, verify=verify),
            backend="onnx",
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SoproRuntime is closed")

    def prepare_voice(self, audio: str | Path, *, stream: bool = False) -> PreparedVoice:
        self._ensure_open()
        if self.backend == "onnx":
            raise OnnxPipelineNotImplementedError(
                "ONNX reference preprocessing/encoding is not yet ported"
            )
        path = Path(audio)
        if not path.is_file():
            raise FileNotFoundError(path)
        state = self._engine.prepare_reference(ref_audio_path=path, stream=stream)
        return PreparedVoice(path, state, id(self))

    def synthesize(
        self,
        request: SynthesisRequest | str,
        *,
        voice: PreparedVoice | str | Path,
        language: str | None = None,
        config: GenerationConfig | None = None,
    ) -> AudioResult:
        self._ensure_open()
        if isinstance(request, str):
            request = SynthesisRequest(
                text=request, language=language, config=config or GenerationConfig()
            )
        elif language is not None or config is not None:
            raise ValueError("language/config overrides require string request")
        if self.backend == "onnx":
            raise OnnxPipelineNotImplementedError(
                "ONNX graph sessions work, but token generation, acoustic solver, "
                "reference DSP and vocoder stitching are not ported. Use from_pretrained() "
                "with soprosynth[native] for audio now."
            )
        kwargs = request.config.as_native_kwargs()
        if request.language not in (None, "auto"):
            kwargs["lang"] = request.language
        if isinstance(voice, PreparedVoice):
            if voice._owner_id != id(self):
                raise ValueError("PreparedVoice belongs to another runtime")
            kwargs["ref"] = voice._state
        else:
            path = Path(voice)
            if not path.is_file():
                raise FileNotFoundError(path)
            kwargs["ref_audio_path"] = path
        raw = self._engine.synthesize(request.text, **kwargs)
        samples = _to_numpy(raw)
        return AudioResult(samples, self.sample_rate, request_id=request.id)

    def stream(
        self,
        request: SynthesisRequest | str,
        *,
        voice: PreparedVoice | str | Path,
        chunk_frames: int | None = None,
    ) -> Iterator[AudioResult]:
        self._ensure_open()
        if isinstance(request, str):
            request = SynthesisRequest(text=request)
        if self.backend == "onnx":
            raise OnnxPipelineNotImplementedError("ONNX streaming pipeline not implemented")
        kwargs = request.config.as_native_kwargs()
        if request.language not in (None, "auto"):
            kwargs["lang"] = request.language
        if chunk_frames is not None:
            kwargs["chunk_frames"] = chunk_frames
        if isinstance(voice, PreparedVoice):
            if voice._owner_id != id(self):
                raise ValueError("PreparedVoice belongs to another runtime")
            kwargs["ref"] = voice._state
        else:
            path = Path(voice)
            if not path.is_file():
                raise FileNotFoundError(path)
            kwargs["ref_audio_path"] = path
        for chunk in self._engine.stream(request.text, **kwargs):
            yield AudioResult(_to_numpy(chunk), self.sample_rate, request_id=request.id)

    @property
    def onnx_adapter(self) -> Any:
        self._ensure_open()
        if self.backend != "onnx":
            raise BackendUnavailableError(
                "OnnxVoice adapter is only available with from_onnx_local"
            )
        return self._engine

    def close(self) -> None:
        if not self._closed:
            if hasattr(self._engine, "close"):
                self._engine.close()
            self._closed = True

    def __enter__(self) -> SoproRuntime:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)
