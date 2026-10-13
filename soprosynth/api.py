"""Public runtime orchestration with a lazy ONNXVoice inference boundary."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .config import GenerationConfig, VoiceLevelConfig
from .errors import (
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    ModelInferenceError,
    RuntimeClosedError,
    SoproSynthError,
    UnsupportedFeatureError,
)
from .types import (
    LANGUAGES,
    PreparedVoice,
    RequestMeasure,
    RuntimeCapabilities,
    RuntimeDiagnostics,
    SynthesisRequest,
    SynthesisResult,
)
from .voice import ReferenceAudio, fingerprint_audio, load_reference_audio


class SoproRuntime:
    """Context-managed synthesis runtime backed exclusively by ONNXVoice."""

    def __init__(
        self,
        engine: Any,
        *,
        model_id: str = "sopro:v2-turbo",
        source_revision: str | None = None,
    ) -> None:
        self._engine = engine
        self.backend = "onnx"
        self.profile = getattr(engine, "profile", None)
        self.model_id = model_id
        self.source_revision = source_revision
        self.sample_rate = int(getattr(engine, "sample_rate", 24000))
        self._closed = False
        self._runtime_fingerprint = uuid.uuid4().hex

    @classmethod
    def from_pretrained(
        cls,
        model: str = "sopro:v2-turbo",
        *,
        profile: str = "wasm-uint8",
        providers: str | tuple[str, ...] = "cpu",
        provider_options: Any = None,
        session_options: Any = None,
        force: bool = False,
        refresh: bool = False,
        progress: Any = None,
    ) -> SoproRuntime:
        from ._onnxvoice import install_and_open

        engine = install_and_open(
            model,
            profile=profile,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            force=force,
            refresh=refresh,
            progress=progress,
        )
        return cls(
            engine,
            model_id=engine.model_id,
            source_revision=engine.source_revision,
        )

    @classmethod
    def from_onnx_local(
        cls,
        directory: str | Path,
        *,
        profile: str = "wasm-uint8",
        providers: str | tuple[str, ...] = "cpu",
        provider_options: Any = None,
        session_options: Any = None,
        verify: bool = True,
    ) -> SoproRuntime:
        """Open a verified ONNXVoice-compatible local Sopro export."""
        from ._onnxvoice import open_local

        engine = open_local(
            directory,
            profile=profile,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            verify=verify,
        )
        return cls(
            engine,
            model_id=engine.model_id,
            source_revision=engine.source_revision,
        )

    @classmethod
    def load(
        cls,
        ref: str,
        *,
        profile: str = "wasm-uint8",
        providers: str | tuple[str, ...] = "cpu",
        provider_options: Any = None,
        session_options: Any = None,
    ) -> SoproRuntime:
        """Open an already-installed ONNXVoice model without acquiring assets."""
        from ._onnxvoice import open_managed

        engine = open_managed(
            ref,
            profile=profile,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(
            engine,
            model_id=engine.model_id,
            source_revision=engine.source_revision,
        )

    @classmethod
    def from_resolved(
        cls,
        installation: Any,
        *,
        profile: str = "wasm-uint8",
        providers: str | tuple[str, ...] = "cpu",
        provider_options: Any = None,
        session_options: Any = None,
    ) -> SoproRuntime:
        """Open an ONNXVoice-resolved installation without resolving or downloading."""
        from ._onnxvoice import open_resolved

        engine = open_resolved(
            installation,
            profile=profile,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(
            engine,
            model_id=engine.model_id,
            source_revision=engine.source_revision,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("SoproRuntime is closed")

    def prepare_voice(
        self,
        audio: Any,
        *,
        resampler: Literal["linear", "audiosig"] = "linear",
    ) -> PreparedVoice:
        self._ensure_open()
        reference = load_reference_audio(
            audio, expected_sample_rate=self.sample_rate, resampler=resampler
        )
        if isinstance(reference, PreparedVoice):
            self._validate_voice(reference)
            return reference
        if not isinstance(reference, ReferenceAudio):
            raise InvalidVoiceError("reference audio could not be normalized")
        try:
            state = self._engine.prepare_reference(reference)
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro reference preparation failed") from exc
        return PreparedVoice(
            state=state,
            sample_rate=reference.sample_rate,
            source=reference.source,
            model_id=self.model_id,
            source_revision=self.source_revision,
            backend=self.backend,
            runtime_fingerprint=self._runtime_fingerprint,
            fingerprint=fingerprint_audio(reference.audio, reference.sample_rate),
            metadata={"reference_sample_rate": reference.sample_rate},
            _owner_id=id(self),
        )

    def _validate_voice(self, voice: PreparedVoice) -> None:
        if not isinstance(voice, PreparedVoice):
            raise InvalidVoiceError("voice must be a PreparedVoice")
        if voice.backend != self.backend:
            raise InvalidVoiceError("PreparedVoice backend does not match this runtime")
        if voice.model_id != self.model_id:
            raise InvalidVoiceError("PreparedVoice model does not match this runtime")
        if voice.source_revision != self.source_revision:
            raise InvalidVoiceError("PreparedVoice revision does not match this runtime")
        if voice.sample_rate != self.sample_rate:
            raise InvalidVoiceError("PreparedVoice sample rate does not match this runtime")
        if voice.runtime_fingerprint != self._runtime_fingerprint or voice._owner_id != id(self):
            raise InvalidVoiceError("PreparedVoice is owned by another runtime")

    def _validate_request(self, request: SynthesisRequest) -> None:
        if not isinstance(request, SynthesisRequest):
            raise InvalidRequestError("request must be a SynthesisRequest")
        if not request.text.strip():
            raise EmptyTextError("text must not be empty or whitespace-only")
        if request.language not in (None, "auto", *LANGUAGES):
            raise InvalidLanguageError("unsupported synthesis language")
        if request.tokens:
            raise UnsupportedFeatureError("linguistic_tokens")
        if request.pronunciation_overrides:
            raise UnsupportedFeatureError("pronunciation_overrides")

    def measure_request(
        self,
        request: SynthesisRequest,
        *,
        maximum: int | None = None,
    ) -> RequestMeasure:
        self._ensure_open()
        self._validate_request(request)
        if maximum is not None and (type(maximum) is not int or maximum < 0):
            raise InvalidRequestError("maximum must be a non-negative integer or None")
        language = request.language if request.language not in (None, "auto") else None
        try:
            amount = self._engine.measure_text(request.text, language=language)
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro request token measurement failed") from exc
        return RequestMeasure(amount=amount, maximum=maximum)

    def synthesize(
        self,
        request: SynthesisRequest,
        *,
        voice: PreparedVoice,
        config: GenerationConfig | None = None,
        voice_level: VoiceLevelConfig | None = None,
    ) -> SynthesisResult:
        self._ensure_open()
        self._validate_request(request)
        if config is not None and not isinstance(config, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        selected_config = config if config is not None else GenerationConfig()
        if voice_level is not None and not isinstance(voice_level, VoiceLevelConfig):
            raise InvalidGenerationConfigError("voice_level must be a VoiceLevelConfig")
        selected_voice_level = voice_level if voice_level is not None else VoiceLevelConfig()
        self._validate_voice(voice)

        kwargs: dict[str, Any] = selected_config.as_inference_kwargs()
        if request.language not in (None, "auto"):
            kwargs["lang"] = request.language
        kwargs["ref"] = voice.state
        try:
            raw_audio = self._engine.synthesize(request.text, **kwargs)
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro synthesis failed") from exc
        audio = _to_numpy(raw_audio)
        from .calibration import apply_voice_level_calibration, load_calibration_catalog

        calibration_catalog = (
            load_calibration_catalog()
            if selected_voice_level.mode == "calibrated" and selected_voice_level.gain_db is None
            else {}
        )
        calibration_identity = {
            "backend": self.backend,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "profile": self.profile,
        }
        audio, voice_level_application = apply_voice_level_calibration(
            audio,
            selected_voice_level,
            identity=calibration_identity,
            catalog=calibration_catalog,
        )
        diagnostics = self.diagnostics
        from .discovery import runtime_identity

        diagnostic_metadata = {
            "backend": diagnostics.backend,
            "model_id": diagnostics.model_id,
            "sample_rate": diagnostics.sample_rate,
            "source_revision": diagnostics.source_revision,
            "device": diagnostics.device,
            "providers_requested": diagnostics.providers_requested,
            "providers_active": diagnostics.providers_active,
            "sessions": diagnostics.sessions,
        }
        return SynthesisResult(
            id=request.id,
            audio=audio,
            sample_rate=self.sample_rate,
            text=request.text,
            language=request.language or "auto",
            metadata={
                "backend": self.backend,
                "model_id": self.model_id,
                "model_revision": self.source_revision,
                "voice_identity": dict(voice.identity),
                "language": request.language or "auto",
                "generation_config": asdict(selected_config),
                "effective_generation": None,
                "voice_level_config": asdict(selected_voice_level),
                "voice_level_application": asdict(voice_level_application),
                "runtime_identity": runtime_identity(self),
                "runtime_diagnostics": diagnostic_metadata,
            },
        )

    def synthesize_text(
        self,
        text: str,
        *,
        voice: PreparedVoice | str | Path | tuple[Any, int],
        id: str = "speech",
        language: str | None = None,
        config: GenerationConfig | None = None,
        voice_level: VoiceLevelConfig | None = None,
    ) -> SynthesisResult:
        self._ensure_open()
        request = SynthesisRequest(id=id, text=text, language=language)
        if not request.text.strip():
            raise EmptyTextError("text must not be empty or whitespace-only")
        prepared = voice if isinstance(voice, PreparedVoice) else self.prepare_voice(voice)
        return self.synthesize(request, voice=prepared, config=config, voice_level=voice_level)

    @property
    def capabilities(self) -> RuntimeCapabilities:
        """Return capabilities of the sole supported ONNXVoice runtime."""
        return RuntimeCapabilities(
            backend="onnx",
            supports_request_measurement=True,
            supports_streaming=False,
        )

    @property
    def diagnostics(self) -> RuntimeDiagnostics:
        self._ensure_open()
        backend_diagnostics = getattr(self._engine, "diagnostics", None)
        if callable(backend_diagnostics):
            return backend_diagnostics()
        return RuntimeDiagnostics(
            backend=self.backend,
            model_id=self.model_id,
            sample_rate=self.sample_rate,
            source_revision=self.source_revision,
        )

    def close(self) -> None:
        if not self._closed:
            try:
                close = getattr(self._engine, "close", None)
                if close is not None:
                    close()
            finally:
                self._closed = True

    def __enter__(self) -> SoproRuntime:  # noqa: PYI034 - keep Python 3.10 free of typing_extensions
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelInferenceError("Backend returned an invalid audio value") from exc
