"""Public runtime orchestration; backend implementations stay behind lazy boundaries."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
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
    RuntimeCapabilityError,
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
    StreamingChunk,
    SynthesisRequest,
    SynthesisResult,
)
from .voice import ReferenceAudio, fingerprint_audio, load_reference_audio


class SoproRuntime:
    """Context-managed synthesis runtime with native Sopro as the default backend."""

    def __init__(
        self,
        engine: Any,
        *,
        backend: str = "native",
        model_id: str = "samuel-vitorino/sopro-v2-turbo",
        source_revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self._engine = engine
        self.backend = backend
        self.profile = getattr(engine, "profile", None)
        self.model_id = model_id
        self.source_revision = source_revision
        self.device = device
        self.sample_rate = int(getattr(engine, "sample_rate", 24000))
        self._closed = False
        self._runtime_fingerprint = uuid.uuid4().hex
        self._active_streams: set[Any] = set()

    @classmethod
    def from_pretrained(
        cls,
        model: str = "samuel-vitorino/sopro-v2-turbo",
        *,
        backend: str = "native",
        device: str = "cpu",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        quantization: str | None = None,
        profile: str | None = None,
        providers: str | tuple[str, ...] | None = None,
        provider_options: Any = None,
        session_options: Any = None,
        force: bool = False,
        refresh: bool = False,
        progress: Any = None,
    ) -> SoproRuntime:
        if backend == "native":
            if any(
                value is not None
                for value in (profile, providers, provider_options, session_options)
            ):
                raise RuntimeCapabilityError(
                    "ONNX runtime options cannot be used with the native backend"
                )
            if force or refresh or progress is not None:
                raise RuntimeCapabilityError("asset installation options require backend='onnx'")
            from ._native import NativeBackend

            engine = NativeBackend.from_pretrained(
                model,
                device=device,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                quantization=quantization,
            )
            return cls(
                engine,
                backend="native",
                model_id=model,
                source_revision=revision,
                device=device,
            )
        if backend == "onnx":
            if revision is not None or cache_dir is not None or quantization is not None:
                raise RuntimeCapabilityError(
                    "revision, cache_dir, and quantization are not ONNXVoice Sopro controls"
                )
            from ._onnxvoice import install_and_open

            engine = install_and_open(
                model,
                profile=profile if profile is not None else "wasm-uint8",
                providers=providers if providers is not None else device,
                provider_options=provider_options,
                session_options=session_options,
                force=force,
                refresh=refresh,
                progress=progress,
            )
            return cls(
                engine,
                backend="onnx",
                model_id=engine.model_id,
                source_revision=engine.source_revision,
                device=device,
            )
        raise RuntimeCapabilityError(f"Unknown SoproSynth backend {backend!r}")

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
            backend="onnx",
            model_id=engine.model_id,
            source_revision=engine.source_revision,
            device=str(providers),
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
            backend="onnx",
            model_id=engine.model_id,
            source_revision=engine.source_revision,
            device=str(providers),
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
            backend="onnx",
            model_id=engine.model_id,
            source_revision=engine.source_revision,
            device=str(providers),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("SoproRuntime is closed")

    def prepare_voice(
        self,
        audio: Any,
        *,
        stream: bool = False,
        resampler: Literal["linear", "audiosig"] = "linear",
    ) -> PreparedVoice:
        self._ensure_open()
        if type(stream) is not bool:
            raise InvalidVoiceError("stream must be a bool")
        reference = load_reference_audio(
            audio, expected_sample_rate=self.sample_rate, resampler=resampler
        )
        if isinstance(reference, PreparedVoice):
            self._validate_voice(reference)
            return reference
        if not isinstance(reference, ReferenceAudio):
            raise InvalidVoiceError("reference audio could not be normalized")
        try:
            state = self._engine.prepare_reference(reference, stream=stream)
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
        if self.backend != "onnx":
            raise UnsupportedFeatureError(
                "request_measurement",
                "Exact token measurement is currently available only through ONNXVoice Sopro",
            )
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

        kwargs: dict[str, Any] = selected_config.as_native_kwargs()
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

    def stream(
        self,
        request: SynthesisRequest,
        *,
        voice: PreparedVoice,
        config: GenerationConfig | None = None,
        chunk_frames: int | None = None,
    ) -> Iterator[StreamingChunk]:
        self._ensure_open()
        self._validate_request(request)
        if config is not None and not isinstance(config, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        selected_config = config if config is not None else GenerationConfig()
        self._validate_voice(voice)
        if self.backend != "native":
            raise RuntimeCapabilityError("streaming is not supported by this backend")
        kwargs: dict[str, Any] = selected_config.as_native_kwargs()
        if request.language not in (None, "auto"):
            kwargs["lang"] = request.language
        if chunk_frames is not None:
            if type(chunk_frames) is not int or chunk_frames <= 0:
                raise InvalidGenerationConfigError("chunk_frames must be a positive integer")
            kwargs["chunk_frames"] = chunk_frames
        kwargs["ref"] = voice.state
        iterator: Iterator[Any] | None = None
        try:
            iterator = iter(self._engine.stream(request.text, **kwargs))
            self._active_streams.add(iterator)
            for index, chunk in enumerate(iterator):
                yield StreamingChunk(
                    id=request.id,
                    index=index,
                    audio=_to_numpy(chunk),
                    sample_rate=self.sample_rate,
                )
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Native Sopro streaming failed") from exc
        finally:
            if iterator is not None:
                self._active_streams.discard(iterator)
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    @property
    def capabilities(self) -> RuntimeCapabilities:
        """Return feature support for this selected backend without loading extras."""
        return RuntimeCapabilities(
            backend=self.backend,
            supports_request_measurement=self.backend == "onnx",
            supports_streaming=self.backend == "native",
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
            device=self.device,
        )

    def close(self) -> None:
        if not self._closed:
            try:
                for stream in tuple(self._active_streams):
                    close_stream = getattr(stream, "close", None)
                    if close_stream is not None:
                        try:
                            close_stream()
                        except RuntimeError:
                            pass
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
