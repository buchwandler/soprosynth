"""Public runtime orchestration with a lazy ONNXVoice inference boundary."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .config import GenerationConfig, SoproOutputConfig, VoiceLevelConfig
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
from .postprocess import DEFAULT_MAX_SEGMENT_CHARS, condition_sopro_output, split_text
from .types import (
    LANGUAGES,
    PreparedVoice,
    RequestMeasure,
    RuntimeCapabilities,
    RuntimeDiagnostics,
    SynthesisRequest,
    SynthesisResult,
)
from .voice import ReferenceAudio, load_reference_audio


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
        resampler: Literal["linear", "audiosig"] | None = None,
    ) -> PreparedVoice:
        """Preserve the source rate by default; explicitly select a legacy model-rate resampler."""
        self._ensure_open()
        reference = load_reference_audio(
            audio,
            expected_sample_rate=self.sample_rate if resampler is not None else None,
            resampler=resampler if resampler is not None else "linear",
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

        prepared_metadata: dict[str, int | float] = {
            "reference_sample_rate": reference.source_sample_rate,
            "prepared_reference_sample_rate": reference.sample_rate,
        }
        extract_reference_level = getattr(self._engine, "reference_level_db", None)
        if callable(extract_reference_level):
            level = extract_reference_level(state)
            if isinstance(level, Real) and not isinstance(level, bool):
                try:
                    finite_level = float(level)
                except (OverflowError, TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(finite_level):
                        prepared_metadata["reference_level_db"] = finite_level
        return PreparedVoice(
            state=state,
            sample_rate=self.sample_rate,
            source=reference.source,
            model_id=self.model_id,
            source_revision=self.source_revision,
            backend=self.backend,
            runtime_fingerprint=self._runtime_fingerprint,
            fingerprint=reference.source_fingerprint,
            metadata=prepared_metadata,
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
        output: SoproOutputConfig | None = None,
    ) -> SynthesisResult:
        self._ensure_open()
        self._validate_request(request)
        if config is not None and not isinstance(config, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        selected_config = config if config is not None else GenerationConfig()
        if output is not None and not isinstance(output, SoproOutputConfig):
            raise InvalidGenerationConfigError("output must be a SoproOutputConfig")
        selected_output = output if output is not None else SoproOutputConfig()
        if voice_level is not None and not isinstance(voice_level, VoiceLevelConfig):
            raise InvalidGenerationConfigError("voice_level must be a VoiceLevelConfig")
        selected_voice_level = voice_level if voice_level is not None else VoiceLevelConfig()
        self._validate_voice(voice)

        max_segment_chars = getattr(self._engine, "max_segment_chars", DEFAULT_MAX_SEGMENT_CHARS)
        if max_segment_chars is None:
            max_segment_chars = DEFAULT_MAX_SEGMENT_CHARS
        if type(max_segment_chars) is not int or max_segment_chars < 1:
            raise RuntimeCapabilityError("ONNXVoice must expose a positive Sopro max_segment_chars")
        text_segments = split_text(request.text, max_segment_chars)
        if not text_segments:
            raise EmptyTextError("text must not be empty or whitespace-only")

        segment_infer = getattr(self._engine, "synthesize_segment", None)
        if len(text_segments) > 1 and not callable(segment_infer):
            raise RuntimeCapabilityError(
                "Long-text Sopro synthesis requires ONNXVoice semantic prompt carry; "
                "upgrade ONNXVoice to a version exposing the segment inference contract"
            )

        raw_reference_level = voice.metadata.get("reference_level_db")
        reference_level_db: float | None = None
        if isinstance(raw_reference_level, Real) and not isinstance(raw_reference_level, bool):
            try:
                candidate_level = float(raw_reference_level)
            except (OverflowError, TypeError, ValueError):
                candidate_level = float("nan")
            if math.isfinite(candidate_level):
                reference_level_db = candidate_level
        if selected_output.mode == "native" and reference_level_db is None:
            raise RuntimeCapabilityError(
                "Native Sopro output requires finite prepared reference_level_db metadata"
            )

        kwargs: dict[str, Any] = selected_config.as_inference_kwargs()
        if request.language not in (None, "auto"):
            kwargs["lang"] = request.language
        kwargs["ref"] = voice.state

        segment_audio: list[np.ndarray] = []
        carry: tuple[int, ...] | None = None
        for index, text_segment in enumerate(text_segments):
            segment_kwargs = dict(kwargs)
            if carry is not None:
                segment_kwargs["prompt_tokens"] = carry
            try:
                if callable(segment_infer):
                    inferred = segment_infer(text_segment, **segment_kwargs)
                    raw_audio, semantic_tokens, prompt_limit = _segment_result_parts(inferred)
                else:
                    raw_audio = self._engine.synthesize(text_segment, **segment_kwargs)
                    semantic_tokens = None
                    prompt_limit = None
                segment_audio.append(_to_numpy(raw_audio))
            except SoproSynthError:
                raise
            except Exception as exc:
                raise ModelInferenceError("Sopro synthesis failed") from exc

            if index + 1 < len(text_segments):
                if semantic_tokens is None or prompt_limit is None:
                    raise RuntimeCapabilityError(
                        "Long-text Sopro synthesis requires ONNXVoice generated semantic tokens "
                        "and prompt_token_limit metadata; upgrade ONNXVoice"
                    )
                if semantic_tokens and prompt_limit > 0:
                    carry = tuple(semantic_tokens[-prompt_limit:])

        empty_audio_samples = (
            getattr(self._engine, "token_samples", 1024) if selected_output.mode == "native" else 0
        )
        audio, sopro_output_metadata = condition_sopro_output(
            segment_audio,
            sample_rate=self.sample_rate,
            mode=selected_output.mode,
            reference_level_db=reference_level_db,
            empty_audio_samples=empty_audio_samples,
        )
        sopro_output_metadata["text_segment_count"] = len(text_segments)
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
                "sopro_output_config": asdict(selected_output),
                "sopro_output": sopro_output_metadata,
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
        output: SoproOutputConfig | None = None,
    ) -> SynthesisResult:
        self._ensure_open()
        request = SynthesisRequest(id=id, text=text, language=language)
        if not request.text.strip():
            raise EmptyTextError("text must not be empty or whitespace-only")
        prepared = voice if isinstance(voice, PreparedVoice) else self.prepare_voice(voice)
        return self.synthesize(
            request, voice=prepared, config=config, voice_level=voice_level, output=output
        )

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
        audio = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelInferenceError("Backend returned an invalid audio value") from exc
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ModelInferenceError("Backend returned invalid mono audio")
    return np.ascontiguousarray(audio)


def _segment_result_parts(result: Any) -> tuple[Any, tuple[int, ...] | None, int | None]:
    raw_audio = getattr(result, "audio", result)
    semantic_tokens = getattr(result, "semantic_tokens", None)
    prompt_limit = getattr(result, "prompt_token_limit", None)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        if semantic_tokens is None:
            semantic_tokens = metadata.get("generated_semantic_tokens")
        if prompt_limit is None:
            prompt_limit = metadata.get("prompt_token_limit")

    if semantic_tokens is not None:
        if isinstance(semantic_tokens, np.ndarray):
            if semantic_tokens.ndim != 1:
                raise RuntimeCapabilityError(
                    "ONNXVoice generated semantic tokens must be one-dimensional"
                )
            semantic_tokens = semantic_tokens.tolist()
        if isinstance(semantic_tokens, (str, bytes)) or not isinstance(semantic_tokens, Sequence):
            raise RuntimeCapabilityError("ONNXVoice generated semantic tokens must be a sequence")
        if any(
            isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral) or token < 0
            for token in semantic_tokens
        ):
            raise RuntimeCapabilityError(
                "ONNXVoice generated semantic tokens must be non-negative integers"
            )
        semantic_tokens = tuple(int(token) for token in semantic_tokens)

    if prompt_limit is not None:
        if (
            isinstance(prompt_limit, (bool, np.bool_))
            or not isinstance(prompt_limit, Integral)
            or prompt_limit < 0
        ):
            raise RuntimeCapabilityError(
                "ONNXVoice prompt_token_limit must be a non-negative integer"
            )
        prompt_limit = int(prompt_limit)
    return raw_audio, semantic_tokens, prompt_limit
