"""Public runtime orchestration with a lazy ONNXVoice inference boundary."""

from __future__ import annotations

import math
import uuid
from collections.abc import Generator, Iterator
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from time import perf_counter
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
from .postprocess import (
    DEFAULT_MAX_SEGMENT_CHARS,
    SoproStreamConditioner,
    condition_sopro_output,
    split_text,
)
from .types import (
    LANGUAGES,
    PreparedVoice,
    RequestMeasure,
    RuntimeCapabilities,
    RuntimeDiagnostics,
    SynthesisChunk,
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
        reference_seconds: float | None = None,
        resampler: Literal["linear", "audiosig"] | None = None,
    ) -> PreparedVoice:
        """Preserve the source rate by default; explicitly select a legacy model-rate resampler."""
        self._ensure_open()
        if reference_seconds is not None:
            if isinstance(reference_seconds, bool) or not isinstance(reference_seconds, Real):
                raise InvalidGenerationConfigError(
                    "reference_seconds must be a finite positive number"
                )
            try:
                reference_seconds = float(reference_seconds)
            except (OverflowError, TypeError, ValueError) as exc:
                raise InvalidGenerationConfigError(
                    "reference_seconds must be a finite positive number"
                ) from exc
            if not math.isfinite(reference_seconds) or reference_seconds <= 0:
                raise InvalidGenerationConfigError(
                    "reference_seconds must be a finite positive number"
                )
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
            state = self._engine.prepare_reference(reference, seconds=reference_seconds)
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro reference preparation failed") from exc

        defaults = getattr(self._engine, "generation_defaults", None)
        effective_reference_seconds = (
            reference_seconds
            if reference_seconds is not None
            else getattr(defaults, "ref_seconds", None)
        )
        prepared_metadata: dict[str, int | float] = {
            "reference_sample_rate": reference.source_sample_rate,
            "prepared_reference_sample_rate": reference.sample_rate,
        }
        if isinstance(effective_reference_seconds, Real) and not isinstance(
            effective_reference_seconds, bool
        ):
            duration = float(effective_reference_seconds)
            if math.isfinite(duration) and duration > 0:
                prepared_metadata["reference_seconds"] = duration
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

    def prepare_streaming(
        self,
        voice: PreparedVoice,
        *,
        config: GenerationConfig | None = None,
    ) -> None:
        """Ask ONNXVoice to prewarm opaque streaming state for a prepared voice."""
        self._ensure_open()
        if config is not None and not isinstance(config, GenerationConfig):
            raise InvalidGenerationConfigError("config must be a GenerationConfig")
        selected_config = config if config is not None else GenerationConfig()
        self._validate_voice(voice)
        fixed_steps = getattr(self._engine, "fixed_steps", None)
        if selected_config.steps is not None:
            if type(fixed_steps) is not int or fixed_steps < 1:
                raise RuntimeCapabilityError("ONNXVoice must expose a positive fixed_steps value")
            if selected_config.steps != fixed_steps:
                raise RuntimeCapabilityError(f"Sopro acoustic steps are fixed at {fixed_steps}")
        prepare = getattr(self._engine, "prepare_streaming", None)
        if not callable(prepare):
            raise RuntimeCapabilityError("ONNXVoice backend does not expose prepare_streaming")
        controls: dict[str, int] = {}
        if selected_config.seed is not None:
            controls["seed"] = selected_config.seed
        if selected_config.steps is not None:
            controls["steps"] = selected_config.steps
        try:
            prepare(voice.state, **controls)
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro streaming preparation failed") from exc

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

    def _effective_max_segment_chars(self, config: GenerationConfig | None = None) -> int:
        max_segment_chars = config.text_chunk_size if config is not None else None
        if max_segment_chars is None:
            max_segment_chars = getattr(
                self._engine, "max_segment_chars", DEFAULT_MAX_SEGMENT_CHARS
            )
        if max_segment_chars is None:
            max_segment_chars = DEFAULT_MAX_SEGMENT_CHARS
        if type(max_segment_chars) is not int or max_segment_chars < 1:
            raise RuntimeCapabilityError("ONNXVoice must expose a positive Sopro max_segment_chars")
        return max_segment_chars

    def _split_request_text(self, text: str, max_segment_chars: int) -> tuple[str, ...]:
        if type(max_segment_chars) is not int or max_segment_chars < 1:
            raise RuntimeCapabilityError("ONNXVoice must expose a positive Sopro max_segment_chars")
        segments = split_text(text, max_segment_chars)
        if not segments:
            raise EmptyTextError("text must not be empty or whitespace-only")
        if any(not segment or len(segment) > max_segment_chars for segment in segments):
            raise RuntimeCapabilityError(
                "Sopro text segmentation did not respect max_segment_chars"
            )
        return segments

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
        max_segment_chars = self._effective_max_segment_chars()
        self._split_request_text(request.text, max_segment_chars)
        return RequestMeasure(
            amount=amount,
            maximum=maximum,
            unit="tokens",
            fits=amount <= maximum if maximum is not None else True,
        )

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
        fixed_steps = getattr(self._engine, "fixed_steps", None)
        if selected_config.steps is not None:
            if type(fixed_steps) is not int or fixed_steps < 1:
                raise RuntimeCapabilityError("ONNXVoice must expose a positive fixed_steps value")
            if selected_config.steps != fixed_steps:
                raise RuntimeCapabilityError(f"Sopro acoustic steps are fixed at {fixed_steps}")

        max_segment_chars = self._effective_max_segment_chars(selected_config)
        text_segments = self._split_request_text(request.text, max_segment_chars)

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

        effective_language = request.language if request.language not in (None, "auto") else None
        try:
            run = self._engine.begin_run(
                voice.state,
                language=effective_language,
                **selected_config.as_inference_kwargs(),
            )
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro synthesis run creation failed") from exc

        segment_audio: list[np.ndarray] = []
        try:
            for text_segment in text_segments:
                try:
                    segment_audio.append(_to_numpy(run.synthesize_segment(text_segment)))
                except SoproSynthError:
                    raise
                except Exception as exc:
                    raise ModelInferenceError("Sopro synthesis failed") from exc
        finally:
            run.close()
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
        generation_defaults = getattr(self._engine, "generation_defaults", None)
        effective_generation = {
            "temperature": selected_config.temperature
            if selected_config.temperature is not None
            else getattr(generation_defaults, "temperature", None),
            "top_p": selected_config.top_p
            if selected_config.top_p is not None
            else getattr(generation_defaults, "top_p", None),
            "top_k": selected_config.top_k
            if selected_config.top_k is not None
            else getattr(generation_defaults, "top_k", None),
            "steps": selected_config.steps if selected_config.steps is not None else fixed_steps,
            "max_seconds": selected_config.max_seconds
            if selected_config.max_seconds is not None
            else getattr(generation_defaults, "max_seconds", None),
            "min_seconds": selected_config.min_seconds
            if selected_config.min_seconds is not None
            else getattr(generation_defaults, "min_seconds", None),
            "seed": selected_config.seed,
            "text_chunk_size": max_segment_chars,
            "reference_seconds": voice.metadata.get(
                "reference_seconds", getattr(generation_defaults, "ref_seconds", None)
            ),
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
                "effective_generation": effective_generation,
                "voice_level_config": asdict(selected_voice_level),
                "voice_level_application": asdict(voice_level_application),
                "runtime_identity": runtime_identity(self),
                "runtime_diagnostics": diagnostic_metadata,
            },
        )

    def stream(
        self,
        request: SynthesisRequest,
        *,
        voice: PreparedVoice,
        config: GenerationConfig | None = None,
        voice_level: VoiceLevelConfig | None = None,
        output: SoproOutputConfig | None = None,
    ) -> Generator[SynthesisChunk]:
        """Synchronously yield conditioned audio chunks without buffering the full request."""
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

        fixed_steps = getattr(self._engine, "fixed_steps", None)
        if selected_config.steps is not None:
            if type(fixed_steps) is not int or fixed_steps < 1:
                raise RuntimeCapabilityError("ONNXVoice must expose a positive fixed_steps value")
            if selected_config.steps != fixed_steps:
                raise RuntimeCapabilityError(f"Sopro acoustic steps are fixed at {fixed_steps}")
        max_segment_chars = self._effective_max_segment_chars(selected_config)
        text_segments = self._split_request_text(request.text, max_segment_chars)

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

        from .calibration import (
            apply_voice_level_calibration,
            apply_voice_level_gain,
            load_calibration_catalog,
            voice_level_gain_scale,
        )

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
        _, voice_level_application = apply_voice_level_calibration(
            np.empty(0, dtype=np.float32),
            selected_voice_level,
            identity=calibration_identity,
            catalog=calibration_catalog,
        )
        voice_level_scale = voice_level_gain_scale(voice_level_application.gain_db)
        conditioner = SoproStreamConditioner(
            sample_rate=self.sample_rate,
            mode=selected_output.mode,
            reference_level_db=reference_level_db,
        )
        effective_language = request.language if request.language not in (None, "auto") else None
        started = perf_counter()
        try:
            run = self._engine.begin_run(
                voice.state,
                language=effective_language,
                **selected_config.as_inference_kwargs(),
            )
        except SoproSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Sopro synthesis run creation failed") from exc

        pending: tuple[int, np.ndarray, int, bool] | None = None
        next_index = 0
        audio_samples_total = 0
        first_chunk_ms: float | None = None

        def candidates(
            audio_chunks: tuple[np.ndarray, ...],
            segment_index: int,
            segment_final: bool,
        ) -> Iterator[tuple[int, np.ndarray, int, bool]]:
            nonlocal audio_samples_total, first_chunk_ms, next_index
            for chunk_audio in audio_chunks:
                calibrated = apply_voice_level_gain(chunk_audio, voice_level_scale)
                if calibrated.size == 0:
                    continue
                if first_chunk_ms is None:
                    first_chunk_ms = (perf_counter() - started) * 1000.0
                audio_samples_total += calibrated.size
                candidate = (next_index, calibrated, segment_index, segment_final)
                next_index += 1
                yield candidate

        def make_chunk(
            candidate: tuple[int, np.ndarray, int, bool],
            *,
            is_final: bool,
            summary: dict[str, Any] | None = None,
        ) -> SynthesisChunk:
            index, chunk_audio, segment_index, segment_final = candidate
            metadata: dict[str, Any] = {
                "backend": self.backend,
                "model_id": self.model_id,
                "model_revision": self.source_revision,
                "language": request.language or "auto",
                "output_mode": selected_output.mode,
                "chunk_index": index,
                "segment_index": segment_index,
                "segment_final": segment_final,
                "stream_final": is_final,
                "voice_level_application": asdict(voice_level_application),
            }
            if summary is not None:
                metadata.update(summary)
            return SynthesisChunk(
                id=request.id,
                audio=chunk_audio,
                sample_rate=self.sample_rate,
                index=index,
                segment_index=segment_index,
                is_final=is_final,
                metadata=metadata,
            )

        try:
            for segment_index, text_segment in enumerate(text_segments):
                conditioner.begin_segment(
                    first=segment_index == 0,
                    last=segment_index == len(text_segments) - 1,
                )
                stream_segment = getattr(run, "stream_segment", None)
                if not callable(stream_segment):
                    raise RuntimeCapabilityError("ONNXVoice run does not expose stream_segment")
                segment_stream = stream_segment(text_segment, language=effective_language)
                try:
                    for chunk_audio, chunk_sample_rate, segment_end, _metadata in segment_stream:
                        if chunk_sample_rate != self.sample_rate:
                            raise RuntimeCapabilityError(
                                "ONNXVoice stream sample rate does not match the runtime"
                            )
                        for candidate in candidates(
                            conditioner.push(chunk_audio), segment_index, segment_end
                        ):
                            if pending is not None:
                                yield make_chunk(pending, is_final=False)
                            pending = candidate
                except SoproSynthError:
                    raise
                except Exception as exc:
                    raise ModelInferenceError("Sopro streaming failed") from exc
                finally:
                    close_stream = getattr(segment_stream, "close", None)
                    if callable(close_stream):
                        try:
                            close_stream()
                        except SoproSynthError:
                            raise
                        except Exception as exc:
                            raise ModelInferenceError("Sopro stream segment close failed") from exc
                for candidate in candidates(conditioner.finish_segment(), segment_index, True):
                    if pending is not None:
                        yield make_chunk(pending, is_final=False)
                    pending = candidate

            for candidate in candidates(conditioner.finish(), len(text_segments) - 1, True):
                if pending is not None:
                    yield make_chunk(pending, is_final=False)
                pending = candidate
            if pending is None:
                raise ModelInferenceError("Sopro streaming inference returned no audio")
        finally:
            try:
                run.close()
            except SoproSynthError:
                raise
            except Exception as exc:
                raise ModelInferenceError("Sopro synthesis run close failed") from exc

        total_stream_ms = (perf_counter() - started) * 1000.0
        summary = {
            "first_chunk_ms": first_chunk_ms,
            "total_stream_ms": total_stream_ms,
            "chunk_count": next_index,
            "segment_count": len(text_segments),
            "audio_samples_total": audio_samples_total,
        }
        yield make_chunk(pending, is_final=True, summary=summary)

    def stream_text(
        self,
        text: str,
        *,
        voice: PreparedVoice | str | Path | tuple[Any, int],
        id: str = "speech",
        language: str | None = None,
        config: GenerationConfig | None = None,
        voice_level: VoiceLevelConfig | None = None,
        output: SoproOutputConfig | None = None,
    ) -> Generator[SynthesisChunk]:
        """Create a request and stream it through the prepared voice."""
        self._ensure_open()
        request = SynthesisRequest(id=id, text=text, language=language)
        if not request.text.strip():
            raise EmptyTextError("text must not be empty or whitespace-only")
        prepared = voice if isinstance(voice, PreparedVoice) else self.prepare_voice(voice)
        return self.stream(
            request,
            voice=prepared,
            config=config,
            voice_level=voice_level,
            output=output,
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
            # Keep capability advertisement gated until the released-wheel real-model test passes.
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
    try:
        audio = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelInferenceError("Backend returned an invalid audio value") from exc
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ModelInferenceError("Backend returned invalid mono audio")
    return np.ascontiguousarray(audio)
