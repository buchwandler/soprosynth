"""Lazy adapter around the supported upstream native Sopro API."""

from __future__ import annotations

import importlib
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .audio import save_wav
from .errors import OptionalDependencyError, SessionCreationError
from .voice import ReferenceAudio


class NativeBackend:
    """Small native boundary; no Sopro imports occur until construction."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.sample_rate = int(getattr(engine, "sample_rate", 24000))

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        device: str,
        revision: str | None,
        cache_dir: str | None,
        quantization: str | None,
    ) -> NativeBackend:
        try:
            SoproTTS = importlib.import_module("sopro").SoproTTS
        except ImportError as exc:
            raise OptionalDependencyError(
                "Native synthesis requires the optional 'sopro>=2.2.0,<3' dependency"
            ) from exc
        try:
            engine = SoproTTS.from_pretrained(
                model,
                device=device,
                revision=revision,
                cache_dir=cache_dir,
                quantization=quantization,
            )
        except Exception as exc:
            raise SessionCreationError(f"Could not load native Sopro model {model!r}") from exc
        return cls(engine)

    def prepare_reference(self, reference: ReferenceAudio, *, stream: bool = False) -> Any:
        if reference.native_source is not None and reference.sample_rate == self.sample_rate:
            return self._engine.prepare_reference(
                ref_audio_path=Path(reference.native_source),
                stream=stream,
            )
        with tempfile.TemporaryDirectory(prefix="soprosynth-reference-") as directory:
            path = Path(directory) / "reference.wav"
            save_wav(reference.audio, reference.sample_rate, path)
            return self._engine.prepare_reference(ref_audio_path=path, stream=stream)

    def synthesize(self, text: str, **kwargs: Any) -> Any:
        return self._engine.synthesize(text, **kwargs)

    def stream(self, text: str, **kwargs: Any) -> Iterator[Any]:
        return iter(self._engine.stream(text, **kwargs))

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if close is not None:
            close()
