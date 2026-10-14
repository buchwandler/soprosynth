"""Lazy boundary for ONNXVoice's built-in Sopro implementation."""

from __future__ import annotations

import importlib
import json
import math
import posixpath
from collections.abc import Callable, Iterator, Mapping, Sequence
from numbers import Real
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import (
    AssetAccessError,
    AssetDownloadError,
    AssetError,
    BundleNotFoundError,
    CatalogUnavailableError,
    InvalidRequestError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    RuntimeCapabilityError,
    SessionCreationError,
    SoproSynthError,
    UnsupportedBundleError,
)
from .types import RuntimeDiagnostics as PublicRuntimeDiagnostics
from .voice import ReferenceAudio

_PROGRESS_FIELDS = (
    "phase",
    "ref",
    "artifact",
    "completed",
    "total",
    "message",
    "role",
    "target",
)


def normalize_ref(ref: str) -> str:
    """Normalize a catalog model name to ONNXVoice's canonical ``sopro:`` form."""
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidRequestError("ONNX Sopro model reference must be a non-empty string")
    value = ref.strip()
    if value == "samuel-vitorino/sopro-v2-turbo" or value == "sopro-v2-turbo":
        return "sopro:v2-turbo"
    if value.startswith("sopro:"):
        item_id = value.removeprefix("sopro:")
        if not item_id or ":" in item_id or "/" in item_id or "\\" in item_id:
            raise InvalidRequestError(f"Invalid ONNX Sopro reference: {ref!r}")
        return f"sopro:{item_id}"
    if ":" in value or "/" in value or "\\" in value:
        raise InvalidRequestError(f"Expected a Sopro catalog reference, got {ref!r}")
    return f"sopro:{value}"


def _supported_profile_and_provider(
    profile: str,
    providers: str | Sequence[str] | None,
) -> str | tuple[str, ...]:
    if profile != "wasm-uint8":
        raise RuntimeCapabilityError(
            "Python Sopro inference is parity-gated only for profile 'wasm-uint8'"
        )
    if providers is None:
        selected: str | tuple[str, ...] = "cpu"
    elif isinstance(providers, str):
        selected = providers
    elif isinstance(providers, Sequence) and not isinstance(providers, (bytes, bytearray)):
        selected = tuple(providers)
    else:
        raise RuntimeCapabilityError("providers must select CPUExecutionProvider")
    names = (selected,) if isinstance(selected, str) else tuple(selected)
    if names not in (("cpu",), ("CPUExecutionProvider",)):
        raise RuntimeCapabilityError(
            "Python Sopro inference is parity-gated only for CPUExecutionProvider"
        )
    return selected


def _load_onnxvoice() -> Any:
    try:
        return importlib.import_module("onnxvoice")
    except ImportError as exc:
        raise OptionalDependencyError("Install soprosynth[cpu] / onnxvoice[cpu]") from exc


def _progress_adapter(callback: Callable[[Mapping[str, Any]], Any] | None):
    if callback is None:
        return None

    def report(event: Any) -> None:
        if isinstance(event, Mapping):
            values = {str(key): value for key, value in event.items() if isinstance(key, str)}
        else:
            values = {
                name: getattr(event, name) for name in _PROGRESS_FIELDS if hasattr(event, name)
            }
        callback(MappingProxyType(values))

    return report


def _map_backend_error(exc: Exception, operation: str) -> SoproSynthError:
    if isinstance(exc, SoproSynthError):
        return exc
    names = {cls.__name__ for cls in type(exc).__mro__}
    message = str(exc) or f"ONNXVoice {operation} failed"
    if "OptionalDependencyError" in names:
        return OptionalDependencyError(message)
    if "CatalogError" in names:
        return CatalogUnavailableError(message)
    if "OfflineError" in names:
        return OfflineAssetError(message)
    if "AssetAccessError" in names:
        return AssetAccessError(message)
    if "AssetDownloadError" in names:
        return AssetDownloadError(message)
    if "NotInstalledError" in names or "AssetNotFoundError" in names:
        return BundleNotFoundError(message)
    if "CapabilityError" in names or "RuntimeContractError" in names:
        return RuntimeCapabilityError(message)
    if "ManifestError" in names or "IntegrityError" in names:
        return UnsupportedBundleError(message)
    if operation == "open":
        return SessionCreationError(message)
    if operation == "install":
        return AssetError(message)
    if operation == "discover":
        return CatalogUnavailableError(message)
    return ModelInferenceError(message)


def install_and_open(
    ref: str,
    *,
    profile: str = "wasm-uint8",
    providers: str | Sequence[str] | None = "cpu",
    provider_options: Any = None,
    session_options: Any = None,
    force: bool = False,
    refresh: bool = False,
    progress: Callable[[Mapping[str, Any]], Any] | None = None,
) -> OnnxVoiceBackend:
    selected_providers = _supported_profile_and_provider(profile, providers)
    api = _load_onnxvoice()
    manager = api.OnnxVoice()
    try:
        installation = manager.install(
            normalize_ref(ref),
            quality=profile,
            force=force,
            refresh=refresh,
            progress=_progress_adapter(progress),
        )
    except Exception as exc:
        raise _map_backend_error(exc, "install") from exc
    return open_resolved(
        installation,
        profile=profile,
        providers=selected_providers,
        provider_options=provider_options,
        session_options=session_options,
        manager=manager,
    )


def open_managed(
    ref: str,
    *,
    profile: str = "wasm-uint8",
    providers: str | Sequence[str] | None = "cpu",
    provider_options: Any = None,
    session_options: Any = None,
) -> OnnxVoiceBackend:
    selected_providers = _supported_profile_and_provider(profile, providers)
    api = _load_onnxvoice()
    manager = api.OnnxVoice()
    try:
        engine = manager.open(
            normalize_ref(ref),
            quality=profile,
            providers=selected_providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except Exception as exc:
        raise _map_backend_error(exc, "open") from exc
    return OnnxVoiceBackend(engine, profile=profile, requested_providers=selected_providers)


def open_resolved(
    installation: Any,
    *,
    profile: str = "wasm-uint8",
    providers: str | Sequence[str] | None = "cpu",
    provider_options: Any = None,
    session_options: Any = None,
    manager: Any = None,
) -> OnnxVoiceBackend:
    installation_metadata = getattr(installation, "metadata", {}) or {}
    runtime_metadata = installation_metadata.get("runtime", {})
    selected_profile = (
        installation_metadata.get("selected_quality")
        or installation_metadata.get("onnx_profile")
        or (runtime_metadata.get("profile") if isinstance(runtime_metadata, Mapping) else None)
    )
    if selected_profile is not None and selected_profile != profile:
        raise RuntimeCapabilityError(
            f"Resolved ONNXVoice installation selected profile {selected_profile!r}, not {profile!r}"
        )
    selected_providers = _supported_profile_and_provider(profile, providers)
    api = _load_onnxvoice()
    manager = manager if manager is not None else api.OnnxVoice()
    try:
        engine = manager.open(
            installation,
            providers=selected_providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except Exception as exc:
        raise _map_backend_error(exc, "open") from exc
    return OnnxVoiceBackend(engine, profile=profile, requested_providers=selected_providers)


def _safe_bundle_file(root: Path, raw_path: Any, *, profile: str | None = None) -> Path:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\x00" in raw_path
        or "\\" in raw_path
        or ":" in raw_path
        or raw_path.startswith("/")
    ):
        raise UnsupportedBundleError("Unsafe local Sopro export path")
    normalized = posixpath.normpath(
        posixpath.join(profile, raw_path) if profile is not None else raw_path
    )
    parts = PurePosixPath(normalized).parts
    if not parts or normalized in {".", ".."} or normalized.startswith("../"):
        raise UnsupportedBundleError("Local Sopro export path escapes its bundle")
    resolved_root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsupportedBundleError("Local Sopro export path escapes its bundle") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _required_bundle_file(
    root: Path,
    raw_path: Any,
    *,
    profile: str | None = None,
    label: str,
) -> Path:
    try:
        return _safe_bundle_file(root, raw_path, profile=profile)
    except FileNotFoundError as exc:
        raise UnsupportedBundleError(f"Local Sopro export is missing {label}: {exc}") from exc


def _local_files(directory: str | Path, profile: str) -> tuple[dict[str, Path], dict[str, Any]]:
    root = Path(directory).expanduser().resolve()
    try:
        manifest_path = _safe_bundle_file(root, "manifest.json")
    except FileNotFoundError as exc:
        raise BundleNotFoundError(f"Local Sopro manifest not found in {root}") from exc
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnsupportedBundleError(
            f"Could not read Sopro export manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping) or type(manifest.get("format")) is not int:
        raise UnsupportedBundleError("Unsupported local Sopro export manifest")
    profiles = manifest.get("profiles")
    if manifest["format"] != 1 or not isinstance(profiles, Mapping) or profile not in profiles:
        raise UnsupportedBundleError(f"Local Sopro export does not contain profile {profile!r}")
    selected = profiles[profile]
    if not isinstance(selected, Mapping):
        raise UnsupportedBundleError(f"Invalid Sopro profile metadata for {profile!r}")
    graphs = selected.get("graphs")
    shards = selected.get("shards")
    if (
        not isinstance(graphs, Mapping)
        or not graphs
        or not isinstance(shards, Mapping)
        or not shards
    ):
        raise UnsupportedBundleError(
            "Local Sopro profile must declare graphs and external-data shards"
        )
    files: dict[str, Path] = {
        "manifest": manifest_path,
        "tokenizer": _required_bundle_file(root, manifest.get("tokenizer"), label="tokenizer"),
    }
    for role, declarations in (("graph", graphs), ("shard", shards)):
        for component, declaration in declarations.items():
            if (
                not isinstance(component, str)
                or not component
                or component in {".", ".."}
                or "/" in component
                or "\\" in component
                or not isinstance(declaration, Mapping)
            ):
                raise UnsupportedBundleError(f"Invalid local Sopro {role} declaration")
            raw_path = declaration.get("url") or declaration.get("file")
            files[f"{role}:{component}"] = _required_bundle_file(
                root,
                raw_path,
                profile=profile,
                label=f"{role} {component}",
            )
    metadata = {
        "model": manifest.get("model"),
        "onnx_profile": profile,
        "source_revision": manifest.get("revision"),
    }
    return files, {"manifest": manifest, "metadata": metadata, "root": root}


def open_local(
    directory: str | Path,
    *,
    profile: str = "wasm-uint8",
    providers: str | Sequence[str] | None = "cpu",
    provider_options: Any = None,
    session_options: Any = None,
    verify: bool = True,
) -> OnnxVoiceBackend:
    selected_providers = _supported_profile_and_provider(profile, providers)
    if verify is not True:
        raise RuntimeCapabilityError(
            "ONNXVoice validates local Sopro bundle integrity unconditionally"
        )
    files, info = _local_files(directory, profile)
    manifest = info["manifest"]
    sample_rate = manifest.get("sampleRate")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise UnsupportedBundleError("Local Sopro export must declare a positive sampleRate")
    if sample_rate != 24000:
        raise RuntimeCapabilityError("Python Sopro ONNX inference requires a 24-kHz profile")
    api = _load_onnxvoice()
    try:
        engine = api.open_local(
            system="sopro",
            files=files,
            sample_rate=sample_rate,
            runtime={"layout": "multi", "profile": profile, "revision": manifest.get("revision")},
            metadata=info["metadata"],
            providers=selected_providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except Exception as exc:
        raise _map_backend_error(exc, "open") from exc
    return OnnxVoiceBackend(engine, profile=profile, requested_providers=selected_providers)


def _validated_audio(value: Any) -> np.ndarray:
    try:
        audio = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelInferenceError("ONNXVoice returned invalid Sopro audio") from exc
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ModelInferenceError("ONNXVoice returned invalid mono Sopro audio")
    return np.ascontiguousarray(audio)


class OnnxVoiceRun:
    """Thin adapter for an ONNXVoice-owned sequential Sopro synthesis run."""

    def __init__(self, run: Any) -> None:
        self._run = run
        self._closed = False

    def synthesize_segment(self, text: str, *, language: str | None = None) -> np.ndarray:
        try:
            result = self._run.infer_segment(text, language=language)
            audio = result.audio
        except Exception as exc:
            raise _map_backend_error(exc, "inference") from exc
        return _validated_audio(audio)

    def stream_segment(
        self, text: str, *, language: str | None = None
    ) -> Iterator[tuple[np.ndarray, int, bool, Mapping[str, Any]]]:
        try:
            stream = self._run.stream_segment(text, language=language)
        except Exception as exc:
            raise _map_backend_error(exc, "inference") from exc
        try:
            for chunk in stream:
                try:
                    audio = _validated_audio(chunk.audio)
                    sample_rate = chunk.sample_rate
                    is_final = chunk.is_final
                    metadata = chunk.metadata
                    if type(sample_rate) is not int or sample_rate <= 0:
                        raise RuntimeCapabilityError("ONNXVoice stream sample rate is invalid")
                    if type(is_final) is not bool or not isinstance(metadata, Mapping):
                        raise RuntimeCapabilityError(
                            "ONNXVoice stream chunk metadata/finality is invalid"
                        )
                except SoproSynthError:
                    raise
                except Exception as exc:
                    raise _map_backend_error(exc, "inference") from exc
                yield audio, sample_rate, is_final, metadata
        except SoproSynthError:
            raise
        except Exception as exc:
            raise _map_backend_error(exc, "inference") from exc
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    raise _map_backend_error(exc, "close") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._run.close()
        except Exception as exc:
            raise _map_backend_error(exc, "close") from exc


class OnnxVoiceBackend:
    """Adapt ONNXVoice's Sopro runtime and run protocols to the stable SoproSynth boundary."""

    def __init__(
        self,
        engine: Any,
        *,
        profile: str,
        requested_providers: str | Sequence[str] | None,
    ) -> None:
        self._engine = engine
        self.installation = getattr(engine, "installation", None)
        self.profile = profile
        self.sample_rate = int(getattr(engine, "sample_rate", 24000))
        self.requested_providers = requested_providers
        self.providers = getattr(engine, "providers", requested_providers)
        installation_metadata = getattr(self.installation, "metadata", {}) or {}
        self.model_id = str(
            installation_metadata.get("model")
            or installation_metadata.get("model_id")
            or getattr(self.installation, "ref", None)
            or "sopro:v2-turbo"
        )
        manifest = getattr(engine, "manifest", {}) or {}
        self.source_revision = (
            manifest.get("revision")
            or installation_metadata.get("source_commit")
            or installation_metadata.get("source_revision")
        )

    @property
    def generation_defaults(self) -> Any:
        """Return ONNXVoice's normalized Sopro generation defaults."""
        try:
            return self._engine.generation_defaults
        except Exception as exc:
            raise _map_backend_error(exc, "generation defaults") from exc

    @property
    def fixed_steps(self) -> int:
        try:
            return self._engine.fixed_steps
        except Exception as exc:
            raise _map_backend_error(exc, "generation capabilities") from exc

    @property
    def max_segment_chars(self) -> int:
        try:
            return self._engine.max_segment_chars
        except Exception as exc:
            raise _map_backend_error(exc, "generation capabilities") from exc

    @property
    def token_samples(self) -> int:
        try:
            return self._engine.token_samples
        except Exception as exc:
            raise _map_backend_error(exc, "generation capabilities") from exc

    def prepare_reference(self, reference: ReferenceAudio, *, seconds: float | None = None) -> Any:
        try:
            return self._engine.prepare_voice(
                reference.audio,
                sample_rate=reference.sample_rate,
                seconds=seconds,
            )
        except Exception as exc:
            raise _map_backend_error(exc, "prepare") from exc

    def prepare_streaming(
        self,
        state: Any,
        *,
        seed: int | None = None,
        steps: int | None = None,
    ) -> None:
        """Forward streaming prewarm controls without exposing ONNXVoice cache state."""
        controls: dict[str, int] = {}
        if seed is not None:
            controls["seed"] = seed
        if steps is not None:
            controls["steps"] = steps
        try:
            self._engine.prepare_streaming(state, **controls)
        except Exception as exc:
            raise _map_backend_error(exc, "prepare") from exc

    @staticmethod
    def reference_level_db(state: Any) -> float | None:
        """Return only a finite scalar level from an ONNXVoice reference state."""
        if isinstance(state, Mapping):
            value = state.get("level_db")
        else:
            value = getattr(state, "level_db", None)
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        try:
            level_db = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return level_db if math.isfinite(level_db) else None

    def measure_text(self, text: str, language: str | None = None) -> int:
        try:
            prepared = self._engine.prepare_text(text, language=language)
        except Exception as exc:
            raise _map_backend_error(exc, "measure") from exc
        return len(prepared.token_ids)

    def begin_run(
        self,
        state: Any,
        *,
        language: str | None = None,
        **generation: Any,
    ) -> OnnxVoiceRun:
        try:
            run = self._engine.begin_run(state, language=language, **generation)
        except Exception as exc:
            raise _map_backend_error(exc, "inference") from exc
        return OnnxVoiceRun(run)

    def diagnostics(self) -> PublicRuntimeDiagnostics:
        try:
            raw = self._engine.diagnostics()
        except Exception as exc:
            raise _map_backend_error(exc, "diagnostics") from exc
        sessions: list[Mapping[str, Any]] = []
        requested: list[str] = []
        active: list[str] = []
        for session in getattr(raw, "sessions", ()):
            requested.extend(getattr(session, "providers_requested", ()))
            active.extend(getattr(session, "providers_active", ()))
            sessions.append(
                {
                    "component": getattr(session, "component", None),
                    "providers_requested": tuple(getattr(session, "providers_requested", ())),
                    "providers_active": tuple(getattr(session, "providers_active", ())),
                    "inputs": tuple(getattr(session, "inputs", ())),
                    "outputs": tuple(getattr(session, "outputs", ())),
                }
            )
        requested_values = tuple(dict.fromkeys(requested)) or _provider_tuple(
            self.requested_providers
        )
        active_values = tuple(dict.fromkeys(active))
        return PublicRuntimeDiagnostics(
            backend="onnx",
            model_id=self.model_id,
            sample_rate=self.sample_rate,
            source_revision=self.source_revision,
            device=active_values[0] if active_values else None,
            providers_requested=requested_values,
            providers_active=active_values,
            sessions=tuple(sessions),
        )

    def close(self) -> None:
        self._engine.close()


def _provider_tuple(providers: str | Sequence[str] | None) -> tuple[str, ...]:
    if providers is None:
        return ()
    return (providers,) if isinstance(providers, str) else tuple(providers)
