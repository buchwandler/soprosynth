"""Lazy model-catalog discovery and stable runtime identity helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .errors import InvalidRequestError, RuntimeCapabilityError, UnsupportedBundleError
from .types import DiscoveredBundle

_NATIVE_BUNDLE = DiscoveredBundle(
    id="sopro-v2-turbo",
    ref="samuel-vitorino/sopro-v2-turbo",
    display_name="Sopro v2 Turbo (native)",
    aliases=("sopro-v2-turbo",),
    languages=("en", "pt", "fr", "de"),
    sample_rate=24000,
    profiles=(),
    metadata={"backend": "native", "model_id": "samuel-vitorino/sopro-v2-turbo"},
)


def _optional_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _item_to_bundle(item: Any) -> DiscoveredBundle:
    raw_metadata = getattr(item, "metadata", {}) or {}
    if not isinstance(raw_metadata, Mapping):
        raw_metadata = {}
    metadata = dict(raw_metadata)
    metadata["backend"] = "onnx"
    profiles = raw_metadata.get("profiles")
    profile_names = (
        {name for name in profiles if isinstance(name, str) and name}
        if isinstance(profiles, Mapping)
        else set()
    )
    for artifact in getattr(item, "artifacts", ()):
        quality = getattr(artifact, "quality", None)
        if isinstance(quality, str) and quality:
            profile_names.add(quality)
    raw_languages = raw_metadata.get("languages")
    if isinstance(raw_languages, str):
        languages = (raw_languages,)
    elif isinstance(raw_languages, Sequence) and not isinstance(raw_languages, (bytes, bytearray)):
        languages = tuple(code for code in raw_languages if isinstance(code, str))
    else:
        language = raw_metadata.get("language")
        languages = (language,) if isinstance(language, str) else ()
    aliases = tuple(
        alias for alias in getattr(item, "aliases", ()) if isinstance(alias, str) and alias
    )
    item_id = getattr(item, "id", None)
    reference = getattr(item, "ref", None)
    if (
        not isinstance(item_id, str)
        or not item_id
        or not isinstance(reference, str)
        or not reference
    ):
        raise UnsupportedBundleError("ONNXVoice catalog item is missing its Sopro identity")
    display_name = raw_metadata.get("display_name") or raw_metadata.get("name") or item_id
    sample_rate = getattr(item, "sample_rate", None)
    if type(sample_rate) is not int or sample_rate <= 0:
        sample_rate = None
    return DiscoveredBundle(
        id=item_id,
        ref=reference,
        display_name=display_name if isinstance(display_name, str) and display_name else item_id,
        aliases=aliases,
        languages=tuple(code for code in languages if code),
        sample_rate=sample_rate,
        profiles=tuple(sorted(profile_names)),
        source_revision=(
            _optional_string(raw_metadata.get("source_revision"))
            or _optional_string(raw_metadata.get("revision"))
        ),
        source_commit=_optional_string(raw_metadata.get("source_commit")),
        metadata=metadata,
    )


def discover_bundles(
    *,
    backend: str = "native",
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    progress: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[DiscoveredBundle, ...]:
    """List native Sopro metadata or ONNXVoice's Sopro catalog without opening a model."""
    if not isinstance(backend, str) or backend not in {"native", "onnx", "all"}:
        raise RuntimeCapabilityError("discovery backend must be 'native', 'onnx', or 'all'")
    if type(offline) is not bool or type(refresh) is not bool:
        raise InvalidRequestError("offline and refresh must be bool values")
    results: list[DiscoveredBundle] = []
    if backend in {"native", "all"}:
        results.append(_NATIVE_BUNDLE)
    if backend in {"onnx", "all"}:
        from ._onnxvoice import _load_onnxvoice, _map_backend_error, _progress_adapter

        api = _load_onnxvoice()
        try:
            catalog = api.CatalogClient(cache_dir=cache_dir, offline=offline)
            items = catalog.list("sopro", refresh=refresh, progress=_progress_adapter(progress))
            results.extend(_item_to_bundle(item) for item in items)
        except Exception as exc:
            raise _map_backend_error(exc, "discover") from exc
    return tuple(results)


def runtime_identity(bundle: Any = None) -> dict[str, str | None]:
    """Return package/runtime and optional bundle revisions without loading model code."""
    from . import __version__
    from .api_contract import REQUEST_API_VERSION

    if isinstance(bundle, Mapping):
        metadata = bundle.get("metadata", bundle)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source_revision = bundle.get("source_revision")
        source_commit = bundle.get("source_commit")
    else:
        metadata = getattr(bundle, "metadata", {}) if bundle is not None else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source_revision = getattr(bundle, "source_revision", None) if bundle is not None else None
        source_commit = getattr(bundle, "source_commit", None) if bundle is not None else None
    if not isinstance(source_revision, str):
        source_revision = _optional_string(metadata.get("source_revision"))
    if not isinstance(source_commit, str):
        source_commit = _optional_string(metadata.get("source_commit"))
    catalog_revision = _optional_string(metadata.get("catalog_revision")) or _optional_string(
        metadata.get("catalog_commit")
    )
    bundle_revision = (
        source_commit or source_revision or _optional_string(metadata.get("manifest_revision"))
    )
    return {
        "engine_version": __version__,
        "native_runtime_revision": _optional_version("sopro"),
        "onnx_runtime_revision": _optional_version("onnxvoice"),
        "request_api_version": str(REQUEST_API_VERSION),
        "catalog_revision": catalog_revision,
        "bundle_revision": bundle_revision,
    }
