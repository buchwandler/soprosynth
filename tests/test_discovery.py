from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import Any, cast

import pytest

from soprosynth import (
    CatalogUnavailableError,
    DiscoveredBundle,
    __version__,
    discover_bundles,
    runtime_identity,
)


def test_discovery_has_no_backend_selector():
    assert "backend" not in inspect.signature(discover_bundles).parameters


def test_onnx_discovery_wraps_catalog_without_opening_a_runtime(monkeypatch, tmp_path):
    import soprosynth._onnxvoice as boundary

    item = SimpleNamespace(
        id="v2-turbo",
        ref="sopro:v2-turbo",
        aliases=("sopro-v2-turbo",),
        sample_rate=24000,
        metadata={
            "display_name": "Sopro v2 Turbo",
            "languages": ["en", "pt"],
            "source_revision": "model-rev",
            "source_commit": "source-commit",
            "catalog_revision": "catalog-rev",
            "profiles": {"wasm-uint8": {}, "webgpu-fp16": {}},
        },
        artifacts=(SimpleNamespace(quality="wasm-uint8"),),
    )
    calls: dict[str, Any] = {}

    class CatalogClient:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def list(self, system, **kwargs):
            calls["list"] = (system, kwargs)
            if kwargs["progress"] is not None:
                kwargs["progress"](SimpleNamespace(phase="catalog_cached", ref=system))
            return [item]

    monkeypatch.setattr(
        boundary,
        "_load_onnxvoice",
        lambda: SimpleNamespace(CatalogClient=CatalogClient),
    )
    progress = []
    bundles = discover_bundles(
        cache_dir=tmp_path,
        offline=True,
        refresh=True,
        progress=progress.append,
    )
    assert calls["init"] == {"cache_dir": tmp_path, "offline": True}
    assert calls["list"][0] == "sopro"
    assert calls["list"][1]["refresh"] is True
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.profiles == ("wasm-uint8", "webgpu-fp16")
    assert bundle.source_revision == "model-rev"
    assert bundle.source_commit == "source-commit"
    assert progress[0]["phase"] == "catalog_cached"
    assert bundle.metadata["backend"] == "onnx"
    with pytest.raises(TypeError):
        cast(Any, bundle.metadata)["mutate"] = True


def test_discovery_catalog_errors_are_mapped(monkeypatch):
    import soprosynth._onnxvoice as boundary

    class CatalogError(Exception):
        pass

    class CatalogClient:
        def __init__(self, **kwargs):
            pass

        def list(self, *args, **kwargs):
            raise CatalogError("catalog is unavailable")

    monkeypatch.setattr(
        boundary,
        "_load_onnxvoice",
        lambda: SimpleNamespace(CatalogClient=CatalogClient),
    )
    with pytest.raises(CatalogUnavailableError, match="catalog is unavailable"):
        discover_bundles()


def test_runtime_identity_is_stable_and_uses_exact_bundle_revision(monkeypatch):
    from soprosynth import discovery

    def fake_version(name: str) -> str:
        if name == "soprosynth":
            return "0.1.0.dev1"
        if name == "onnxvoice":
            raise PackageNotFoundError(name)
        raise AssertionError(name)

    monkeypatch.setattr(discovery, "version", fake_version)
    bundle = DiscoveredBundle(
        id="v2-turbo",
        ref="sopro:v2-turbo",
        display_name="Sopro v2 Turbo",
        aliases=(),
        languages=("en",),
        sample_rate=24000,
        profiles=("wasm-uint8",),
        source_revision="model-rev",
        source_commit="source-commit",
        metadata={"catalog_revision": "catalog-rev"},
    )
    result = runtime_identity(bundle)
    assert result == {
        "engine_version": __version__,
        "onnxvoice_version": None,
        "request_api_version": "1",
        "catalog_revision": "catalog-rev",
        "bundle_revision": "source-commit",
    }
    assert runtime_identity()["bundle_revision"] is None
