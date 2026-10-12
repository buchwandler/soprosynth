"""ONNX graph loading bridge through the existing OnnxVoice public extension API.

No private OnnxVoice monkeypatching or reimplementation of the ONNX session abstraction.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from .errors import BackendUnavailableError, OnnxPipelineNotImplementedError
from onnxvoice.systems.base import SystemAdapter


def _safe_manifest_path(profile: str, name: str) -> str:
    # Manifest links are relative to the profile directory, including ../common.
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(("/", "\\"))
        or chr(92) in name
        or "://" in name
    ):
        raise ValueError(f"unsafe upstream artifact path: {name!r}")
    canonical = posixpath.normpath(posixpath.join(profile, name))
    if canonical in (".", "..") or canonical.startswith("../") or canonical.startswith("/"):
        raise ValueError(f"artifact escapes bundle root: {name!r}")
    return canonical


def _check_hash(path: Path, sha256: str | None) -> None:
    if sha256 is None:
        return
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    if digest.hexdigest().lower() != sha256.lower():
        raise ValueError(f"SHA256 mismatch: {path}")


class SoproOnnxAdapter(SystemAdapter):
    """OnnxVoice SystemAdapter with graph-level execution and typed diagnostics.

    The Sopro multi-stage generation loop is *not* implemented. ``run_graph`` is
    useful for porting one graph at a time against the upstream JavaScript runtime.
    """

    system = "sopro"

    def __init__(
        self,
        installation: Any,
        *,
        providers: Any = None,
        provider_options: Any = None,
        session_options: Any = None,
    ) -> None:
        from onnxvoice import OnnxSession

        super().__init__(
            installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        self.sample_rate = installation.sample_rate or 24000
        self._sessions: dict[str, Any] = {}
        self._graph_paths = {
            item.component: item.path
            for item in installation.artifacts
            if item.role == "graph" and item.component
        }
        if not self._graph_paths:
            raise ValueError("No Sopro ONNX graph artifacts in installation")
        self._session_args = dict(
            providers=providers, provider_options=provider_options, session_options=session_options
        )
        self._session_type = OnnxSession

    @property
    def graphs(self) -> tuple[str, ...]:
        return tuple(sorted(self._graph_paths))

    def session(self, graph: str) -> Any:
        if graph not in self._graph_paths:
            raise KeyError(f"Unknown Sopro graph: {graph!r}")
        if graph not in self._sessions:
            self._sessions[graph] = self._session_type(
                self._graph_paths[graph], component=graph, **self._session_args
            )
        return self._sessions[graph]

    def graph_specs(self, graph: str) -> dict[str, Any]:
        s = self.session(graph)
        return {
            "inputs": s.input_specs,
            "outputs": s.output_specs,
            "providers": s.resolved_providers,
        }

    def run_graph(self, graph: str, inputs: Mapping[str, Any]) -> list[Any]:
        """Execute one ONNX graph. Input tensors must match its real ONNX contract."""
        return self.session(graph).run(dict(inputs))

    def infer(self, token_ids: Any, **kwargs: Any) -> Any:
        raise OnnxPipelineNotImplementedError(
            "Sopro does not have a single token-id -> audio graph; port reference DSP, "
            "semantic decoding, acoustic conditioning, ODE and vocoder sequencing first."
        )

    def diagnostics(self) -> Any:
        # Reuse the public SystemAdapter diagnostics contract.
        return super().diagnostics()

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()

    def __enter__(self) -> SoproOnnxAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def register_sopro_adapter() -> None:
    """Register with OnnxVoice's documented public SystemAdapter registry."""
    try:
        from onnxvoice import register_adapter
    except ImportError as exc:
        raise BackendUnavailableError("onnxvoice is required") from exc
    register_adapter("sopro", SoproOnnxAdapter)


def open_onnx_bundle(
    directory: str | Path,
    *,
    profile: str = "wasm-uint8",
    providers: Any = "cpu",
    verify: bool = True,
) -> SoproOnnxAdapter:
    """Open an upstream-exported local bundle (manifest.json + tokenizer + ONNX graphs).

    ``directory`` follows upstream export layout: manifest.json, common/, wasm-uint8/,
    webgpu-fp16/. The exporter uses external weights in sibling shards. The
    Python ONNX Runtime provider is selected explicitly by the caller.
    """
    root = Path(directory).expanduser().resolve()
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if raw.get("format") != 1 or not isinstance(raw.get("profiles"), dict):
        raise ValueError("Not a supported Sopro upstream ONNX manifest (format=1)")
    if profile not in raw["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}; choices: {sorted(raw['profiles'])}")
    info = raw["profiles"][profile]
    graphs, shards = info.get("graphs"), info.get("shards")
    if not isinstance(graphs, dict) or not graphs or not isinstance(shards, dict):
        raise ValueError("Sopro manifest has no graph and shard definitions")
    files: dict[str, Path] = {"manifest": root / "manifest.json"}
    for name, item in graphs.items():
        rel = _safe_manifest_path(profile, item.get("url") or item["file"])
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify:
            _check_hash(path, item.get("sha256"))
        files[f"graph:{name}"] = path
    for name, item in shards.items():
        rel = _safe_manifest_path(profile, item.get("url") or item["file"])
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify:
            _check_hash(path, item.get("sha256"))
        files[f"shard:{name}"] = path
    tokenizer = root / str(raw["tokenizer"])
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    files["tokenizer"] = tokenizer
    register_sopro_adapter()
    try:
        from onnxvoice import OnnxVoice
    except ImportError as exc:
        raise BackendUnavailableError("onnxvoice not installed") from exc
    return OnnxVoice.open_local(
        system="sopro",
        files=files,
        providers=providers,
        sample_rate=raw["sampleRate"],
        runtime={"layout": "multi", "profile": profile, "revision": raw.get("revision")},
        metadata={"model": raw.get("model"), "onnx_profile": profile},
    )
