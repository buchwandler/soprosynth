"""Smoke-test the installed wheel in isolated Python mode, outside the checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import soprosynth

required = (
    "REQUEST_API_VERSION",
    "RequestApiContract",
    "RuntimeCapabilities",
    "request_api_contract",
    "SynthesisRequest",
    "SynthesisResult",
    "GenerationConfig",
    "PreparedVoice",
    "VoiceLevelConfig",
    "SoproRuntime",
    "discover_bundles",
    "runtime_identity",
)
for name in required:
    assert hasattr(soprosynth, name), name
assert soprosynth.REQUEST_API_VERSION == 1
assert soprosynth.request_api_contract().version == 1
native_capabilities = soprosynth.SoproRuntime(object()).capabilities
onnx_capabilities = soprosynth.SoproRuntime(object(), backend="onnx").capabilities
assert native_capabilities.supports_streaming is True
assert native_capabilities.supports_request_measurement is False
assert onnx_capabilities.supports_streaming is False
assert onnx_capabilities.supports_request_measurement is True
assert soprosynth.request_api_contract().supports_streaming is True
assert soprosynth.request_api_contract().supports_request_measurement is True

checkout = Path(__file__).resolve().parents[1]
installed_package = Path(soprosynth.__file__).resolve()
assert checkout not in installed_package.parents, (checkout, installed_package)

soprosynth.discover_bundles()
soprosynth.runtime_identity()
assert not ({"sopro", "onnxvoice", "onnxruntime"} & sys.modules.keys())
