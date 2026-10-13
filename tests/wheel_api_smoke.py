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
contract = soprosynth.request_api_contract()
assert contract.version == 1
assert contract.supports_streaming is False
capabilities = soprosynth.SoproRuntime(object()).capabilities
assert capabilities.backend == "onnx"
assert capabilities.supports_streaming is False
assert capabilities.supports_request_measurement is True
assert soprosynth.runtime_identity()["engine_version"]

checkout = Path(__file__).resolve().parents[1]
installed_package = Path(soprosynth.__file__).resolve()
assert checkout not in installed_package.parents, (checkout, installed_package)
assert not ({"sopro", "onnxvoice", "onnxruntime", "torch"} & sys.modules.keys())
