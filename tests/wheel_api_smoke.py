"""Smoke-test the installed wheel in isolated Python mode, outside the checkout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soprosynth

required = (
    "REQUEST_API_VERSION",
    "RequestApiContract",
    "SoproOutputConfig",
    "SoproOutputMode",
    "RuntimeCapabilities",
    "request_api_contract",
    "SynthesisRequest",
    "SynthesisChunk",
    "SynthesisResult",
    "GenerationConfig",
    "PreparedVoice",
    "VoiceLevelConfig",
    "SoproRuntime",
    "discover_bundles",
    "runtime_identity",
    "to_json_value",
    "discovered_bundle_to_dict",
)
for name in required:
    assert hasattr(soprosynth, name), name
assert callable(soprosynth.SoproRuntime.stream)
assert callable(soprosynth.SoproRuntime.stream_text)
assert callable(soprosynth.SoproRuntime.prepare_streaming)
assert soprosynth.REQUEST_API_VERSION == 1
contract = soprosynth.request_api_contract()
assert contract.version == 1
assert contract.supports_streaming is False
assert contract.supports_sopro_output_policy is True
assert contract.caller_owns_text_boundaries is False
capabilities = soprosynth.SoproRuntime(object()).capabilities
assert capabilities.backend == "onnx"
assert capabilities.supports_streaming is False
assert capabilities.supports_request_measurement is True
assert soprosynth.runtime_identity()["engine_version"]
bundle = soprosynth.DiscoveredBundle(
    id="wheel-smoke",
    ref="sopro:wheel-smoke",
    display_name="Wheel smoke",
    metadata={"catalog": {"profile": {"providers": ("CPUExecutionProvider",)}}},
)
bundle_payload = soprosynth.discovered_bundle_to_dict(bundle)
assert bundle_payload["metadata"]["catalog"]["profile"]["providers"] == ["CPUExecutionProvider"]
assert soprosynth.to_json_value(bundle) == bundle_payload
json.dumps(bundle_payload, allow_nan=False)

checkout = Path(__file__).resolve().parents[1]
installed_package = Path(soprosynth.__file__).resolve()
assert checkout not in installed_package.parents, (checkout, installed_package)
assert not ({"sopro", "onnxvoice", "onnxruntime", "torch"} & sys.modules.keys())
