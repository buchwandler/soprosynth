"""Smoke-test the installed wheel in isolated Python mode, outside the checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import soprosynth

required = (
    "REQUEST_API_VERSION",
    "RequestApiContract",
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

checkout = Path(__file__).resolve().parents[1]
installed_package = Path(soprosynth.__file__).resolve()
assert checkout not in installed_package.parents, (checkout, installed_package)

soprosynth.discover_bundles()
soprosynth.runtime_identity()
assert not ({"sopro", "onnxvoice", "onnxruntime"} & sys.modules.keys())
