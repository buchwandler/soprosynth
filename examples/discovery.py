"""Inspect the ONNXVoice Sopro catalog and runtime identity.

Discovery may access ONNXVoice catalog metadata but does not open a synthesis
runtime. Model inference requires ``soprosynth[cpu]``.
"""

from soprosynth import discover_bundles, request_api_contract, runtime_identity

print(request_api_contract())
print(discover_bundles())
print(runtime_identity())
