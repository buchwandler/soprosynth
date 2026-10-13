"""Inspect API capabilities and installed/runtime identities without model loading."""

from soprosynth import discover_bundles, request_api_contract, runtime_identity

print(request_api_contract())
print(discover_bundles())
print(runtime_identity())
