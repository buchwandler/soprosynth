"""Inspect static native discovery and identities with the base installation only.

This does not load a speech model or import optional runtime packages.
"""

from soprosynth import discover_bundles, request_api_contract, runtime_identity

print(request_api_contract())
print(discover_bundles())
print(runtime_identity())
