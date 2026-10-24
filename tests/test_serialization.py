"""Tests for the public immutable-object JSON transport contract."""

from __future__ import annotations

import json
import math
from collections import UserDict
from pathlib import Path

import pytest

from soprosynth import DiscoveredBundle, discovered_bundle_to_dict, to_json_value


def _bundle() -> DiscoveredBundle:
    return DiscoveredBundle(
        id="model-v1",
        ref="sopro:model-v1",
        display_name="Model v1",
        languages=("en", "de"),
        metadata={
            "catalog": {
                "profile": {
                    "precision": "uint8",
                    "providers": ("CPUExecutionProvider",),
                }
            }
        },
    )


def test_discovered_bundle_to_dict_serializes_nested_read_only_metadata() -> None:
    bundle = _bundle()

    assert type(bundle.metadata).__name__ == "mappingproxy"
    assert type(bundle.metadata["catalog"]).__name__ == "mappingproxy"
    assert type(bundle.metadata["catalog"]["profile"]).__name__ == "mappingproxy"

    payload = discovered_bundle_to_dict(bundle)
    json.dumps(payload, allow_nan=False)

    assert payload == {
        "id": "model-v1",
        "ref": "sopro:model-v1",
        "display_name": "Model v1",
        "aliases": [],
        "languages": ["en", "de"],
        "sample_rate": None,
        "profiles": [],
        "source_revision": None,
        "source_commit": None,
        "metadata": {
            "catalog": {
                "profile": {
                    "precision": "uint8",
                    "providers": ["CPUExecutionProvider"],
                }
            }
        },
    }


def test_serialized_values_are_detached_and_mutable() -> None:
    bundle = _bundle()
    payload = discovered_bundle_to_dict(bundle)

    payload["metadata"]["catalog"]["profile"]["precision"] = "fp16"
    payload["metadata"]["catalog"]["profile"]["providers"].append("OtherProvider")
    payload["languages"].append("fr")

    assert bundle.metadata["catalog"]["profile"]["precision"] == "uint8"
    assert bundle.metadata["catalog"]["profile"]["providers"] == ("CPUExecutionProvider",)
    assert bundle.languages == ("en", "de")


def test_to_json_value_accepts_arbitrary_mappings_and_converts_paths() -> None:
    payload = to_json_value(
        UserDict({"nested": UserDict({"path": Path("/tmp/model"), "items": (1, 2)})})
    )

    assert payload == {"nested": {"path": "/tmp/model", "items": [1, 2]}}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_to_json_value_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        to_json_value({1: "not a JSON object key"})


def test_discovered_bundle_helper_validates_input_type() -> None:
    with pytest.raises(TypeError, match="must be a DiscoveredBundle"):
        discovered_bundle_to_dict({"id": "not a bundle"})  # type: ignore[arg-type]


def test_to_json_value_rejects_unsupported_values_without_stringifying() -> None:
    class Unsupported:
        pass

    with pytest.raises(TypeError, match="Unsupported.*not JSON-compatible"):
        to_json_value(Unsupported())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_to_json_value_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        to_json_value(value)


def test_to_json_value_preserves_json_scalars() -> None:
    value = {"none": None, "boolean": False, "integer": 7, "float": 1.25, "string": "ok"}

    assert to_json_value(value) == value
