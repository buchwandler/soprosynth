"""JSON-safe transport conversion for public SoproSynth values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeAlias

from .types import DiscoveredBundle

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def to_json_value(value: Any) -> JsonValue:
    """Return a detached JSON-compatible representation of ``value``.

    Dataclasses are traversed field-by-field (never via ``dataclasses.asdict``),
    and arbitrary mappings such as ``MappingProxyType`` are copied to dicts.
    Unsupported values and non-string mapping keys are rejected instead of
    being silently coerced.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            result[key] = to_json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain non-finite floats")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def discovered_bundle_to_dict(bundle: DiscoveredBundle) -> dict[str, JsonValue]:
    """Convert one ``DiscoveredBundle`` to detached JSON-compatible fields."""
    if not isinstance(bundle, DiscoveredBundle):
        raise TypeError(f"bundle must be a DiscoveredBundle, got {type(bundle).__name__}")
    payload = to_json_value(bundle)
    assert isinstance(payload, dict)
    return payload
