import hashlib
import json
from pathlib import Path
import pytest
from soprosynth.onnx import _safe_manifest_path, open_onnx_bundle


def test_paths():
    assert _safe_manifest_path("wasm-uint8", "../common/reference.onnx") == "common/reference.onnx"
    for bad in ("../../escape", "https://example.org/x", "/etc/passwd", "../.."):
        with pytest.raises(ValueError):
            _safe_manifest_path("wasm-uint8", bad)


def test_open_local_uses_onnxvoice_api(tmp_path, monkeypatch):
    from soprosynth import onnx

    (tmp_path / "common").mkdir()
    data = b"example fixture, not an ONNX model"
    (tmp_path / "common" / "ref.onnx").write_bytes(data)
    (tmp_path / "common" / "weights.bin").write_bytes(data)
    (tmp_path / "tokenizer.json").write_text("{}")
    h = hashlib.sha256(data).hexdigest()
    manifest = {
        "format": 1,
        "sampleRate": 24000,
        "tokenizer": "tokenizer.json",
        "profiles": {
            "wasm-uint8": {
                "graphs": {
                    "reference": {"url": "../common/ref.onnx", "file": "ref.onnx", "sha256": h}
                },
                "shards": {
                    "reference": {
                        "url": "../common/weights.bin",
                        "file": "weights.bin",
                        "sha256": h,
                    }
                },
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    class FakeOnnxVoice:
        @staticmethod
        def open_local(**kwargs):
            return kwargs

    monkeypatch.setattr(onnx, "register_sopro_adapter", lambda: None)
    import onnxvoice

    monkeypatch.setattr(onnxvoice, "OnnxVoice", FakeOnnxVoice)
    result = open_onnx_bundle(tmp_path)
    assert result["system"] == "sopro" and "graph:reference" in result["files"]
    assert result["sample_rate"] == 24000
    (tmp_path / "common" / "ref.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        open_onnx_bundle(tmp_path)
