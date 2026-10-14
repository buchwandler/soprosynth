from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from soprosynth import (
    BundleNotFoundError,
    InvalidRequestError,
    RuntimeCapabilityError,
    SoproRuntime,
    SynthesisRequest,
    UnsupportedBundleError,
)
from soprosynth._onnxvoice import OnnxVoiceBackend, normalize_ref
from soprosynth.config import SoproOutputConfig


class FakeEngine:
    sample_rate = 24000
    providers = "cpu"
    manifest: ClassVar[dict[str, str]] = {"revision": "upstream-revision"}

    generation_defaults = SimpleNamespace(
        temperature=0.8,
        top_p=0.9,
        top_k=25,
        steps=32,
        max_seconds=30.0,
        min_seconds=0.0,
        ref_seconds=10.0,
    )
    fixed_steps = 32
    max_segment_chars = 300
    token_samples = 1024

    def __init__(self, installation=None):
        self.installation = installation or SimpleNamespace(
            ref="sopro:v2-turbo",
            metadata={"model": "sopro-v2-turbo", "selected_quality": "wasm-uint8"},
        )
        self.infer_args: tuple[tuple[int, ...], dict[str, Any]] | None = None
        self.closed = False
        self.reference_level = -21.25
        self.infer_metadata = {}

        self.run_args = None
        self.run_closed = False
        self.prepare_seconds = []

    def prepare_voice(self, waveform, sample_rate, *, seconds=None):
        assert sample_rate == 24000
        self.prepare_seconds.append(seconds)
        return {"reference_samples": len(waveform), "level_db": self.reference_level}

    def prepare_text(self, text, language=None):
        return SimpleNamespace(token_ids=(11, 12), language=language)

    def infer(self, token_ids, **kwargs):
        self.infer_args = (tuple(token_ids), kwargs)
        return SimpleNamespace(
            audio=np.array([0.1, -0.1], dtype=np.float32), metadata=self.infer_metadata
        )

    def begin_run(self, voice_state, *, language=None, **generation):
        self.run_args = (voice_state, language, generation)
        return FakeSoproRun(self, voice_state, language, generation)

    def diagnostics(self):
        session = SimpleNamespace(
            component="semantic",
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=("text",),
            outputs=("audio",),
        )
        return SimpleNamespace(sessions=(session,))

    def close(self):
        self.closed = True


class FakeSoproRun:
    def __init__(self, engine, voice_state, language, generation):
        self.engine = engine
        self.voice_state = voice_state
        self.language = language
        self.generation = generation

    def infer_segment(self, text, *, language=None):
        text_input = self.engine.prepare_text(text, language=language or self.language)
        return self.engine.infer(
            text_input.token_ids,
            voice_state=self.voice_state,
            language=text_input.language,
            **self.generation,
        )

    def close(self):
        self.engine.run_closed = True


class FakeManager:
    def __init__(self, engine=None):
        self.engine = engine or FakeEngine()
        self.install_args: tuple[str, dict[str, Any]] | None = None
        self.open_args: tuple[Any, dict[str, Any]] | None = None

    def install(self, ref, **kwargs):
        self.install_args = (ref, kwargs)
        return self.engine.installation

    def open(self, ref, **kwargs):
        self.open_args = (ref, kwargs)
        return self.engine


def test_normalize_ref_and_reject_ambiguous_refs():
    assert normalize_ref("sopro:v2-turbo") == "sopro:v2-turbo"
    assert normalize_ref("v2-turbo") == "sopro:v2-turbo"
    assert normalize_ref("samuel-vitorino/sopro-v2-turbo") == "sopro:v2-turbo"
    with pytest.raises(InvalidRequestError, match="Expected a Sopro"):
        normalize_ref("kokoro:model")


def test_managed_runtime_uses_onnxvoice_and_public_api(monkeypatch):
    import soprosynth._onnxvoice as boundary

    engine = FakeEngine()
    manager = FakeManager(engine)
    fake_api = SimpleNamespace(OnnxVoice=lambda: manager)
    monkeypatch.setattr(boundary, "_load_onnxvoice", lambda: fake_api)

    runtime = SoproRuntime.from_pretrained(
        model="sopro:v2-turbo",
        providers="cpu",
        provider_options={"CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}},
    )
    assert manager.install_args is not None
    assert manager.open_args is not None
    assert manager.install_args[0] == "sopro:v2-turbo"
    assert manager.install_args[1]["quality"] == "wasm-uint8"
    assert manager.open_args[1]["provider_options"]["CPUExecutionProvider"]

    voice = runtime.prepare_voice(
        (np.array([0.1, -0.1], dtype=np.float32), 24000), reference_seconds=4.5
    )
    assert voice.metadata["reference_level_db"] == -21.25
    assert engine.prepare_seconds == [4.5]
    assert voice.metadata["reference_seconds"] == 4.5
    backend = runtime._engine
    assert backend.generation_defaults is engine.generation_defaults
    assert backend.fixed_steps == engine.fixed_steps == 32
    assert backend.max_segment_chars == engine.max_segment_chars == 300
    assert backend.token_samples == engine.token_samples == 1024
    result = runtime.synthesize(
        SynthesisRequest("req-1", "Hello", language="en"),
        voice=voice,
        output=SoproOutputConfig(mode="raw"),
    )
    assert result.audio.tolist() == pytest.approx([0.1, -0.1])
    assert engine.infer_args is not None
    assert engine.infer_args[0] == (11, 12)
    assert engine.infer_args[1]["voice_state"] == {"reference_samples": 2, "level_db": -21.25}
    assert engine.infer_args[1]["language"] == "en"
    measured = runtime.measure_request(
        SynthesisRequest("measure", "Hello", language="en"), maximum=2
    )
    assert measured.amount == 2 and measured.fits is True
    assert runtime.diagnostics.providers_active == ("CPUExecutionProvider",)
    runtime.close()
    assert engine.closed


def test_sopro_run_adapter_forwards_generation_language_audio_and_close():
    engine = FakeEngine()
    backend = OnnxVoiceBackend(engine, profile="wasm-uint8", requested_providers="cpu")

    run = backend.begin_run({"voice": 1}, language="en", seed=123, temperature=0.8)
    audio = run.synthesize_segment("Hello")
    run.close()

    assert engine.run_args == ({"voice": 1}, "en", {"seed": 123, "temperature": 0.8})
    assert engine.infer_args is not None
    assert engine.infer_args[0] == (11, 12)
    assert engine.infer_args[1]["voice_state"] == {"voice": 1}
    assert engine.infer_args[1]["language"] == "en"
    assert audio.dtype == np.float32 and audio.flags.c_contiguous
    assert audio.tolist() == pytest.approx([0.1, -0.1])
    assert engine.run_closed is True


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (-21.5, -21.5),
        (np.float32(-18.25), -18.25),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (True, None),
        ("not-a-level", None),
        ([1.0], None),
        (None, None),
    ],
)
def test_reference_level_extraction_only_accepts_finite_scalars(level, expected):
    backend = OnnxVoiceBackend(FakeEngine(), profile="wasm-uint8", requested_providers="cpu")
    assert backend.reference_level_db({"level_db": level}) == expected
    assert backend.reference_level_db(SimpleNamespace(level_db=level)) == expected


@pytest.mark.parametrize("level", [float("nan"), float("inf"), True, "bad", [1.0], None])
def test_prepared_voice_excludes_invalid_reference_level_metadata(level):
    engine = FakeEngine()
    engine.reference_level = level
    backend = OnnxVoiceBackend(engine, profile="wasm-uint8", requested_providers="cpu")
    runtime = SoproRuntime(backend)

    voice = runtime.prepare_voice((np.array([0.1, -0.1], dtype=np.float32), 24000))

    assert "reference_level_db" not in voice.metadata
    assert voice.metadata["reference_seconds"] == 10.0
    runtime.close()


def test_onnxvoice_capability_errors_are_mapped(monkeypatch):
    import soprosynth._onnxvoice as boundary

    class CapabilityError(Exception):
        pass

    class Manager:
        def open(self, *args, **kwargs):
            raise CapabilityError("profile unsupported")

    fake_api = SimpleNamespace(OnnxVoice=Manager)
    monkeypatch.setattr(boundary, "_load_onnxvoice", lambda: fake_api)
    with pytest.raises(RuntimeCapabilityError, match="profile unsupported"):
        SoproRuntime.load("sopro:v2-turbo")


def test_load_opens_managed_model_without_install(monkeypatch):
    import soprosynth._onnxvoice as boundary

    manager = FakeManager()
    fake_api = SimpleNamespace(OnnxVoice=lambda: manager)
    monkeypatch.setattr(boundary, "_load_onnxvoice", lambda: fake_api)

    runtime = SoproRuntime.load("v2-turbo")
    assert manager.open_args is not None
    assert manager.install_args is None
    assert manager.open_args[0] == "sopro:v2-turbo"
    assert manager.open_args[1]["providers"] == "cpu"
    runtime.close()


def test_from_resolved_opens_installation_without_catalog_work(monkeypatch):
    import soprosynth._onnxvoice as boundary

    manager = FakeManager()
    fake_api = SimpleNamespace(OnnxVoice=lambda: manager)
    monkeypatch.setattr(boundary, "_load_onnxvoice", lambda: fake_api)
    installation = manager.engine.installation

    runtime = SoproRuntime.from_resolved(installation)
    assert manager.open_args is not None
    assert manager.install_args is None
    assert manager.open_args[0] is installation
    runtime.close()


def test_local_open_passes_builtin_sopro_files_and_profile(tmp_path, monkeypatch):
    import soprosynth._onnxvoice as boundary

    data = b"test fixture"
    (tmp_path / "common").mkdir()
    (tmp_path / "wasm-uint8").mkdir()
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    graph = tmp_path / "common" / "reference.onnx"
    shard = tmp_path / "common" / "weights.bin"
    graph.write_bytes(data)
    shard.write_bytes(data)
    manifest = {
        "format": 1,
        "model": "sopro-v2-turbo",
        "revision": "fixture-revision",
        "sampleRate": 24000,
        "tokenizer": "tokenizer.json",
        "profiles": {
            "wasm-uint8": {
                "graphs": {"reference": {"url": "../common/reference.onnx"}},
                "shards": {"reference": {"url": "../common/weights.bin"}},
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    engine = FakeEngine(
        SimpleNamespace(ref="sopro:local-manifest", metadata={"model": manifest["model"]})
    )
    captured = {}

    def open_local(**kwargs):
        captured.update(kwargs)
        return engine

    fake_api = SimpleNamespace(open_local=open_local)
    monkeypatch.setattr(boundary, "_load_onnxvoice", lambda: fake_api)

    runtime = SoproRuntime.from_onnx_local(tmp_path)
    assert captured["system"] == "sopro"
    assert captured["runtime"]["profile"] == "wasm-uint8"
    assert captured["files"]["graph:reference"] == graph.resolve()
    assert captured["files"]["shard:reference"] == shard.resolve()
    assert captured["files"]["manifest"] == (tmp_path / "manifest.json").resolve()
    runtime.close()


def test_missing_local_manifest_is_bundle_not_found(tmp_path):
    with pytest.raises(BundleNotFoundError, match="manifest not found"):
        SoproRuntime.from_onnx_local(tmp_path)


def test_local_export_path_escape_rejected(tmp_path):
    import soprosynth._onnxvoice as boundary

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "sampleRate": 24000,
                "tokenizer": "tokenizer.json",
                "profiles": {
                    "wasm-uint8": {
                        "graphs": {"reference": {"url": "../../outside.onnx"}},
                        "shards": {"reference": {"url": "weights.bin"}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedBundleError, match="escapes its bundle"):
        boundary._local_files(tmp_path, "wasm-uint8")


def test_unsupported_profile_and_provider_are_explicit():
    with pytest.raises(RuntimeCapabilityError, match="wasm-uint8"):
        SoproRuntime.load("sopro:v2-turbo", profile="webgpu-fp16")
    with pytest.raises(RuntimeCapabilityError, match="CPUExecutionProvider"):
        SoproRuntime.load("sopro:v2-turbo", providers="cuda")


def test_local_verification_cannot_be_silently_disabled():
    from soprosynth._onnxvoice import open_local

    with pytest.raises(RuntimeCapabilityError, match="unconditionally"):
        open_local("unused", verify=False)
