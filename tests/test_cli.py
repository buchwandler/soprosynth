import json
from pathlib import Path

import pytest

import soprosynth.__main__ as cli
from soprosynth import RuntimeDiagnostics, __version__


def test_info_reports_contract_and_runtime_identity_without_model(monkeypatch, capsys):
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"backend": "native", "revision": None})

    assert cli.main(["info"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__
    assert payload["api"]["version"] == 1
    assert payload["api"]["supports_request_measurement"] is True
    assert payload["runtime"]["revision"] is None


def test_discover_prints_json_serializable_immutable_descriptors(capsys):
    assert cli.main(["discover", "--backend", "native"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["metadata"]["backend"] == "native"


def test_speak_routes_backend_generation_and_voice_level_options(monkeypatch, capsys):
    calls = {}

    class FakeResult:
        def __init__(self):
            self.metadata = {
                "voice_level_application": {"mode": "calibrated", "source": "override"}
            }

        def save_wav(self, path):
            calls["saved"] = path
            return Path(path)

    class FakeRuntime:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["runtime"] = kwargs
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def synthesize_text(self, text, **kwargs):
            calls["synthesis"] = {"text": text, **kwargs}
            return FakeResult()

    monkeypatch.setattr(cli, "SoproRuntime", FakeRuntime)

    assert (
        cli.main(
            [
                "speak",
                "Hallo",
                "--ref",
                "reference.wav",
                "--out",
                "hello.wav",
                "--backend",
                "native",
                "--language",
                "de",
                "--top-k",
                "8",
                "--max-seconds",
                "4.5",
                "--voice-level",
                "calibrated",
                "--gain-db",
                "2.0",
            ]
        )
        == 0
    )

    assert calls["runtime"] == {
        "model": "samuel-vitorino/sopro-v2-turbo",
        "backend": "native",
        "device": "cpu",
    }
    synthesis = calls["synthesis"]
    assert synthesis["text"] == "Hallo"
    assert synthesis["id"] == "cli-speak"
    assert synthesis["voice"] == "reference.wav"
    assert synthesis["language"] == "de"
    assert synthesis["config"].top_k == 8
    assert synthesis["config"].max_seconds == pytest.approx(4.5)
    assert synthesis["voice_level"].mode == "calibrated"
    assert synthesis["voice_level"].gain_db == pytest.approx(2.0)
    assert calls["saved"] == "hello.wav"
    output = capsys.readouterr().out
    assert "hello.wav" in output
    assert '"source": "override"' in output


def test_gain_override_requires_calibrated_mode(monkeypatch):
    monkeypatch.setattr(
        cli.SoproRuntime,
        "from_pretrained",
        classmethod(lambda cls, **kwargs: pytest.fail("runtime must not be opened")),
    )

    with pytest.raises(SystemExit) as error:
        cli.main(["speak", "Hello", "--ref", "reference.wav", "--gain-db", "2"])
    assert error.value.code == 2


def test_inspect_is_labeled_experimental_and_only_prints_diagnostics(monkeypatch, capsys):
    class FakeRuntime:
        diagnostics = RuntimeDiagnostics(
            backend="onnx",
            model_id="sopro:model",
            sample_rate=24000,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(cli.SoproRuntime, "from_onnx_local", lambda *args, **kwargs: FakeRuntime())

    assert cli.main(["inspect", "./export"]) == 0

    output = capsys.readouterr().out
    assert "Experimental developer inspection only" in output
    assert '"model_id": "sopro:model"' in output
