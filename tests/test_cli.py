import json
from pathlib import Path

import pytest

import soprosynth.__main__ as cli
from soprosynth import DiscoveredBundle, RuntimeDiagnostics, __version__


def test_info_reports_contract_and_runtime_identity_without_model(monkeypatch, capsys):
    identity = {"engine_version": "0.1.0", "onnxvoice_version": None}
    monkeypatch.setattr(cli, "runtime_identity", lambda: identity)

    assert cli.main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__
    assert payload["api"]["version"] == 1
    assert payload["api"]["supports_request_measurement"] is True
    assert payload["api"]["supports_sopro_output_policy"] is True
    assert payload["api"]["caller_owns_text_boundaries"] is False
    # Intentionally false until the opt-in released-wheel real-model integration gate passes.
    assert payload["api"]["supports_streaming"] is False
    assert payload["runtime"] == identity


def test_discover_prints_onnxvoice_catalog_without_backend_selector(monkeypatch, capsys):
    calls = []
    bundle = DiscoveredBundle(
        id="v2-turbo",
        ref="sopro:v2-turbo",
        display_name="Sopro v2 Turbo",
        metadata={"backend": "onnx"},
    )
    monkeypatch.setattr(cli, "discover_bundles", lambda **kwargs: calls.append(kwargs) or (bundle,))

    assert cli.main(["discover", "--offline", "--refresh"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == [{"offline": True, "refresh": True, "cache_dir": None}]
    assert payload[0]["metadata"]["backend"] == "onnx"
    assert "backend" not in vars(cli._build_parser().parse_args(["discover"]))


def test_speak_routes_canonical_runtime_options_and_voice_level(monkeypatch, capsys):
    calls = {}

    class FakeResult:
        def __init__(self):
            self.metadata = {
                "voice_level_application": {"mode": "calibrated", "source": "override"},
                "sopro_output": {"mode": "raw", "gain_db": None},
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
                "--language",
                "de",
                "--output-mode",
                "raw",
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
        "model": "sopro:v2-turbo",
        "profile": "wasm-uint8",
        "providers": "cpu",
    }
    synthesis = calls["synthesis"]
    assert synthesis["text"] == "Hallo"
    assert synthesis["id"] == "cli-speak"
    assert synthesis["voice"] == "reference.wav"
    assert synthesis["language"] == "de"
    assert synthesis["config"].top_k == 8
    assert synthesis["config"].max_seconds == pytest.approx(4.5)
    assert synthesis["output"].mode == "raw"
    assert synthesis["voice_level"].mode == "calibrated"
    assert synthesis["voice_level"].gain_db == pytest.approx(2.0)
    assert calls["saved"] == "hello.wav"
    output = capsys.readouterr().out
    assert "hello.wav" in output
    assert '"source": "override"' in output
    assert '"sopro_output"' in output and '"mode": "raw"' in output


def test_cli_has_no_backend_switch_and_uses_cpu_profile_defaults(capsys):
    parser = cli._build_parser()
    args = parser.parse_args(["speak", "hello", "--ref", "reference.wav"])
    assert args.model == "sopro:v2-turbo"
    assert args.profile == "wasm-uint8"
    assert args.provider == "cpu"
    assert args.output_mode == "native"

    with pytest.raises(SystemExit) as error:
        cli.main(["speak", "--help"])
    assert error.value.code == 0
    help_output = capsys.readouterr().out
    assert "--backend" not in help_output
    assert "--output-mode" in help_output


def test_gain_override_requires_calibrated_mode(monkeypatch):
    monkeypatch.setattr(
        cli.SoproRuntime,
        "from_pretrained",
        classmethod(lambda cls, **kwargs: pytest.fail("runtime must not be opened")),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["speak", "Hello", "--ref", "reference.wav", "--gain-db", "2"])
    assert error.value.code == 2


def test_inspect_is_developer_diagnostic_and_only_prints_diagnostics(monkeypatch, capsys):
    class FakeRuntime:
        diagnostics = RuntimeDiagnostics(
            backend="onnx",
            model_id="sopro:v2-turbo",
            sample_rate=24000,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(cli.SoproRuntime, "from_onnx_local", lambda *args, **kwargs: FakeRuntime())
    assert cli.main(["inspect", "./export"]) == 0
    output = capsys.readouterr().out
    assert "Developer inspection only" in output
    assert '"model_id": "sopro:v2-turbo"' in output
