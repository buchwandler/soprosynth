from __future__ import annotations

import os
import subprocess
import sys
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    output = tmp_path_factory.mktemp("wheel")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "wheel build failed\n"
            f"command: {completed.args!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    wheels = list(output.glob("soprosynth-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_has_core_audiosig_and_no_upstream_sopro_requirement(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_paths) == 1
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))

    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    assert all(requirement.name.lower() != "sopro" for requirement in requirements)

    onnxvoice = [
        requirement for requirement in requirements if requirement.name.lower() == "onnxvoice"
    ]
    assert len(onnxvoice) == 1
    assert onnxvoice[0].extras == {"cpu"}
    assert onnxvoice[0].specifier.contains("0.2.7")
    assert not onnxvoice[0].specifier.contains("0.2.6")
    audiosig = [
        requirement for requirement in requirements if requirement.name.lower() == "audiosig"
    ]
    assert len(audiosig) == 1
    assert audiosig[0].marker is None
    assert audiosig[0].specifier.contains("0.1.8")
    assert not audiosig[0].specifier.contains("0.2")


def test_wheel_imports_without_upstream_sopro_or_inference_modules(built_wheel, tmp_path):
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(built_wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(python), "-I", str(ROOT / "tests" / "wheel_api_smoke.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
