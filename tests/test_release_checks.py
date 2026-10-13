import tarfile
import urllib.error
import zipfile
from email.message import Message
from io import BytesIO

import pytest

from scripts.check_pypi_name import check_name
from scripts.check_release_artifacts import _sdist_version, verify_release_artifacts


def _make_wheel(
    directory,
    *,
    filename="soprosynth-0.1.0-py3-none-any.whl",
    dist_info_version="0.1.0",
    name="soprosynth",
    metadata_version="0.1.0",
):
    path = directory / filename
    metadata = (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {metadata_version}\n"
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"soprosynth-{dist_info_version}.dist-info/METADATA", metadata
        )
    return path


def _make_sdist(
    directory,
    *,
    filename="soprosynth-0.1.0.tar.gz",
    root="soprosynth-0.1.0",
    name="soprosynth",
    version="0.1.0",
):
    path = directory / filename
    payload = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n".encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(f"{root}/PKG-INFO")
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))
    return path


def test_sdist_version_uses_distribution_pkg_info_not_egg_info(tmp_path):
    archive_path = tmp_path / "soprosynth-0.1.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, version in (
            ("soprosynth-0.1.0/PKG-INFO", "0.1.0"),
            ("soprosynth-0.1.0/soprosynth.egg-info/PKG-INFO", "0.0.1.dev2"),
        ):
            payload = f"Metadata-Version: 2.4\nName: soprosynth\nVersion: {version}\n".encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))

    assert _sdist_version(archive_path) == "0.1.0"


def test_release_artifact_verifier_accepts_matching_filename_tag_and_metadata(tmp_path):
    _make_wheel(tmp_path)
    _make_sdist(tmp_path)

    assert verify_release_artifacts("v0.1.0", tmp_path) == "0.1.0"


def test_release_artifact_verifier_rejects_mismatched_wheel_filename(tmp_path):
    _make_wheel(tmp_path, filename="soprosynth-0.1.1-py3-none-any.whl")
    _make_sdist(tmp_path)

    with pytest.raises(SystemExit, match="wheel filename"):
        verify_release_artifacts("v0.1.0", tmp_path)


def test_release_artifact_verifier_rejects_metadata_or_sdist_root_mismatch(tmp_path):
    _make_wheel(tmp_path, metadata_version="0.1.1")
    _make_sdist(tmp_path, root="soprosynth-0.1.1")

    with pytest.raises(SystemExit, match="sdist top-level directory"):
        verify_release_artifacts("v0.1.0", tmp_path)



def test_release_artifact_verifier_rejects_embedded_metadata_version_mismatch(tmp_path):
    _make_wheel(tmp_path, metadata_version="0.1.1")
    _make_sdist(tmp_path)

    with pytest.raises(SystemExit, match="tag version"):
        verify_release_artifacts("v0.1.0", tmp_path)



def test_release_artifact_verifier_rejects_mismatched_wheel_dist_info(tmp_path):
    _make_wheel(tmp_path, dist_info_version="0.1.1")
    _make_sdist(tmp_path)

    with pytest.raises(SystemExit, match="wheel dist-info filename"):
        verify_release_artifacts("v0.1.0", tmp_path)


def test_first_publication_requires_unclaimed_name(monkeypatch):
    def http_404(request, timeout):
        assert request.full_url == "https://pypi.org/pypi/soprosynth/json"
        assert timeout == 15
        raise urllib.error.HTTPError(request.full_url, 404, "not found", Message(), None)

    monkeypatch.setattr("scripts.check_pypi_name.urllib.request.urlopen", http_404)
    check_name("soprosynth", mode="first")


def test_first_publication_refuses_existing_name(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return None

    def return_response(request, timeout):
        assert request.full_url.startswith("https://pypi.org/")
        assert timeout == 15
        return Response()

    monkeypatch.setattr("scripts.check_pypi_name.urllib.request.urlopen", return_response)
    with pytest.raises(SystemExit, match="refusing first publication"):
        check_name("soprosynth", mode="first")


def test_subsequent_publication_skips_unclaimed_name_lookup(monkeypatch):
    def should_not_call(request, timeout):
        pytest.fail(f"unexpected PyPI lookup for {request!r} with timeout {timeout}")

    monkeypatch.setattr("scripts.check_pypi_name.urllib.request.urlopen", should_not_call)
    check_name("soprosynth", mode="subsequent")
