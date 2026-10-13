import tarfile
from io import BytesIO

from scripts.check_release_artifacts import _sdist_version


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
