"""Verify that wheel and sdist metadata match the version encoded by a release tag."""

from __future__ import annotations

import argparse
import email.parser
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def _metadata_version(text: str) -> str:
    message = email.parser.Parser().parsestr(text)
    name = message.get("Name", "").lower().replace("_", "-")
    if name != "soprosynth":
        raise SystemExit(f"unexpected distribution name in metadata: {name!r}")
    version = message.get("Version")
    if not version:
        raise SystemExit("distribution metadata is missing Version")
    return version


def _wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise SystemExit(f"expected one wheel METADATA file in {path.name}")
        return _metadata_version(archive.read(metadata[0]).decode("utf-8"))


def _sdist_version(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        metadata = [
            member
            for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) == 2
            and PurePosixPath(member.name).name == "PKG-INFO"
        ]
        if len(metadata) != 1:
            raise SystemExit(f"expected one sdist PKG-INFO file in {path.name}")
        stream = archive.extractfile(metadata[0])
        if stream is None:
            raise SystemExit(f"could not read PKG-INFO from {path.name}")
        return _metadata_version(stream.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="release tag such as v0.1.0")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    if not args.tag.startswith("v"):
        raise SystemExit("release tags must start with 'v'")
    expected = args.tag[1:]
    wheels = sorted(args.dist.glob("soprosynth-*.whl"))
    sdists = sorted(args.dist.glob("soprosynth-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one SoproSynth wheel and one sdist")
    versions = {_wheel_version(wheels[0]), _sdist_version(sdists[0])}
    if versions != {expected}:
        raise SystemExit(
            f"tag version {expected!r} does not match artifact versions {sorted(versions)!r}"
        )
    print(f"Release tag and wheel/sdist metadata agree: {expected}")


if __name__ == "__main__":
    main()
