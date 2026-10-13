"""Verify release tag, artifact filenames, and embedded distribution metadata agree."""

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


def _wheel_version(path: Path, *, expected: str | None = None) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise SystemExit(f"expected one wheel METADATA file in {path.name}")
        dist_info = PurePosixPath(metadata[0]).parts[0]
        if expected is not None and dist_info != f"soprosynth-{expected}.dist-info":
            raise SystemExit(
                f"wheel dist-info filename does not match release version: {dist_info}"
            )
        return _metadata_version(archive.read(metadata[0]).decode("utf-8"))


def _sdist_version(path: Path, *, expected: str | None = None) -> str:
    with tarfile.open(path, "r:gz") as archive:
        metadata = [
            member
            for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) == 2
            and PurePosixPath(member.name).name == "PKG-INFO"
        ]
        if len(metadata) != 1:
            raise SystemExit(f"expected one sdist PKG-INFO file in {path.name}")
        root = PurePosixPath(metadata[0].name).parts[0]
        if expected is not None and root != f"soprosynth-{expected}":
            raise SystemExit(f"sdist top-level directory does not match release version: {root}")
        stream = archive.extractfile(metadata[0])
        if stream is None:
            raise SystemExit(f"could not read PKG-INFO from {path.name}")
        return _metadata_version(stream.read().decode("utf-8"))


def verify_release_artifacts(tag: str, dist: Path) -> str:
    if not tag.startswith("v") or not tag[1:]:
        raise SystemExit("release tags must start with 'v' and include a version")
    expected = tag[1:]
    wheels = sorted(dist.glob("soprosynth-*.whl"))
    sdists = sorted(dist.glob("soprosynth-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one SoproSynth wheel and one sdist")
    wheel_prefix = f"soprosynth-{expected}-"
    sdist_name = f"soprosynth-{expected}.tar.gz"
    if not wheels[0].name.startswith(wheel_prefix) or not wheels[0].name.endswith(".whl"):
        raise SystemExit(f"wheel filename does not match release tag {tag!r}: {wheels[0].name}")
    if sdists[0].name != sdist_name:
        raise SystemExit(f"sdist filename does not match release tag {tag!r}: {sdists[0].name}")
    versions = {
        _wheel_version(wheels[0], expected=expected),
        _sdist_version(sdists[0], expected=expected),
    }
    if versions != {expected}:
        raise SystemExit(
            f"tag version {expected!r} does not match artifact versions {sorted(versions)!r}"
        )
    return expected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="release tag such as v0.1.0")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    expected = verify_release_artifacts(args.tag, args.dist)
    print(f"Release tag, filenames, and wheel/sdist metadata agree: {expected}")


if __name__ == "__main__":
    main()
