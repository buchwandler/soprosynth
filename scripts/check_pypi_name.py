"""Fail closed on first publication; defer later ownership checks to PyPI OIDC."""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request


def check_name(package: str, *, mode: str) -> None:
    if mode == "subsequent":
        print(
            "Skipping the unclaimed-name check for a subsequent release; "
            "PyPI Trusted Publishing must authorize this repository/environment."
        )
        return
    if mode != "first":
        raise ValueError(f"unsupported publication mode: {mode!r}")

    request = urllib.request.Request(
        f"https://pypi.org/pypi/{package}/json",
        headers={"User-Agent": "soprosynth-release-name-check/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status == 200:
                raise SystemExit(
                    f"PyPI project {package!r} already exists; refusing first publication"
                )
            raise SystemExit(f"Unexpected PyPI response for {package!r}: HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"PyPI API currently reports {package!r} as unclaimed")
            return
        raise SystemExit(f"Could not verify PyPI project name: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach PyPI to verify project name: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="soprosynth")
    parser.add_argument("--mode", choices=("first", "subsequent"), default="first")
    args = parser.parse_args(argv)
    check_name(args.package, mode=args.mode)


if __name__ == "__main__":
    main()
