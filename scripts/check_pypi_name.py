"""Fail closed unless the PyPI JSON API says the project name is unclaimed."""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="soprosynth")
    package = parser.parse_args().package
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{package}/json",
        headers={"User-Agent": "soprosynth-release-name-check/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status == 200:
                raise SystemExit(
                    f"PyPI project {package!r} already exists; refusing automatic publish"
                )
            raise SystemExit(f"Unexpected PyPI response for {package!r}: HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"PyPI API currently reports {package!r} as unclaimed")
            return
        raise SystemExit(f"Could not verify PyPI project name: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach PyPI to verify project name: {exc}") from exc


if __name__ == "__main__":
    main()
