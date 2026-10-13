import subprocess
import sys


def test_package_import_does_not_import_optional_runtimes():
    code = """
import sys
import soprosynth
soprosynth.discover_bundles()
soprosynth.runtime_identity()
assert soprosynth.REQUEST_API_VERSION == 1
assert not ({'sopro', 'onnxvoice', 'onnxruntime'} & sys.modules.keys())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
