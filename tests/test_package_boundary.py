import subprocess
import sys


def test_package_import_does_not_import_optional_runtimes():
    code = """
import sys
import soprosynth

soprosynth.request_api_contract()
soprosynth.runtime_identity()
assert soprosynth.REQUEST_API_VERSION == 1
assert "sopro" not in sys.modules
assert "torch" not in sys.modules
assert "onnxruntime" not in sys.modules
assert "onnxvoice" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
