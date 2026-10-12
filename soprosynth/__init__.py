"""Small, explicit Sopro API. Importing this module never downloads weights."""

try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("soprosynth")
    except PackageNotFoundError:
        __version__ = "0.0.0"

from .api import AudioResult, GenerationConfig, PreparedVoice, SoproRuntime, SynthesisRequest
from .errors import SoproError, BackendUnavailableError, OnnxPipelineNotImplementedError
from .onnx import SoproOnnxAdapter, open_onnx_bundle, register_sopro_adapter

__all__ = [
    "__version__",
    "AudioResult",
    "GenerationConfig",
    "PreparedVoice",
    "SoproRuntime",
    "SynthesisRequest",
    "SoproError",
    "BackendUnavailableError",
    "OnnxPipelineNotImplementedError",
    "SoproOnnxAdapter",
    "open_onnx_bundle",
    "register_sopro_adapter",
]
