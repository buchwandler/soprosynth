class SoproError(Exception):
    """Base SoproSynth error."""


class BackendUnavailableError(SoproError):
    """Required optional runtime missing or unsupported."""


class OnnxPipelineNotImplementedError(SoproError, NotImplementedError):
    """The upstream ONNX graph orchestration is not ported to Python yet."""
