"""Dependency-light public API for the Sopro speech-synthesis runtime."""

try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("soprosynth")
    except PackageNotFoundError:
        __version__ = "0.0.0"

from .api import SoproRuntime
from .api_contract import REQUEST_API_VERSION, RequestApiContract, request_api_contract
from .config import (
    GenerationConfig,
    VoiceLevelApplication,
    VoiceLevelConfig,
    VoiceLevelMode,
    VoiceLevelSource,
)
from .discovery import discover_bundles, runtime_identity
from .errors import (
    AssetAccessError,
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    BundleNotFoundError,
    CatalogUnavailableError,
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    RuntimeCapabilityError,
    RuntimeClosedError,
    SessionCreationError,
    SoproSynthError,
    UnsupportedBundleError,
    UnsupportedFeatureError,
)
from .types import (
    DiscoveredBundle,
    PreparedVoice,
    RequestMeasure,
    RuntimeCapabilities,
    RuntimeDiagnostics,
    SynthesisRequest,
    SynthesisResult,
    WordTiming,
)

__all__ = [
    "REQUEST_API_VERSION",
    "AssetAccessError",
    "AssetCacheError",
    "AssetDownloadError",
    "AssetError",
    "BundleNotFoundError",
    "CatalogUnavailableError",
    "DiscoveredBundle",
    "EmptyTextError",
    "GenerationConfig",
    "InvalidGenerationConfigError",
    "InvalidLanguageError",
    "InvalidRequestError",
    "InvalidVoiceError",
    "ModelInferenceError",
    "OfflineAssetError",
    "OptionalDependencyError",
    "PreparedVoice",
    "RequestApiContract",
    "RequestMeasure",
    "RuntimeCapabilities",
    "RuntimeCapabilityError",
    "RuntimeClosedError",
    "RuntimeDiagnostics",
    "SessionCreationError",
    "SoproRuntime",
    "SoproSynthError",
    "SynthesisRequest",
    "SynthesisResult",
    "UnsupportedBundleError",
    "UnsupportedFeatureError",
    "VoiceLevelApplication",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "VoiceLevelSource",
    "WordTiming",
    "__version__",
    "discover_bundles",
    "request_api_contract",
    "runtime_identity",
]
