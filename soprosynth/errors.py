"""Public exception hierarchy for SoproSynth."""

from __future__ import annotations


class SoproSynthError(Exception):
    """Base class for errors raised by the SoproSynth public API."""


class OptionalDependencyError(SoproSynthError):
    """A declared optional or core dependency is not installed."""


class InvalidRequestError(SoproSynthError):
    """A request does not satisfy the public request contract."""


class EmptyTextError(InvalidRequestError):
    """Synthesis was requested for empty or whitespace-only text."""


class InvalidLanguageError(InvalidRequestError):
    """The requested language is not supported."""


class InvalidVoiceError(InvalidRequestError):
    """A voice is invalid or incompatible with the selected runtime."""


class InvalidGenerationConfigError(InvalidRequestError):
    """A generation or voice-level configuration value is invalid."""


class UnsupportedFeatureError(SoproSynthError):
    """A requested API feature is not implemented by the selected backend."""

    def __init__(self, feature: str, message: str | None = None) -> None:
        self.feature = feature
        super().__init__(message or f"Unsupported feature: {feature}")


class RuntimeClosedError(SoproSynthError):
    """An operation was attempted after its runtime was closed."""


class RuntimeCapabilityError(SoproSynthError):
    """A backend cannot provide a requested capability or control."""


class SessionCreationError(SoproSynthError):
    """Backend model/session initialization failed."""


class ModelInferenceError(SoproSynthError):
    """Backend synthesis or inference failed."""


class AssetError(SoproSynthError):
    """Base error for asset access and installation failures."""


class AssetDownloadError(AssetError):
    """Asset acquisition failed."""


class AssetAccessError(AssetError):
    """An asset could not be accessed."""


class AssetCacheError(AssetError):
    """The local asset cache is invalid or unavailable."""


class CatalogUnavailableError(AssetError):
    """A required model catalog is unavailable."""


class OfflineAssetError(AssetError):
    """A required asset is unavailable in offline mode."""


class BundleNotFoundError(AssetError):
    """The requested model bundle does not exist."""


class UnsupportedBundleError(AssetError):
    """The requested bundle format or profile is unsupported."""
