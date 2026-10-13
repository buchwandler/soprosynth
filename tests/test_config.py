import pytest

from soprosynth import GenerationConfig, VoiceLevelConfig
from soprosynth.errors import InvalidGenerationConfigError


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": True},
        {"top_p": 0},
        {"top_p": 1.01},
        {"top_k": True},
        {"top_k": 0},
        {"steps": 1.5},
        {"steps": 0},
        {"max_seconds": -1},
        {"max_seconds": float("inf")},
        {"min_seconds": -0.1},
        {"min_seconds": True},
        {"min_seconds": 3, "max_seconds": 2},
    ],
)
def test_invalid_generation_configs(kwargs):
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(**kwargs)


def test_native_kwargs_omit_unspecified_controls():
    config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=12, max_seconds=3)
    assert config.as_native_kwargs() == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 12,
        "max_seconds": 3,
    }


def test_voice_level_config_is_strict_and_finite():
    assert VoiceLevelConfig().mode == "off"
    assert VoiceLevelConfig(mode="calibrated", gain_db=-1.0).gain_db == -1.0
    with pytest.raises(InvalidGenerationConfigError):
        VoiceLevelConfig(mode="normalized")
    with pytest.raises(InvalidGenerationConfigError):
        VoiceLevelConfig(gain_db=True)
    with pytest.raises(InvalidGenerationConfigError):
        VoiceLevelConfig(gain_db=float("nan"))
