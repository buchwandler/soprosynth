import numpy as np
import pytest

from soprosynth import (
    DiscoveredBundle,
    PreparedVoice,
    RequestMeasure,
    SynthesisRequest,
    SynthesisResult,
)
from soprosynth.errors import InvalidLanguageError, InvalidRequestError


def test_request_requires_identifier_but_defers_empty_text_to_synthesis():
    request = SynthesisRequest(id="part-1", text=" \t")
    assert request.id == "part-1"
    assert request.text == " \t"
    with pytest.raises(InvalidRequestError):
        SynthesisRequest(id="  ", text="hello")
    with pytest.raises(InvalidRequestError):
        SynthesisRequest(id="part-1", text=4)
    with pytest.raises(InvalidLanguageError):
        SynthesisRequest(id="part-1", text="hello", language="ru")


def test_request_structural_context_is_immutable_and_well_formed():
    token = {"kind": "word", "value": "hello"}
    request = SynthesisRequest(id="part-1", text="hello", tokens=(token,))
    token["value"] = "changed"
    assert request.tokens[0]["value"] == "hello"
    with pytest.raises(TypeError):
        request.tokens[0]["value"] = "mutate"
    with pytest.raises(InvalidRequestError):
        SynthesisRequest(id="part-1", text="hello", tokens=[token])
    with pytest.raises(InvalidRequestError):
        SynthesisRequest(id="part-1", text="hello", tokens=(object(),))


def test_result_audio_and_metadata_validation():
    source = {"nested": {"items": [1, 2]}}
    result = SynthesisResult(
        id="part-1",
        audio=[0.1, -0.2],
        sample_rate=24000,
        text="hello",
        language="en",
        metadata=source,
    )
    source["nested"]["items"].append(3)
    assert result.audio.dtype == np.float32
    assert result.metadata["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        result.metadata["nested"]["items"] = ()
    for bad_audio in ([], [[0.1]], [float("inf")]):
        with pytest.raises(InvalidRequestError):
            SynthesisResult("part-1", bad_audio, 24000, "hello", "en")
    with pytest.raises(InvalidRequestError):
        SynthesisResult("part-1", [0.1], True, "hello", "en")


def test_prepared_voice_identity_excludes_reference_audio():
    voice = PreparedVoice(
        state=object(),
        sample_rate=24000,
        source="reference.wav",
        model_id="m",
        source_revision="r1",
        backend="native",
        runtime_fingerprint="runtime-1",
        fingerprint="a" * 64,
    )
    assert voice.identity == {
        "kind": "reference",
        "sha256": "a" * 64,
        "sample_rate": 24000,
        "model_id": "m",
        "source_revision": "r1",
        "backend": "native",
    }
    assert "audio" not in voice.identity


def test_measure_fits_is_derived():
    assert RequestMeasure(4, None).fits is None
    assert RequestMeasure(4, 4).fits is True
    assert RequestMeasure(5, 4).fits is False
    with pytest.raises(InvalidRequestError):
        RequestMeasure(5, 4, fits=True)


def test_discovery_metadata_is_copy_protected():
    metadata = {"nested": {"profiles": ["p"]}}
    bundle = DiscoveredBundle("id", "sopro:id", "Model", metadata=metadata)
    metadata["nested"]["profiles"].append("changed")
    assert bundle.metadata["nested"]["profiles"] == ("p",)
