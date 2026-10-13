from soprosynth import (
    REQUEST_API_VERSION,
    RequestApiContract,
    RuntimeCapabilities,
    SoproRuntime,
    SynthesisRequest,
    SynthesisResult,
    request_api_contract,
)


def test_v1_request_contract():
    contract = request_api_contract()
    assert isinstance(contract, RequestApiContract)
    assert contract.version == REQUEST_API_VERSION == 1
    assert contract.request_type is SynthesisRequest
    assert contract.result_type is SynthesisResult
    assert contract.runtime_type is SoproRuntime
    assert contract.synthesis_method == "synthesize"
    assert contract.measurement_method == "measure_request"
    assert contract.discovery_method == "discover_bundles"
    assert contract.runtime_identity_method == "runtime_identity"
    assert contract.caller_owns_text_boundaries is True
    assert contract.supports_reference_voice is True
    assert contract.supports_request_measurement is True
    assert contract.supports_linguistic_tokens is False
    assert contract.supports_pronunciation_overrides is False


def test_runtime_capabilities_are_backend_specific_without_optional_imports():
    native = SoproRuntime(object()).capabilities
    onnx = SoproRuntime(object(), backend="onnx").capabilities

    assert isinstance(native, RuntimeCapabilities)
    assert native.backend == "native"
    assert native.supports_streaming is True
    assert native.supports_request_measurement is False
    assert native.supports_prepared_voice_reuse is True
    assert onnx.backend == "onnx"
    assert onnx.supports_streaming is False
    assert onnx.supports_request_measurement is True
    assert request_api_contract().version == 1
    assert request_api_contract().supports_streaming is True
    assert request_api_contract().supports_request_measurement is True
    assert "onnxvoice" not in __import__("sys").modules
