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
    assert contract.caller_owns_text_boundaries is False
    assert contract.supports_sopro_output_policy is True
    assert contract.supports_reference_voice is True
    assert contract.supports_request_measurement is True
    assert contract.supports_streaming is False
    assert contract.supports_linguistic_tokens is False
    assert contract.supports_pronunciation_overrides is False


def test_runtime_capabilities_report_onnxvoice_only():
    capabilities = SoproRuntime(object()).capabilities

    assert isinstance(capabilities, RuntimeCapabilities)
    assert capabilities.backend == "onnx"
    assert capabilities.supports_streaming is False
    assert capabilities.supports_request_measurement is True
    assert capabilities.supports_prepared_voice_reuse is True
