from smoke.capabilities import (
    CAPABILITY_CONTRACTS,
    capability_names,
    contracted_feature_ids,
)
from smoke.features import FEATURE_INVENTORY, feature_ids

EXPECTED_CAPABILITIES = {
    "api_compatibility",
    "auth",
    "cli",
    "config",
    "extensibility",
    "local_providers",
    "messaging",
    "openrouter",
    "persistence",
    "provider_routing",
    "provider_runtime",
    "request_behavior",
    "streaming_conversion",
    "voice",
}


def test_capability_map_covers_every_public_feature() -> None:
    assert contracted_feature_ids() == feature_ids()


def test_capability_map_has_expected_top_level_groups() -> None:
    assert capability_names() == EXPECTED_CAPABILITIES


def test_capability_contracts_are_decision_complete() -> None:
    known_features = {feature.feature_id: feature for feature in FEATURE_INVENTORY}

    for contract in CAPABILITY_CONTRACTS:
        assert contract.feature_id in known_features, contract
        assert contract.capability.strip(), contract
        assert contract.subfeature.strip(), contract
        assert contract.owner.strip(), contract
        assert contract.inputs.strip(), contract
        assert contract.outputs.strip(), contract
        assert contract.failure.strip(), contract

        assert contract.pytest_tests or contract.smoke_tests, contract


def test_capability_contract_owners_do_not_reference_removed_request_pipeline() -> None:
    stale = [
        contract
        for contract in CAPABILITY_CONTRACTS
        if "free_claude_code.api.request_pipeline" in contract.owner
        or "ApiRequestPipeline" in contract.owner
    ]

    assert stale == []
