import pytest
from unittest.mock import MagicMock

from src.cicd.ci_gate import CIGate, CIGateResult, CIGateConfig


CER_THRESHOLD = 0.15
REGRESSION_TOLERANCE = 1.05


@pytest.fixture
def mock_mlflow_client():
    return MagicMock()


@pytest.fixture
def gate_config():
    return CIGateConfig(
        model_name="granite-docling-adapter",
        cer_threshold=CER_THRESHOLD,
        regression_tolerance=REGRESSION_TOLERANCE,
    )


@pytest.fixture
def gate(mock_mlflow_client, gate_config):
    return CIGate(client=mock_mlflow_client, config=gate_config)


class TestCIGateEvaluate:
    def test_passes_when_cer_below_threshold_no_production(self, gate):
        result = gate.evaluate(staging_cer=0.10, production_cer=None)
        assert result == CIGateResult.PASS

    def test_passes_when_cer_at_threshold_no_production(self, gate):
        result = gate.evaluate(staging_cer=CER_THRESHOLD, production_cer=None)
        assert result == CIGateResult.PASS

    def test_fails_when_cer_above_threshold(self, gate):
        result = gate.evaluate(staging_cer=0.20, production_cer=None)
        assert result == CIGateResult.FAIL_CER

    def test_fails_when_cer_slightly_above_threshold(self, gate):
        result = gate.evaluate(staging_cer=CER_THRESHOLD + 0.001, production_cer=None)
        assert result == CIGateResult.FAIL_CER

    def test_passes_when_staging_better_than_production(self, gate):
        result = gate.evaluate(staging_cer=0.08, production_cer=0.10)
        assert result == CIGateResult.PASS

    def test_passes_when_staging_equal_to_production(self, gate):
        result = gate.evaluate(staging_cer=0.10, production_cer=0.10)
        assert result == CIGateResult.PASS

    def test_passes_within_regression_tolerance(self, gate):
        # staging=0.104 <= production=0.10 * 1.05 = 0.105 → PASS
        result = gate.evaluate(staging_cer=0.104, production_cer=0.10)
        assert result == CIGateResult.PASS

    def test_fails_when_regresses_beyond_tolerance(self, gate):
        # staging=0.12 > production=0.10 * 1.05 = 0.105 → FAIL_REGRESSION
        result = gate.evaluate(staging_cer=0.12, production_cer=0.10)
        assert result == CIGateResult.FAIL_REGRESSION

    def test_fail_cer_takes_priority_over_regression(self, gate):
        # staging=0.20 > threshold AND regresses — CER check fires first
        result = gate.evaluate(staging_cer=0.20, production_cer=0.10)
        assert result == CIGateResult.FAIL_CER

    def test_no_production_skips_regression_check(self, gate):
        result = gate.evaluate(staging_cer=0.10, production_cer=None)
        assert result == CIGateResult.PASS

    def test_result_is_enum_type(self, gate):
        result = gate.evaluate(staging_cer=0.10, production_cer=None)
        assert isinstance(result, CIGateResult)


class TestCIGateMLflowTransitions:
    def test_promotes_to_production_on_pass(self, gate, mock_mlflow_client):
        gate.apply_transition(version=2, result=CIGateResult.PASS)
        mock_mlflow_client.transition_model_version_stage.assert_called_once_with(
            name="granite-docling-adapter",
            version=2,
            stage="Production",
            archive_existing_versions=True,   # single Production
        )

    def test_archives_on_fail_cer(self, gate, mock_mlflow_client):
        gate.apply_transition(version=2, result=CIGateResult.FAIL_CER)
        mock_mlflow_client.transition_model_version_stage.assert_called_once_with(
            name="granite-docling-adapter",
            version=2,
            stage="Archived",
            archive_existing_versions=False,
        )

    def test_archives_on_fail_regression(self, gate, mock_mlflow_client):
        gate.apply_transition(version=3, result=CIGateResult.FAIL_REGRESSION)
        mock_mlflow_client.transition_model_version_stage.assert_called_once_with(
            name="granite-docling-adapter",
            version=3,
            stage="Archived",
            archive_existing_versions=False,
        )

    def test_transition_called_exactly_once(self, gate, mock_mlflow_client):
        gate.apply_transition(version=1, result=CIGateResult.PASS)
        assert mock_mlflow_client.transition_model_version_stage.call_count == 1


class TestCIGateResultEnum:
    def test_all_members_exist(self):
        members = set(CIGateResult)
        assert CIGateResult.PASS in members
        assert CIGateResult.FAIL_CER in members
        assert CIGateResult.FAIL_REGRESSION in members

    def test_fail_cer_distinct_from_fail_regression(self):
        assert CIGateResult.FAIL_CER != CIGateResult.FAIL_REGRESSION

    def test_pass_distinct_from_failures(self):
        assert CIGateResult.PASS != CIGateResult.FAIL_CER
        assert CIGateResult.PASS != CIGateResult.FAIL_REGRESSION
