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
    # ── CER = absolute quality floor ──────────────────────────────────────────
    def test_passes_when_cer_below_threshold(self, gate):
        assert gate.evaluate(staging_cer=0.10) == CIGateResult.PASS

    def test_passes_when_cer_at_threshold(self, gate):
        assert gate.evaluate(staging_cer=CER_THRESHOLD) == CIGateResult.PASS

    def test_fails_when_cer_above_threshold(self, gate):
        assert gate.evaluate(staging_cer=0.20) == CIGateResult.FAIL_CER

    def test_fails_when_cer_slightly_above_threshold(self, gate):
        assert gate.evaluate(staging_cer=CER_THRESHOLD + 0.001) == CIGateResult.FAIL_CER

    # ── Regression = eval_loss (val) vs Production ────────────────────────────
    def test_passes_when_loss_better_than_production(self, gate):
        assert gate.evaluate(staging_cer=0.10, staging_loss=0.08,
                             production_loss=0.10) == CIGateResult.PASS

    def test_passes_when_loss_equal_to_production(self, gate):
        assert gate.evaluate(staging_cer=0.10, staging_loss=0.10,
                             production_loss=0.10) == CIGateResult.PASS

    def test_passes_within_regression_tolerance(self, gate):
        # staging_loss=0.104 <= production_loss=0.10 * 1.05 = 0.105 → PASS
        assert gate.evaluate(staging_cer=0.10, staging_loss=0.104,
                             production_loss=0.10) == CIGateResult.PASS

    def test_fails_when_loss_regresses_beyond_tolerance(self, gate):
        # staging_loss=0.12 > 0.10 * 1.05 = 0.105 → FAIL_REGRESSION
        assert gate.evaluate(staging_cer=0.10, staging_loss=0.12,
                             production_loss=0.10) == CIGateResult.FAIL_REGRESSION

    def test_fail_cer_takes_priority_over_regression(self, gate):
        # cer above threshold AND loss regresses — CER check fires first
        assert gate.evaluate(staging_cer=0.20, staging_loss=0.12,
                             production_loss=0.10) == CIGateResult.FAIL_CER

    def test_no_production_loss_skips_regression(self, gate):
        # loss would regress, but no Production loss → regression skipped → PASS
        assert gate.evaluate(staging_cer=0.10, staging_loss=0.99,
                             production_loss=None) == CIGateResult.PASS

    def test_missing_staging_loss_skips_regression(self, gate):
        assert gate.evaluate(staging_cer=0.10, staging_loss=None,
                             production_loss=0.10) == CIGateResult.PASS

    def test_result_is_enum_type(self, gate):
        assert isinstance(gate.evaluate(staging_cer=0.10), CIGateResult)


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
