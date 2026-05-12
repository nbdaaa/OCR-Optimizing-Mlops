"""
CI/CD gate: evaluate a Staging model version and transition it to
Production or Archived based on CER threshold and regression check.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── Result enum ───────────────────────────────────────────────────────────────

class CIGateResult(Enum):
    PASS             = "pass"
    FAIL_CER         = "fail_cer"
    FAIL_REGRESSION  = "fail_regression"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class CIGateConfig:
    """
    model_name:           Registered model name in MLflow Registry.
    cer_threshold:        Max acceptable CER for a model to reach Production.
    regression_tolerance: Staging CER must be <= production_cer * tolerance.
                          1.05 means up to 5% worse than Production is allowed.
    """
    model_name: str
    cer_threshold: float = 0.15
    regression_tolerance: float = 1.05


# ── Gate ──────────────────────────────────────────────────────────────────────

class CIGate:
    def __init__(self, client, config: CIGateConfig) -> None:
        """
        Args:
            client: mlflow.MlflowClient instance.
            config: CIGateConfig.
        """
        self.client = client
        self.config = config

    def evaluate(
        self,
        staging_cer: float,
        production_cer: float | None,
    ) -> CIGateResult:
        """
        Decide whether the Staging model passes the CI gate.

        Checks (in order):
            1. staging_cer > cer_threshold         → FAIL_CER
            2. production_cer is not None AND
               staging_cer > production_cer * regression_tolerance
                                                   → FAIL_REGRESSION
            3. Otherwise                           → PASS

        Args:
            staging_cer:    CER of the model version under evaluation.
            production_cer: CER of the current Production version.
                            Pass None when there is no Production version yet.

        Returns:
            CIGateResult enum value.
        """
        if staging_cer > self.config.cer_threshold:
            return CIGateResult.FAIL_CER

        if production_cer is not None:
            if staging_cer > production_cer * self.config.regression_tolerance:
                return CIGateResult.FAIL_REGRESSION

        return CIGateResult.PASS

    def apply_transition(self, version: int, result: CIGateResult) -> None:
        """
        Transition the model version in MLflow Registry based on gate result.

            PASS             → stage "Production"
            FAIL_CER         → stage "Archived"
            FAIL_REGRESSION  → stage "Archived"

        Args:
            version: MLflow model version number.
            result:  CIGateResult from evaluate().
        """
        stage = "Production" if result == CIGateResult.PASS else "Archived"
        self.client.transition_model_version_stage(
            name=self.config.model_name,
            version=version,
            stage=stage,
        )
