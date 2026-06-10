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
    cer_threshold:        Max acceptable CER (on the fixed benchmark) for a model
                          to reach Production — absolute quality floor.
    regression_tolerance: Staging eval_loss (val cross-entropy) must be
                          <= production eval_loss * tolerance. 1.05 means up to
                          5% worse than Production is allowed.
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
        staging_loss: float | None = None,
        production_loss: float | None = None,
    ) -> CIGateResult:
        """
        Decide whether the Staging model passes the CI gate.

        Two independent signals:
          - CER (on the fixed benchmark) = absolute quality floor.
          - eval_loss (val cross-entropy) = regression vs current Production.

        Checks (in order):
            1. staging_cer > cer_threshold              → FAIL_CER
            2. production_loss and staging_loss present AND
               staging_loss > production_loss * regression_tolerance
                                                        → FAIL_REGRESSION
            3. Otherwise                                → PASS

        Regression is skipped when either loss is missing (e.g. no Production
        yet, or a Production trained before per-epoch eval was added) — the CER
        floor still applies.

        Args:
            staging_cer:     CER of the version under evaluation (benchmark).
            staging_loss:    Staging val eval_loss (None → skip regression).
            production_loss: Current Production val eval_loss (None → skip).

        Returns:
            CIGateResult enum value.
        """
        if staging_cer > self.config.cer_threshold:
            return CIGateResult.FAIL_CER

        if production_loss is not None and staging_loss is not None:
            if staging_loss > production_loss * self.config.regression_tolerance:
                return CIGateResult.FAIL_REGRESSION

        return CIGateResult.PASS

    def apply_transition(self, version: int, result: CIGateResult) -> None:
        """
        Transition the model version in MLflow Registry based on gate result.

            PASS             → stage "Production"
            FAIL_CER         → stage "Archived"
            FAIL_REGRESSION  → stage "Archived"

        On PASS, archive_existing_versions=True so the newly promoted version is
        the SINGLE Production version (any prior Production → Archived) — MLflow
        does not enforce one-Production-per-model on its own.

        Args:
            version: MLflow model version number.
            result:  CIGateResult from evaluate().
        """
        stage = "Production" if result == CIGateResult.PASS else "Archived"
        self.client.transition_model_version_stage(
            name=self.config.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
