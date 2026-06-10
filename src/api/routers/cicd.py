"""
/cicd/* endpoints — CI/CD gate for model stage transitions.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_mlflow_client
from src.api.schemas import CIGateRequest, CIGateResponse
from src.cicd.ci_gate import CIGate, CIGateConfig, CIGateResult

router = APIRouter(prefix="/cicd", tags=["cicd"])

_MODEL_NAME = "granite-docling-adapter"


def _get_metric(client, run_id: str, key: str) -> float | None:
    """Read a metric from an MLflow run. Returns None if not logged."""
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get(key)
    except Exception:
        return None


@router.post("/gate", response_model=CIGateResponse)
def run_ci_gate(
    request: CIGateRequest,
    client=Depends(get_mlflow_client),
):
    """
    Evaluate the Staging model and transition it to Production or Archived.

    Flow:
      1. Find the Staging version in MLflow Registry.
      2. Read its 'cer' (benchmark) + 'eval_loss' (val) from the training run.
      3. Read Production version's 'eval_loss' (if a Production version exists).
      4. Apply CIGate logic: cer_threshold (absolute) + eval_loss regression.
      5. Transition: PASS → Production, FAIL → Archived.

    Raises 404 if no Staging version is found.
    Raises 422 if the Staging run has no 'cer' metric logged.
    """
    versions = client.search_model_versions(f"name='{_MODEL_NAME}'")

    staging = next((v for v in versions if v.current_stage == "Staging"), None)
    if staging is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Staging version found for '{_MODEL_NAME}'.",
        )

    staging_cer = _get_metric(client, staging.run_id, "cer")
    if staging_cer is None:
        raise HTTPException(
            status_code=422,
            detail=f"Staging version {staging.version} has no 'cer' metric. "
                   "Make sure train.py logs cer before registering the adapter.",
        )
    # Regression uses benchmark_loss (CE on the FIXED benchmark) — comparable
    # across versions, unlike per-epoch eval_loss (each version's own val split).
    staging_loss = _get_metric(client, staging.run_id, "benchmark_loss")

    production = next((v for v in versions if v.current_stage == "Production"), None)
    production_cer = _get_metric(client, production.run_id, "cer") if production else None
    production_loss = _get_metric(client, production.run_id, "benchmark_loss") if production else None

    cfg = CIGateConfig(
        model_name=_MODEL_NAME,
        cer_threshold=request.cer_threshold,
        regression_tolerance=request.regression_tolerance,
    )
    gate = CIGate(client=client, config=cfg)
    result = gate.evaluate(staging_cer, staging_loss, production_loss)
    gate.apply_transition(version=int(staging.version), result=result)

    return CIGateResponse(
        result=result.value,
        staging_version=int(staging.version),
        staging_cer=staging_cer,
        production_cer=production_cer,
        staging_loss=staging_loss,
        production_loss=production_loss,
        new_stage="Production" if result == CIGateResult.PASS else "Archived",
    )
