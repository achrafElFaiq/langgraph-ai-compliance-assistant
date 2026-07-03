"""Integration test for MLflow experiment tracking.

Asserts the deliverable contract: after an evaluation pipeline run, a metric named
`faithfulness` exists in the experiment. Requires a reachable MLflow tracking
server (MLFLOW_TRACKING_URI) that already holds at least one eval run; skips
otherwise (this suite does not spend a real LLM run to produce the data).
"""
import os

import pytest

pytestmark = pytest.mark.integration

EXPERIMENT = "compliance-assistant"


def test_faithfulness_metric_present_in_experiment():
    """An eval run in the experiment logged a 'faithfulness' metric.

    Matters because faithfulness is the headline quality signal; if the pipeline
    stops logging it, regressions become invisible.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        pytest.skip("Set MLFLOW_TRACKING_URI to run the MLflow integration test")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT)
    except Exception as e:  # pragma: no cover - env dependent
        pytest.skip(f"MLflow server not reachable: {e}")
    if experiment is None:
        pytest.skip(f"Experiment {EXPERIMENT!r} does not exist yet — run the eval pipeline first")

    runs = client.search_runs([experiment.experiment_id], max_results=100)
    if not runs:
        pytest.skip("No runs logged yet — run the eval pipeline first")

    assert any("faithfulness" in run.data.metrics for run in runs), (
        "no run in the experiment logged a 'faithfulness' metric"
    )
