import logging
import os

import mlflow
from mlex_utils.prefect_utils.core import get_flow_run_name
from mlflow.tracking import MlflowClient

from src.utils.prefect import get_flow_run_parent_id

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

logger = logging.getLogger(__name__)


def check_mlflow_ready():
    """
    Check if MLflow is reachable.

    Raises:
        Exception: If MLflow is not reachable
    """
    try:
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()  # noqa: F841
        return True
    except Exception as e:
        logger.warning(f"MLflow is not reachable: {e}")
        return False


def get_mlflow_models():
    """
    Retrieve available MLflow models and create dropdown options.
    Models are sorted by creation timestamp, from newest to oldest.

    Returns:
        list: Dropdown options for MLflow models
    """
    try:
        # MLflow configuration
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = MlflowClient()

        # Search for registered models
        registered_models = client.search_registered_models()

        # Sort models by creation timestamp (newest first)
        registered_models = sorted(
            registered_models, key=lambda model: model.creation_timestamp, reverse=True
        )

        # Create dropdown options
        model_options = [
            {
                "label": get_flow_run_name(get_flow_run_parent_id(model.name)),
                "value": model.name,
            }
            for model in registered_models
        ]

        return model_options
    except Exception as e:
        logger.warning(f"Error retrieving MLflow models: {e}")
        return [{"label": "No models found", "value": None}]


def get_mlflow_params(mlflow_model_id):
    client = MlflowClient()

    model_version_details = client.get_model_version(
        name=mlflow_model_id,  # The registered model name
        version="1",  # The version you care about
    )
    run_id = model_version_details.run_id

    run_info = client.get_run(run_id)
    params = run_info.data.params
    return params
