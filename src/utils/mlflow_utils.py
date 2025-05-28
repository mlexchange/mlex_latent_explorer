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

        # Filter out models with "smi" in the name
        registered_models = [model for model in registered_models if "smi" not in model.name.lower()]

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


def get_mlflow_models_live(model_type=None):
    """
    Retrieve available MLflow models and create dropdown options
    
    Args:
        model_type (str, optional): Filter models by type tag. Possible values: 
                                    'autoencoder', 'dimension_reduction', or None to return all models
    
    Returns:
        list: Dropdown options for MLflow models filtered by type if specified
    """
    try:
        # Set MLflow tracking URI and credentials
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get all registered models
        client = mlflow.MlflowClient()
        models = client.search_registered_models()
        
        # Filter models with "smi" in the name as a basic filter
        models = [model for model in models if "smi" in model.name.lower()]
        
        # Since MLflow doesn't support searching by tags directly,
        # we need to manually filter the models by checking their tags
        if model_type:
            filtered_models = []
            for model in models:
                # Get the latest version of the model
                latest_versions = client.search_model_versions(f"name='{model.name}'")
                if not latest_versions:
                    continue
                    
                latest_version = max(latest_versions, key=lambda mv: int(mv.version))
                
                # Get run ID associated with the model version
                run_id = latest_version.run_id
                if not run_id:
                    continue
                    
                # Get the run and check its tags
                try:
                    run = client.get_run(run_id)
                    if run.data.tags.get("model_type") == model_type:
                        filtered_models.append(model)
                except Exception as e:
                    logger.warning(f"Error retrieving run {run_id}: {e}")
                    continue
            
            models = filtered_models
        
        # Format as dropdown options
        model_options = [
            {"label": model.name, "value": model.name}
            for model in models
        ]
        
        return model_options
    except Exception as e:
        logger.warning(f"Error retrieving MLflow models: {e}")
        return [{"label": "Error loading models", "value": None}]