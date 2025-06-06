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


class MLflowClient:
    """A wrapper class for MLflow client operations."""
    
    def __init__(
        self, 
        tracking_uri=None,
        username=None, 
        password=None
    ):
        """
        Initialize the MLflow client with connection parameters.
        
        Args:
            tracking_uri: MLflow tracking server URI
            username: MLflow authentication username
            password: MLflow authentication password
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.username = username or os.getenv("MLFLOW_TRACKING_USERNAME", "")
        self.password = password or os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        
        # Set environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.password
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create client
        self.client = MlflowClient()  # noqa: F841

    def check_mlflow_ready(self):
        """
        Check if MLflow is reachable.

        Raises:
            Exception: If MLflow is not reachable
        """
        try:
            # Use the existing client
            _ = self.client
            return True
        except Exception as e:
            logger.warning(f"MLflow is not reachable: {e}")
            return False

    def get_mlflow_models(self):
        """
        Retrieve available MLflow models and create dropdown options.
        Models are sorted by creation timestamp, from newest to oldest.

        Returns:
            list: Dropdown options for MLflow models
        """
        try:
            # Search for registered models using existing client
            registered_models = self.client.search_registered_models()

            # Filter out models with "smi" in the name
            filtered_models = [model for model in registered_models if "smi" not in model.name.lower()]

            # Sort models by creation timestamp (newest first)
            filtered_models = sorted(
                filtered_models, key=lambda model: model.creation_timestamp, reverse=True
            )

            # Create dropdown options
            model_options = [
                {
                    "label": get_flow_run_name(get_flow_run_parent_id(model.name)),
                    "value": model.name,
                }
                for model in filtered_models
            ]

            return model_options
        except Exception as e:
            logger.warning(f"Error retrieving MLflow models: {e}")
            return [{"label": "No models found", "value": None}]

    def get_mlflow_params(self, mlflow_model_id):
        model_version_details = self.client.get_model_version(
            name=mlflow_model_id,  # The registered model name
            version="1",  # The version you care about
        )
        run_id = model_version_details.run_id

        run_info = self.client.get_run(run_id)
        params = run_info.data.params
        return params

    def get_mlflow_models_live(self, model_type=None):
        """
        Retrieve available MLflow models and create dropdown options
        
        Args:
            model_type (str, optional): Filter models by type tag. Possible values: 
                                        'autoencoder', 'dimension_reduction', or None to return all models
        
        Returns:
            list: Dropdown options for MLflow models filtered by type if specified
        """
        try:
            # Get all registered models using existing client
            models = self.client.search_registered_models()
            
            # Filter models with "smi" in the name as a basic filter
            models = [model for model in models if "smi" in model.name.lower()]
            
            # Since MLflow doesn't support searching by tags directly,
            # we need to manually filter the models by checking their tags
            if model_type:
                filtered_models = []
                for model in models:
                    # Get the latest version of the model
                    latest_versions = self.client.search_model_versions(f"name='{model.name}'")
                    if not latest_versions:
                        continue
                        
                    latest_version = max(latest_versions, key=lambda mv: int(mv.version))
                    
                    # Get run ID associated with the model version
                    run_id = latest_version.run_id
                    if not run_id:
                        continue
                        
                    # Get the run and check its tags
                    try:
                        run = self.client.get_run(run_id)
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

    def load_model(self, model_name):
        """
        Load a model from MLflow by name
        
        Args:
            model_name: Name of the model in MLflow
            
        Returns:
            The loaded model or None if loading fails
        """
        
        try:
            # Get latest version using existing client
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                logger.error(f"No versions found for model {model_name}")
                return None
                
            latest_version = max([int(mv.version) for mv in versions])
            model_uri = f"models:/{model_name}/{latest_version}"
            
            # Load via PyFunc wrapper
            logger.info(f"Loading model {model_name}, version {latest_version}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model: {model_name}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None


