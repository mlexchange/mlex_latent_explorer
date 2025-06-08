import logging
import os
import shutil
import hashlib

import mlflow
from mlex_utils.prefect_utils.core import get_flow_run_name
from mlflow.tracking import MlflowClient

from src.utils.prefect import get_flow_run_parent_id

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
# Define a cache directory that will be mounted as a volume
MLFLOW_CACHE_DIR = os.getenv("MLFLOW_CACHE_DIR", "/mlflow_cache")

logger = logging.getLogger(__name__)


class MLflowClient:
    """A wrapper class for MLflow client operations."""
    
    # In-memory model cache (for quick access)
    _model_cache = {}
    
    def __init__(
        self, 
        tracking_uri=None,
        username=None, 
        password=None,
        cache_dir=None
    ):
        """
        Initialize the MLflow client with connection parameters.
        
        Args:
            tracking_uri: MLflow tracking server URI
            username: MLflow authentication username
            password: MLflow authentication password
            cache_dir: Directory to store cached models
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.username = username or os.getenv("MLFLOW_TRACKING_USERNAME", "")
        self.password = password or os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        self.cache_dir = cache_dir or MLFLOW_CACHE_DIR
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.password
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create client
        self.client = MlflowClient()

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

    def _get_cache_path(self, model_name, version=None):
        """Get the cache path for a model"""
        # Create a unique filename based on model name and version
        if version is None:
            # Use a hash of the model name as part of the filename
            hash_obj = hashlib.md5(model_name.encode())
            hash_str = hash_obj.hexdigest()
            return os.path.join(self.cache_dir, f"{model_name}_{hash_str}")
        else:
            # Include version in the filename
            return os.path.join(self.cache_dir, f"{model_name}_v{version}")

    def load_model(self, model_name):
        """
        Load a model from MLflow by name with disk caching
        
        Args:
            model_name: Name of the model in MLflow
            
        Returns:
            The loaded model or None if loading fails
        """
        if model_name is None:
            logger.error("Cannot load model: model_name is None")
            return None
        
        # Check in-memory cache first
        if model_name in self._model_cache:
            logger.info(f"Using in-memory cached model: {model_name}")
            return self._model_cache[model_name]
        
        try:
            # Get latest version using existing client
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                logger.error(f"No versions found for model {model_name}")
                return None
                
            latest_version = max([int(mv.version) for mv in versions])
            model_uri = f"models:/{model_name}/{latest_version}"
            
            # Check disk cache
            cache_path = self._get_cache_path(model_name, latest_version)
            if os.path.exists(cache_path):
                logger.info(f"Loading model from disk cache: {cache_path}")
                try:
                    # Load from cached MLflow model
                    model = mlflow.pyfunc.load_model(cache_path)
                    
                    # Store in memory cache
                    self._model_cache[model_name] = model
                    
                    logger.info(f"Successfully loaded cached model: {model_name}")
                    return model
                except Exception as e:
                    logger.warning(f"Error loading model from cache: {e}")
                    # Continue to download if cache load fails
            
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
            
            # Instead of loading and then saving, we'll download directly to the cache location
            # This is more efficient and avoids the save_model error
            logger.info(f"Downloading model {model_name}, version {latest_version} from MLflow to cache")
            
            # Use mlflow.artifacts.download_artifacts to get the model artifacts
            try:
                # First method: Download the model directly to the cache location
                download_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{model_name}/{latest_version}",
                    dst_path=cache_path
                )
                logger.info(f"Downloaded model artifacts to: {download_path}")
                
                # Now load the model from the cached location
                model = mlflow.pyfunc.load_model(download_path)
                logger.info(f"Successfully loaded model from cache: {model_name}")
                
                # Store in memory cache
                self._model_cache[model_name] = model
                
                return model
            except Exception as e:
                logger.warning(f"Error downloading artifacts: {e}")
                
                # Fallback: Load the model directly from MLflow if download fails
                logger.info(f"Falling back to direct model loading from MLflow")
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Successfully loaded model: {model_name}")
                
                # Store in memory cache
                self._model_cache[model_name] = model
                
                return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    @classmethod
    def clear_memory_cache(cls):
        """Clear the in-memory model cache"""
        logger.info("Clearing in-memory model cache")
        cls._model_cache.clear()
    
    def clear_disk_cache(self):
        """Clear the disk cache"""
        logger.info(f"Clearing disk cache at {self.cache_dir}")
        try:
            # Delete and recreate the cache directory
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")