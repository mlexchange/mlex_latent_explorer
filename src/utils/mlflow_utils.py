import os
import mlflow
from mlflow.tracking import MlflowClient

def get_mlflow_models():
    """
    Retrieve available MLflow models and create dropdown options.
    Models are sorted by creation timestamp, from newest to oldest.
   
    Returns:
        list: Dropdown options for MLflow models
    """
    try:
        # MLflow configuration
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()
       
        # Search for registered models
        registered_models = client.search_registered_models()
        
        # Sort models by creation timestamp (newest first)
        registered_models = sorted(registered_models, key=lambda model: model.creation_timestamp, reverse=True)
       
        # Create dropdown options
        model_options = [
            {"label": model.name, "value": model.name}
            for model in registered_models
        ]
       
        return model_options
    except Exception as e:
        print(f"Error retrieving MLflow models: {e}")
        return [{"label": "No models found", "value": None}]