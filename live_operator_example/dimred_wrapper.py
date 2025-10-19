"""
Dimensionality reduction wrapper for MLflow.
Provides functionality to load a dimensionality reduction model and register it with MLflow.
"""

import os
import time
import traceback
from datetime import datetime

import joblib
import mlflow
import numpy as np
from sklearn.preprocessing import RobustScaler  # ✅ ADDED


def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


class DimRedModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for dimensionality reduction model with direct model access"""

    def __init__(self, use_transformation=False):  # ✅ ADDED use_transformation
        self.model = None
        self.use_transformation = use_transformation  # ✅ ADDED

    def load_context(self, context):
        """Load dimensionality reduction model from context artifacts"""
        # Load model
        model_path = context.artifacts.get("dimred_model")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Dimensionality reduction model not found in artifacts")

        print(f"Loading dimensionality reduction model from {model_path}")
        self.model = joblib.load(model_path)
        print("✅ Dimensionality reduction model loaded successfully")
        print(f"Transformation enabled: {self.use_transformation}")  # ✅ ADDED

    # ✅ ADDED METHOD
    def _apply_transformation(self, features):
        """Apply square + robust_scale transformation"""
        # Step 1: Square transformation
        transformed = np.sign(features) * (features ** 2)
        
        # Step 2: Robust scale
        scaler = RobustScaler()
        transformed = scaler.fit_transform(transformed)
        
        return transformed

    def predict(self, context, model_input):
        """
        Standard predict method for dimensionality reduction model

        Args:
            context: MLflow context
            model_input: Input data as numpy array

        Returns:
            Dictionary with coordinates
        """
        if self.model is None:
            raise RuntimeError("Dimensionality reduction model not loaded. Call load_context first.")

        # Validate input
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Input must be a numpy array, got {type(model_input)}")

        # Check input dimensions
        if len(model_input.shape) != 2 or model_input.shape[0] != 1:
            raise ValueError(
                f"Input must be a 2D array with shape (1, latent_dim), got shape {model_input.shape}"
            )

        # ✅ ADDED: Apply feature transformation if enabled
        if self.use_transformation:
            model_input = self._apply_transformation(model_input)

        # Apply transformation
        coords = self.model.transform(model_input)

        # Return results
        return {"coords": coords}


def save_dimred_model_with_wrapper(
    model_config, tracking_uri, experiment_name, model_name=None
):
    """
    Save dimensionality reduction model with a direct access wrapper

    Args:
        model_config: Dictionary with model configuration
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        model_name: Optional model name, defaults to name from config with date

    Returns:
        Tuple of (model_name, run_id) or (None, None) on failure
    """

    # Set MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Set model name
    if model_name is None:
        model_name = f"{model_config['name']}_v{datetime.now().strftime('%Y%m%d')}"

    print(f"\nSaving dimensionality reduction model as: {model_name}")

    start_time = time.time()

    with mlflow.start_run(
        run_name=f"dr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")

        # Check file existence
        if not os.path.exists(model_config["file"]):
            print(
                f"⚠️ Dimensionality reduction model file not found at {model_config['file']}"
            )
            return None, None

        # Check file size
        joblib_size = get_file_size_mb(model_config["file"])
        print(f"\nFile size: {joblib_size:.1f} MB")

        try:
            use_transformation = model_config.get("use_transformation", False)  # ✅ ADDED
            print(f"Transformation enabled: {use_transformation}")  # ✅ ADDED
            
            # Create model wrapper
            dimred_wrapper = DimRedModelWrapper(use_transformation=use_transformation)  # ✅ MODIFIED

            # Log parameters
            mlflow.log_params(
                {
                    "model_name": model_config["name"],
                    "model_type": model_config["type"],
                    "joblib_size_mb": joblib_size,
                    "using_wrapper": True,
                    "input_dim": model_config["input_dim"],
                    "use_transformation": use_transformation  # ✅ ADDED
                }
            )

            # Set tags
            mlflow.set_tags(
                {"exp_type": "live_mode", "model_type": "dimension_reduction"}
            )

            # Create artifacts dictionary
            artifacts = {"dimred_model": model_config["file"]}

            # Define explicit requirements
            pip_requirements = [
                "numpy",
                "scikit-learn",
                "joblib",
                "mlflow==2.22.0",
                "umap-learn",
            ]

            # Log the dimensionality reduction model wrapper
            try:
                print("\nLogging dimensionality reduction model wrapper to MLflow...")
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=dimred_wrapper,
                    artifacts=artifacts,
                    registered_model_name=model_name,
                    pip_requirements=pip_requirements,
                    code_path=[__file__],  # Include this file's path
                )
                print(
                    f"✅ Dimensionality reduction model wrapper registered as '{model_name}'"
                )
            except Exception as e:
                print(f"⚠️ Error logging dimensionality reduction model wrapper: {e}")
                traceback.print_exc()
                return None, None

            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)

            print(
                f"\n✅ Dimensionality reduction model saved successfully in {total_time:.1f}s!"
            )
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {tracking_uri}")

            return model_name, run.info.run_id

        except Exception as e:
            print(f"\n❌ Error saving dimensionality reduction model: {e}")
            traceback.print_exc()
            return None, None