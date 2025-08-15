#!/usr/bin/env python
"""
Neural UMAP wrapper for MLflow.
Provides functionality to load a PyTorch-based UMAP approximator model and register it with MLflow.
"""

import os
import sys
import time
import traceback
from datetime import datetime

import joblib
import mlflow
import numpy as np
import torch


def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


class SimpleUMAPApproximator(torch.nn.Module):
    """PyTorch neural network that approximates UMAP dimensionality reduction"""
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=2):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, h))
            layers.append(torch.nn.ReLU())
            input_dim = h
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class NeuralUMAPWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for PyTorch-based UMAP approximator with scaler preprocessing
    """

    def __init__(self, input_dim=512, hidden_dims=[128, 64], output_dim=2):
        self.model = None
        self.scaler = None
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

    def load_context(self, context):
        """Load neural UMAP model and scaler from context artifacts"""
        # Load model weights
        model_path = context.artifacts.get("model_weights")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        # Load scaler if available
        scaler_path = context.artifacts.get("scaler")
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            print("⚠️ No scaler found, will use raw input data")
            self.scaler = None

        # Initialize model
        print(f"Initializing Neural UMAP with input_dim={self.input_dim}, output_dim={self.output_dim}")
        self.model = SimpleUMAPApproximator(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )

        # Check for CUDA availability and set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Load weights
        try:
            # Handle both state_dict and full model saves
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                state_dict = torch.load(model_path, map_location=self.device)
                # Check if it's just the state dict or full model
                if isinstance(state_dict, dict) and "network.0.weight" in state_dict:
                    # Direct state dict
                    self.model.load_state_dict(state_dict)
                elif hasattr(state_dict, 'state_dict'):
                    # Full model
                    self.model.load_state_dict(state_dict.state_dict())
                else:
                    raise ValueError(f"Unexpected format in weights file: {type(state_dict)}")
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            print("✓ Neural UMAP model weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            traceback.print_exc()
            raise

        # Set model to eval mode and move to device
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, context, model_input):
        """
        Standard predict method for neural UMAP model

        Args:
            context: MLflow context
            model_input: Input data as numpy array (latent vectors from autoencoder)

        Returns:
            Dictionary with UMAP coordinates
        """
        if self.model is None:
            raise RuntimeError("Neural UMAP model not loaded. Call load_context first.")

        # Validate input
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Input must be a numpy array, got {type(model_input)}")

        # Check input dimensions
        if len(model_input.shape) > 2:
            # Flatten all dimensions except the first (batch)
            model_input = model_input.reshape(model_input.shape[0], -1)
        
        # Ensure input matches expected dimension
        if model_input.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {model_input.shape[1]}"
            )

        # Apply scaler if available
        if self.scaler is not None:
            try:
                model_input = self.scaler.transform(model_input)
            except Exception as e:
                print(f"⚠️ Error applying scaler: {e}")
                raise ValueError(f"Failed to apply scaler to input: {e}")

        # Convert to torch tensor
        input_tensor = torch.tensor(model_input, dtype=torch.float32).to(self.device)

        # Process with model
        with torch.no_grad():
            # Apply neural UMAP transformation
            umap_coords = self.model(input_tensor).cpu().numpy()

        # Return results
        return {"umap_coords": umap_coords}


def save_neural_umap_model_with_wrapper(
    model_config, tracking_uri, experiment_name, model_name=None
):
    """
    Save neural UMAP model with a PyFunc wrapper

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

    print(f"\nSaving neural UMAP model as: {model_name}")

    start_time = time.time()

    with mlflow.start_run(
        run_name=f"neural_umap_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")

        # Check file existence
        if not os.path.exists(model_config["weights_path"]):
            print(
                f"⚠️ Neural UMAP model weights not found at {model_config['weights_path']}"
            )
            return None, None

        # Check scaler existence if specified
        scaler_exists = False
        if "scaler_path" in model_config and model_config["scaler_path"]:
            if os.path.exists(model_config["scaler_path"]):
                scaler_exists = True
            else:
                print(
                    f"⚠️ Scaler not found at {model_config['scaler_path']}, will proceed without it"
                )

        # Get file sizes
        weights_size = get_file_size_mb(model_config["weights_path"])
        scaler_size = get_file_size_mb(model_config.get("scaler_path", "")) if scaler_exists else 0
        print(f"\nWeights file size: {weights_size:.1f} MB")
        if scaler_exists:
            print(f"Scaler file size: {scaler_size:.1f} MB")

        try:
            # Get model parameters
            input_dim = model_config.get("input_dim", 512)
            
            # Create model wrapper with default hidden_dims and output_dim
            neural_umap_wrapper = NeuralUMAPWrapper(
                input_dim=input_dim
            )

            # Log parameters
            params = {
                "model_name": model_config.get("name", "SMI_NeuralDimRed"),
                "model_type": "neural_umap",
                "weights_size_mb": weights_size,
                "using_wrapper": True,
                "input_dim": input_dim,
                "has_scaler": scaler_exists
            }
            if scaler_exists:
                params["scaler_size_mb"] = scaler_size
            
            mlflow.log_params(params)

            # Set tags
            mlflow.set_tags(
                {"exp_type": "live_mode", "model_type": "dimension_reduction"}
            )

            # Create artifacts dictionary
            artifacts = {"model_weights": model_config["weights_path"]}
            if scaler_exists:
                artifacts["scaler"] = model_config["scaler_path"]

            # Define explicit requirements
            pip_requirements = [
                "torch==2.2.2",
                "numpy",
                "scikit-learn",
                "joblib",
                "mlflow==2.22.0"
            ]

            # Log the neural UMAP model wrapper
            try:
                print("\nLogging neural UMAP model wrapper to MLflow...")
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=neural_umap_wrapper,
                    artifacts=artifacts,
                    registered_model_name=model_name,
                    pip_requirements=pip_requirements,
                    code_path=[__file__],  # Include this file's path
                )
                print(
                    f"✓ Neural UMAP model wrapper registered as '{model_name}'"
                )
            except Exception as e:
                print(f"⚠️ Error logging neural UMAP model wrapper: {e}")
                traceback.print_exc()
                return None, None

            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)

            print(
                f"\n✅ Neural UMAP model saved successfully in {total_time:.1f}s!"
            )
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {tracking_uri}")

            return model_name, run.info.run_id

        except Exception as e:
            print(f"\n❌ Error saving neural UMAP model: {e}")
            traceback.print_exc()
            return None, None