#!/usr/bin/env python
"""
VAE wrapper for MLflow.
Provides functionality to load a VAE model and register it with MLflow.
"""

import importlib.util
import os
import sys
import time
import traceback
from datetime import datetime

import mlflow
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


class VAEModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for VAE with direct model access and latent features functionality
    """

    def __init__(self, latent_dim=64, image_size=(512, 512)):
        self.model = None
        # Explicitly convert to integer to avoid type issues
        self.latent_dim = int(latent_dim) if latent_dim is not None else 64
        self.image_size = image_size

    def load_context(self, context):
        """Load VAE model from context artifacts"""
        # Get the model code path
        model_code_path = context.artifacts.get("model_code")
        if not model_code_path or not os.path.exists(model_code_path):
            raise FileNotFoundError(f"Model code not found at {model_code_path}")

        # Add directory to sys.path if needed
        model_dir = os.path.dirname(model_code_path)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        # Import module
        module_name = os.path.basename(model_code_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, model_code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print(
            f"Initializing VAE with latent_dim={self.latent_dim}, image_size={self.image_size}"
        )

        # Create model instance directly using ConvVAE class
        self.model = module.ConvVAE(
            latent_dim=self.latent_dim, image_size=self.image_size
        )

        # Load weights from NPZ file
        weights_path = context.artifacts.get("weights_path")
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

        # Load weights
        weights_npz = np.load(weights_path)

        # Convert numpy arrays to PyTorch tensors
        state_dict = {key: torch.tensor(weights_npz[key]) for key in weights_npz.files}

        # Load state dict
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("✓ VAE model weights loaded successfully (strict)")
        except Exception as e:
            print(f"⚠️ Strict loading failed: {e}")
            print("Attempting non-strict loading...")

            result = self.model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"Missing keys: {result.missing_keys[:5]}...")
            if result.unexpected_keys:
                print(f"Unexpected keys: {result.unexpected_keys[:5]}...")

        # Check for CUDA availability and set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Set model to eval mode and move to device
        self.model.eval()
        self.model = self.model.to(self.device)

        # Define the image transformation pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0,)),
            ]
        )

        print(f"✓ VAE model loaded successfully")

    def predict(self, context, model_input):
        """
        Standard predict method (required by MLflow)

        This method processes the input through the VAE model and returns both
        the reconstruction and the latent features.

        Args:
            context: MLflow context
            model_input: Input data as 2D/3D numpy array (H,W) or (H,W,C)

        Returns:
            Dictionary with reconstruction and latent features
        """
        if self.model is None:
            raise RuntimeError("VAE model not loaded. Call load_context first.")

        # Validate input
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Input must be a numpy array, got {type(model_input)}")

        # Check input dimensions
        model_input = np.squeeze(model_input)
        if not (len(model_input.shape) == 2 or len(model_input.shape) == 3):
            raise ValueError(
                f"Input must be a 2D or 3D array (H,W) or (H,W,C), got shape {model_input.shape}"
            )

        # Check and handle input dtype
        if model_input.dtype == np.uint8:
            # uint8 is already the preferred format, no conversion needed
            img_array = model_input
        elif model_input.dtype == np.uint32:
            # Convert uint32 to uint8 with robust min-max scaling
            array_min = model_input.min()
            array_max = model_input.max()
            # Protect against divide-by-zero and handle the case where all values are the same
            if array_max > array_min:
                # Scale using full range from min to max for better contrast
                img_array = (
                    (
                        (model_input.astype(np.float32) - array_min)
                        / (array_max - array_min)
                    )
                    * 255
                ).astype(np.uint8)
            else:
                # If all values are the same, create a uniform image
                img_array = np.zeros_like(model_input, dtype=np.uint8)
        elif model_input.dtype == np.float32:
            # Convert float32 to uint8 with robust min-max scaling
            array_min = model_input.min()
            array_max = model_input.max()

            # Protect against divide-by-zero and handle the case where all values are the same
            if array_max > array_min:
                # Scale using full range from min to max for better contrast
                img_array = (
                    ((model_input - array_min) / (array_max - array_min)) * 255
                ).astype(np.uint8)
            else:
                # If all values are the same, create a uniform image
                img_array = np.zeros_like(model_input, dtype=np.uint8)
        else:
            # Raise exception for unsupported dtypes
            raise ValueError(
                f"Input must be uint8, uint32, or float32, got {model_input.dtype}"
            )

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(img_array)

            # Apply transformations
            tensor = self.transform(pil_image)

            # Add batch dimension and move to device
            tensor = tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            raise ValueError(f"Failed to process input image: {e}")

        # Process with model
        with torch.no_grad():
            # Forward pass through the model
            x_reconstructed, mu, logvar = self.model(tensor)

            # Convert reconstruction to numpy
            reconstruction_np = x_reconstructed.cpu().numpy()

            # Get latent features (mu is the mean vector in the latent space)
            latent_features = mu.cpu().numpy()

        # Return results
        return {
            "reconstruction": reconstruction_np,
            "latent_features": latent_features,
            "logvar": logvar.cpu().numpy(),  # Also return logvar for completeness
        }


def save_vae_model_with_wrapper(
    model_config, tracking_uri, experiment_name, model_name=None
):
    """
    Save VAE model using PyFunc wrapper with latent features functionality

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

    print(f"\nSaving VAE model with PyFunc wrapper as: {model_name}")

    start_time = time.time()

    with mlflow.start_run(
        run_name=f"vae_model_wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")

        # Check file size and existence
        if not os.path.exists(model_config["state_dict"]):
            print(f"❌ Error: Weights file not found at {model_config['state_dict']}")
            return None, None

        if not os.path.exists(model_config["python_file"]):
            print(
                f"❌ Error: Model code file not found at {model_config['python_file']}"
            )
            return None, None

        npz_size = get_file_size_mb(model_config["state_dict"])
        print(f"\nNPZ file size: {npz_size:.1f} MB")

        try:
            # Get model parameters
            latent_dim = model_config.get("latent_dim", 64)
            image_size = model_config.get("image_size", (512, 512))

            # Create model wrapper
            vae_wrapper = VAEModelWrapper(latent_dim=latent_dim, image_size=image_size)

            # Log model information
            mlflow.log_params(
                {
                    "model_name": model_config["name"],
                    "model_type": model_config["type"],
                    "python_class": model_config["python_class"],
                    "latent_dim": latent_dim,
                    "image_size": f"{image_size[0]}x{image_size[1]}",
                    "npz_size_mb": npz_size,
                    "using_wrapper": True,
                }
            )

            # Set tags
            mlflow.set_tags({"exp_type": "live_mode", "model_type": "autoencoder"})

            # Create artifacts dictionary
            artifacts = {
                "weights_path": model_config["state_dict"],
                "model_code": model_config["python_file"],
            }

            # Define explicit requirements
            pip_requirements = [
                "torch==2.2.2",
                "numpy",
                "mlflow==2.22.0",
                "Pillow",
                "torchvision",
            ]

            # Log the VAE model with PyFunc wrapper
            print("\nLogging VAE model with PyFunc wrapper to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=vae_wrapper,
                artifacts=artifacts,
                registered_model_name=model_name,
                pip_requirements=pip_requirements,
                code_path=[__file__],  # Include this file's path
            )

            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)

            print(f"\n✅ VAE model saved with PyFunc wrapper in {total_time:.1f}s!")
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {tracking_uri}")

            return model_name, run.info.run_id

        except Exception as e:
            print(f"\n❌ Error saving VAE model with wrapper: {e}")
            traceback.print_exc()
            return None, None
