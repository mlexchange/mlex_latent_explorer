#!/usr/bin/env python
"""
Simplified MLflow Model Processing

This script loads both direct and wrapped models from MLflow and uses them for inference.
- ViT model: Loaded directly as a PyTorch model (no wrapper)
- UMAP model: Loaded through a wrapper

Usage:
    python process_models.py
"""

import os
import sys
import logging

import torch
import torchvision.transforms as transforms
import numpy as np
import mlflow
from tiled.client import from_uri
from tiled_utils import write_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATA_TILED_URI = os.getenv("DATA_TILED_URI", "")
DATA_TILED_KEY = os.getenv("DATA_TILED_KEY", None)
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", None)

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")


# Add compatibility patch for torch if needed
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device


def get_model(model_name, needs_eval=True, use_wrapper=True):
    """
    Load a model from MLflow by its name
    
    Parameters:
    -----------
    model_name : str
        Name of the registered model in MLflow
    needs_eval : bool
        If True, sets the model to evaluation mode (for PyTorch models)
    use_wrapper : bool
        If True, assumes model is wrapped with PyFunc and needs unwrapping
        If False, loads model directly as PyTorch model
    
    Returns:
    --------
    model : object
        The loaded model from MLflow
    """
    # Set MLflow tracking URI and credentials
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get latest version
    client = mlflow.tracking.MlflowClient()
    latest_version = max([mv.version for mv in client.search_model_versions(f"name='{model_name}'")])
    model_uri = f"models:/{model_name}/{latest_version}"
    
    try:
        if use_wrapper:
            # Load via PyFunc wrapper
            logger.info(f"Loading model with wrapper: {model_name}")
            wrapper = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model wrapper: {model_name}, version: {latest_version}")
            
            # Unwrap the python model
            python_model = wrapper.unwrap_python_model()
            
            if hasattr(python_model, 'model') and python_model.model is not None:
                logger.info(f"Found model in the wrapper")
                model = python_model.model
            else:
                logger.error(f"No 'model' attribute found in the Python model")
                return None
        else:
            # Load directly as PyTorch model
            logger.info(f"Loading model directly (no wrapper): {model_name}")
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Successfully loaded PyTorch model: {model_name}, version: {latest_version}")
        
        # Set to evaluation mode if needed
        if needs_eval and hasattr(model, 'eval'):
            model.eval()
            logger.info(f"Set model to evaluation mode")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Get model names from command line arguments or environment variables
    autoencoder_model_name = "smi_autoencoder_model"
    dimred_model_name = "smi_umap_model"
    
    # Load the autoencoder model directly (no wrapper)
    logger.info(f"Loading autoencoder model: {autoencoder_model_name}")
    model = get_model(autoencoder_model_name, needs_eval=True, use_wrapper=False)  # PyTorch model without wrapper
    
    if model is None:
        logger.error("Failed to load autoencoder model. Exiting.")
        sys.exit(1)
    
    # Load the umap model with wrapper
    logger.info(f"Loading UMAP model from MLflow: {dimred_model_name}")
    dim_reduction_model = get_model(dimred_model_name, needs_eval=False, use_wrapper=True)  # UMAP model with wrapper

    if dim_reduction_model is None:
        logger.error("Failed to load UMAP model. Exiting.")
        sys.exit(1)
    
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Move model to the appropriate device
    model = model.to(device)

    # Define image transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize to smaller dimensions
            transforms.ToTensor(),  # Convert image to PyTorch tensor (0-1 range)
            transforms.Normalize((0.0,), (1.0,)),  # Normalize tensor
        ]
    )

    # Set up Tiled clients
    data_client = from_uri(DATA_TILED_URI, api_key=DATA_TILED_KEY)
    write_client = from_uri(RESULTS_TILED_URI, api_key=RESULTS_TILED_API_KEY)

    # Processing loop
    logger.info("Starting processing loop...")
    for i in range(1000):  # TODO: Change to long running loop that detects new data
        try:
            # Get datapoint from Tiled
            datapoint = data_client[indx]  # noqa: F821
            tensor = transform(datapoint).unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            # Process with autoencoder (extract latent vector)
            with torch.no_grad():
                # For autoencoder models, access the encoder
                latent, _ = model.encoder(tensor)

                # Convert to numpy for dimension reduction
                f_vec_nn = latent.cpu().numpy()
            
            # Apply dimension reduction
            f_vec = dim_reduction_model.transform(f_vec_nn)
            
            # Save results to Tiled
            write_results(
                write_client,
                f_vec,
                io_parameters,  # noqa: F821
                latent_vectors_path,  # noqa: F821
                metadata={
                    "autoencoder_model": autoencoder_model_name,
                    "dimred_model": dimred_model_name
                },
            )
            
            logger.info(f"Processed datapoint {i} successfully")
            
        except Exception as e:
            logger.error(f"Error processing datapoint {i}: {e}")
            import traceback
            traceback.print_exc()