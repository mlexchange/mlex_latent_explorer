#!/usr/bin/env python
"""
Simplified MLflow Model Processing using PyFunc Wrappers

This script loads both ViT and UMAP models from MLflow using PyFunc wrappers and uses them for inference.
- ViT model: Loaded through a PyFunc wrapper
- UMAP model: Loaded through a PyFunc wrapper

Usage:
    python lse_operator_mlflow.py
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


def get_model(model_name):
    """
    Load a model from MLflow by its name using PyFunc wrapper
    
    Parameters:
    -----------
    model_name : str
        Name of the registered model in MLflow
    
    Returns:
    --------
    model : mlflow.pyfunc.PyFuncModel
        The loaded model wrapper from MLflow
    """
    # Set MLflow tracking URI and credentials
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Get latest version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.error(f"No versions found for model {model_name}")
            return None
            
        latest_version = max([mv.version for mv in versions])
        model_uri = f"models:/{model_name}/{latest_version}"
        
        # Get model info to check tags
        model_info = client.get_model_version(model_name, latest_version)
        model_tags = {}
        if hasattr(model_info, "tags") and model_info.tags:
            model_tags = model_info.tags
            
        # Show model type if available
        model_type = model_tags.get("model_type", "unknown")
        logger.info(f"Loading {model_type} model via PyFunc wrapper: {model_name}")
        
        # Load via PyFunc wrapper
        wrapper = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✓ Successfully loaded {model_type} model: {model_name}, version: {latest_version}")
        
        return wrapper
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Get model names from command line arguments or environment variables
    autoencoder_model_name = "smi_autoencoder_model_wrapper"
    dimred_model_name = "smi_umap_model_wrapper"
    
    # Load the autoencoder model with wrapper
    logger.info(f"Loading autoencoder model: {autoencoder_model_name}")
    autoencoder_wrapper = get_model(autoencoder_model_name)
    
    if autoencoder_wrapper is None:
        logger.error("Failed to load autoencoder model. Exiting.")
        sys.exit(1)
    
    # Load the umap model with wrapper
    logger.info(f"Loading UMAP model from MLflow: {dimred_model_name}")
    umap_wrapper = get_model(dimred_model_name)

    if umap_wrapper is None:
        logger.error("Failed to load UMAP model. Exiting.")
        sys.exit(1)
    
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
            
            # Transform the datapoint
            tensor = transform(datapoint).numpy()
            
            # Process with autoencoder to get latent features using wrapper
            autoencoder_result = autoencoder_wrapper.predict({"image": tensor})
            latent_features = autoencoder_result["latent_features"]
            
            # Apply dimension reduction using wrapper
            umap_result = umap_wrapper.predict({"latent": latent_features})
            f_vec = umap_result["umap_coords"]
            
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