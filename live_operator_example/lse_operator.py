import os
import sys
import logging
import numpy as np
import torch
from tiled.client import from_uri
from tiled_utils import write_results

# Import the MLflowClient class
from src.utils.mlflow_utils import MLflowClient

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

if __name__ == "__main__":
    # Create MLflow client
    mlflow_client = MLflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
        username=MLFLOW_TRACKING_USERNAME,
        password=MLFLOW_TRACKING_PASSWORD
    )
    
    # Check if MLflow is reachable
    if not mlflow_client.check_mlflow_ready():
        logger.error("MLflow server is not reachable. Exiting.")
        sys.exit(1)
    
    # Get model names from environment variables or command line arguments
    autoencoder_model_name = os.getenv("MLFLOW_AUTO_MODEL_NAME", "smi_autoencoder_model_wrapper")
    dimred_model_name = os.getenv("MLFLOW_DR_MODEL_NAME", "smi_umap_model_wrapper")
    
    # Load the autoencoder model with wrapper
    logger.info(f"Loading autoencoder model: {autoencoder_model_name}")
    autoencoder_wrapper = mlflow_client.load_model(autoencoder_model_name)
    
    if autoencoder_wrapper is None:
        logger.error("Failed to load autoencoder model. Exiting.")
        sys.exit(1)
    
    # Load the dimension reduction model with wrapper
    logger.info(f"Loading dimension reduction model from MLflow: {dimred_model_name}")
    dimred_wrapper = mlflow_client.load_model(dimred_model_name)

    if dimred_wrapper is None:
        logger.error("Failed to load dimension reduction model. Exiting.")
        sys.exit(1)
    
    # The PyFunc wrappers handle device management internally
    
    # No need for custom transforms as the PyFunc wrapper handles preprocessing

    # Set up Tiled clients
    data_client = from_uri(DATA_TILED_URI, api_key=DATA_TILED_KEY)
    write_client = from_uri(RESULTS_TILED_URI, api_key=RESULTS_TILED_API_KEY)

    # Processing loop
    logger.info("Starting processing loop...")
    for i in range(1000):  # TODO: Change to long running loop that detects new data
        try:
            # Get datapoint from Tiled
            datapoint = data_client[indx]  # noqa: F821
            
            # datapoint is a numpy array (the wrapper expects numpy input)
            img_array = datapoint
            
            # Log information about the image
            logger.info(f"Processing image {i}: shape={img_array.shape}, dtype={img_array.dtype}")
            
            # Process with autoencoder to get latent features
            # Pass the numpy array directly to the model wrapper
            autoencoder_result = autoencoder_wrapper.predict(img_array)
            latent_features = autoencoder_result["latent_features"]
            logger.info(f"Extracted latent features: shape={latent_features.shape}")
            
            # Apply dimension reduction using the UMAP wrapper
            umap_result = dimred_wrapper.predict(latent_features)
            f_vec = umap_result["umap_coords"]
            logger.info(f"UMAP coordinates: shape={f_vec.shape}")
            
            # Save results to Tiled
            write_results(
                write_client,
                f_vec,
                io_parameters,  # noqa: F821
                latent_vectors_path,  # noqa: F821
                metadata=None,
            )
            
            logger.info(f"Processed datapoint {i} successfully")
            
        except Exception as e:
            logger.error(f"Error processing datapoint {i}: {e}")
            import traceback
            traceback.print_exc()