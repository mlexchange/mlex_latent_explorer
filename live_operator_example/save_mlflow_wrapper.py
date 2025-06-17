#!/usr/bin/env python
"""
MLflow Model Saving Utility

This script saves autoencoder and dimensionality reduction models to MLflow:
- Autoencoder model: Using PyFunc wrapper with get_latent_features functionality
- Dimensionality reduction model: Using PyFunc wrapper
"""

import os
import sys
import traceback

# Fix transformers compatibility BEFORE any imports
os.environ["TRANSFORMERS_USE_TORCH_EXPORT"] = "0"

from dotenv import load_dotenv
import mlflow

# Import wrappers and saving functions
from vit_wrapper import save_vit_model_with_wrapper
from umap_wrapper import save_umap_model_with_wrapper

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env') 

# MLflow Configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
# Names for logging experiment and model
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "smi_exp")
MLFLOW_AUTO_MODEL_NAME = os.getenv("MLFLOW_AUTO_MODEL_NAME", "smi_auto")
MLFLOW_DR_MODEL_NAME = os.getenv("MLFLOW_DR_MODEL_NAME", "smi_dr")

# Load source files
AUTOENCODER_WEIGHTS_PATH= os.getenv("AUTOENCODER_WEIGHTS_PATH")
AUTOENCODER_CODE_PATH= os.getenv("AUTOENCODER_CODE_PATH")
DR_WEIGHTS_PATH= os.getenv("DR_WEIGHTS_PATH")
LATENT_DIM= os.getenv("LATENT_DIM")

# Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# Print configuration for verification
print("----------------------------------------------")
print("MLFLOW_TRACKING_URI:",MLFLOW_TRACKING_URI)
print("MLFLOW_EXPERIMENT_NAME:",MLFLOW_EXPERIMENT_NAME)
print("MLFLOW_AUTO_MODEL_NAME:",MLFLOW_AUTO_MODEL_NAME)
print("MLFLOW_DR_MODEL_NAME:",MLFLOW_DR_MODEL_NAME)
print("AUTOENCODER_WEIGHTS_PATH:",AUTOENCODER_WEIGHTS_PATH)
print("AUTOENCODER_CODE_PATH:",AUTOENCODER_CODE_PATH)
print("DR_WEIGHTS_PATH:",DR_WEIGHTS_PATH)
print("LATENT_DIM:",LATENT_DIM)
print("----------------------------------------------")

# Model configurations
MODEL_CONFIG = {
    "name": "SMI_Autoencoder",
    "state_dict": AUTOENCODER_WEIGHTS_PATH,
    "python_class": "Autoencoder",
    "python_file": AUTOENCODER_CODE_PATH,
    "type": "torch",
    "latent_dim": LATENT_DIM
}

JOBLIB_CONFIG = {
    "name": "SMI_DimRed",
    "file": DR_WEIGHTS_PATH,
    "type": "joblib"
}

if __name__ == "__main__":
    try:
        # Check if MLflow server is accessible
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.search_experiments()
            print(f"✓ MLflow server accessible at {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"⚠️  Cannot connect to MLflow server at {MLFLOW_TRACKING_URI}: {e}")
            sys.exit(1)
        
        print(f"\nSaving both models with names:")
        print(f"  Autoencoder model: {MLFLOW_AUTO_MODEL_NAME}")
        print(f"  Dimensionality reduction model: {MLFLOW_DR_MODEL_NAME}")
        
        # Save autoencoder model with PyFunc wrapper
        auto_name, auto_run_id = save_vit_model_with_wrapper(
            MODEL_CONFIG, 
            MLFLOW_TRACKING_URI,
            MLFLOW_EXPERIMENT_NAME,
            MLFLOW_AUTO_MODEL_NAME
        )
        
        # Save dimensionality reduction model with wrapper
        dr_name, dr_run_id = save_umap_model_with_wrapper(
            JOBLIB_CONFIG,
            MLFLOW_TRACKING_URI,
            MLFLOW_EXPERIMENT_NAME, 
            MLFLOW_DR_MODEL_NAME
        )
        
        # Report results
        if auto_name and dr_name:
            print(f"\n✅ Both models saved successfully!")
        elif auto_name:
            print(f"\n✅ Autoencoder model saved successfully!")
            print(f"⚠️ Dimensionality reduction model failed to save.")
        elif dr_name:
            print(f"\n✅ Dimensionality reduction model saved successfully!")
            print(f"⚠️ Autoencoder model failed to save.")
        else:
            print(f"\n⚠️ Failed to save both models.")
        
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()