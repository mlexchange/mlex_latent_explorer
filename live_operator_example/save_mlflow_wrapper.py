#!/usr/bin/env python
"""
MLflow Model Saving Utility

This script saves either a ViT autoencoder or VAE model to MLflow, along with their
corresponding dimensionality reduction model:
- Autoencoder/VAE model: Using PyFunc wrapper with get_latent_features functionality
- Dimensionality reduction model: Using PyFunc wrapper

The model type (VIT or VAE) is determined by the AUTOENCODER_TYPE environment variable.
"""

import os
import sys
import traceback

# Fix transformers compatibility BEFORE any imports
os.environ["TRANSFORMERS_USE_TORCH_EXPORT"] = "0"

import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Determine which model wrappers to import based on autoencoder type
AUTOENCODER_TYPE = os.getenv("AUTOENCODER_TYPE", "VIT").upper()
if AUTOENCODER_TYPE == "VIT":
    from vit_wrapper import save_vit_model_with_wrapper as save_model_fn
elif AUTOENCODER_TYPE == "VAE":
    from vae_wrapper import save_vae_model_with_wrapper as save_model_fn
else:
    print(f"⚠️ Invalid AUTOENCODER_TYPE: {AUTOENCODER_TYPE}. Must be 'VIT' or 'VAE'.")
    sys.exit(1)

# Always import UMAP wrapper
from umap_wrapper import save_umap_model_with_wrapper

# MLflow Configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# Load model-specific configuration based on autoencoder type
if AUTOENCODER_TYPE == "VIT":
    # For ViT model, use VIT-specific variables
    AUTOENCODER_WEIGHTS_PATH = os.getenv("VIT_WEIGHTS_PATH")
    AUTOENCODER_CODE_PATH = os.getenv("VIT_CODE_PATH")
    DR_WEIGHTS_PATH = os.getenv("VIT_DR_WEIGHTS_PATH")
    LATENT_DIM = os.getenv("VIT_LATENT_DIM")
    IMAGE_SIZE_STR = os.getenv("VIT_IMAGE_SIZE", "(512, 512)")
    MLFLOW_EXPERIMENT_NAME = os.getenv("VIT_MLFLOW_EXPERIMENT_NAME")
    MLFLOW_AUTO_MODEL_NAME = os.getenv("VIT_MLFLOW_AUTO_MODEL_NAME")
    MLFLOW_DR_MODEL_NAME = os.getenv("VIT_MLFLOW_DR_MODEL_NAME")
else:  # AUTOENCODER_TYPE == "VAE"
    # For VAE model, use VAE-specific variables
    AUTOENCODER_WEIGHTS_PATH = os.getenv("VAE_WEIGHTS_PATH")
    AUTOENCODER_CODE_PATH = os.getenv("VAE_CODE_PATH")
    DR_WEIGHTS_PATH = os.getenv("VAE_DR_WEIGHTS_PATH")
    LATENT_DIM = os.getenv("VAE_LATENT_DIM")
    IMAGE_SIZE_STR = os.getenv("VAE_IMAGE_SIZE", "(512, 512)")
    MLFLOW_EXPERIMENT_NAME = os.getenv("VAE_MLFLOW_EXPERIMENT_NAME")
    MLFLOW_AUTO_MODEL_NAME = os.getenv("VAE_MLFLOW_AUTO_MODEL_NAME")
    MLFLOW_DR_MODEL_NAME = os.getenv("VAE_MLFLOW_DR_MODEL_NAME")

# Parse IMAGE_SIZE for both models
try:
    # Convert string to tuple
    IMAGE_SIZE = eval(IMAGE_SIZE_STR)
except Exception as e:
    print(f"⚠️ Error parsing IMAGE_SIZE: {e}. Using default (512, 512).")
    IMAGE_SIZE = (512, 512)


# Print configuration for verification
print("----------------------------------------------")
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("AUTOENCODER_TYPE:", AUTOENCODER_TYPE)
print("MLFLOW_EXPERIMENT_NAME:", MLFLOW_EXPERIMENT_NAME)
print("MLFLOW_AUTO_MODEL_NAME:", MLFLOW_AUTO_MODEL_NAME)
print("MLFLOW_DR_MODEL_NAME:", MLFLOW_DR_MODEL_NAME)
print("AUTOENCODER_WEIGHTS_PATH:", AUTOENCODER_WEIGHTS_PATH)
print("AUTOENCODER_CODE_PATH:", AUTOENCODER_CODE_PATH)
print("DR_WEIGHTS_PATH:", DR_WEIGHTS_PATH)
print("LATENT_DIM:", LATENT_DIM)
print("IMAGE_SIZE:", IMAGE_SIZE)
print("----------------------------------------------")

# Configure model with a single configuration structure for both types
MODEL_CONFIG = {
    "name": "SMI_Autoencoder",
    "state_dict": AUTOENCODER_WEIGHTS_PATH,
    "python_class": "Autoencoder",
    "python_file": AUTOENCODER_CODE_PATH,
    "type": "torch",
    "latent_dim": LATENT_DIM,
    "image_size": IMAGE_SIZE,
}

# UMAP configuration with consistent name
JOBLIB_CONFIG = {"name": "SMI_DimRed", "file": DR_WEIGHTS_PATH, "type": "joblib", "input_dim": LATENT_DIM}

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

        print(f"\nSaving models with names:")
        print(f"  {AUTOENCODER_TYPE} model: {MLFLOW_AUTO_MODEL_NAME}")
        print(f"  Dimensionality reduction model: {MLFLOW_DR_MODEL_NAME}")

        # Initialize success trackers
        auto_success = False
        dr_success = False

        # Save model with PyFunc wrapper
        if AUTOENCODER_WEIGHTS_PATH and AUTOENCODER_CODE_PATH:
            auto_name, auto_run_id = save_model_fn(
                MODEL_CONFIG,
                MLFLOW_TRACKING_URI,
                MLFLOW_EXPERIMENT_NAME,
                MLFLOW_AUTO_MODEL_NAME,
            )
            auto_success = bool(auto_name)
        else:
            print(f"\n⚠️ Skipping {AUTOENCODER_TYPE} model: missing configuration")

        # Save dimensionality reduction model with wrapper
        if DR_WEIGHTS_PATH:
            dr_name, dr_run_id = save_umap_model_with_wrapper(
                JOBLIB_CONFIG,
                MLFLOW_TRACKING_URI,
                MLFLOW_EXPERIMENT_NAME,
                MLFLOW_DR_MODEL_NAME,
            )
            dr_success = bool(dr_name)
        else:
            print("\n⚠️ Skipping dimensionality reduction model: missing configuration")

        # Report results
        print("\n---------- SUMMARY ----------")
        if auto_success and dr_success:
            print(f"\n✅ Both models saved successfully!")
        elif auto_success:
            print(f"\n✅ {AUTOENCODER_TYPE} model saved successfully!")
            print(f"⚠️ Dimensionality reduction model failed to save.")
        elif dr_success:
            print(f"\n✅ Dimensionality reduction model saved successfully!")
            print(f"⚠️ {AUTOENCODER_TYPE} model failed to save.")
        else:
            print(f"\n⚠️ Failed to save both models.")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
