#!/usr/bin/env python
"""
MLflow Model Saving Utility

This script saves either a ViT autoencoder or VAE model to MLflow, along with their
corresponding dimensionality reduction model:
- Autoencoder/VAE model: Using PyFunc wrapper with get_latent_features functionality
- Dimensionality reduction model: Using PyFunc wrapper

The model type (VIT or VAE) is determined by the configuration.
"""

import os
import sys
import traceback
import yaml
from pathlib import Path

# Fix transformers compatibility BEFORE any imports
os.environ["TRANSFORMERS_USE_TORCH_EXPORT"] = "0"

import mlflow
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Load MLflow configuration from YAML
CONFIG_PATH = Path(__file__).parent / "mlflow_config.yaml"
config = {}  # Initialize as empty dict
try:
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    print("✓ Loaded MLflow configuration from YAML")
except Exception as e:
    print(f"⚠️ Error loading configuration: {e}")
    sys.exit(1)


# Get autoencoder type from config
AUTOENCODER_TYPE = config.get("common", {}).get("autoencoder_type", "VIT").upper()
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
model_config_key = "vit" if AUTOENCODER_TYPE == "VIT" else "vae"
model_config = config.get("models", {}).get(model_config_key, {})

if model_config:
    # For the selected model, use values from YAML config
    AUTOENCODER_WEIGHTS_PATH = model_config.get("weights_path")
    AUTOENCODER_CODE_PATH = model_config.get("code_path")
    DR_WEIGHTS_PATH = model_config.get("dr_weights_path")
    LATENT_DIM = model_config.get("latent_dim")
    IMAGE_SIZE = tuple(model_config.get("image_size", [512, 512]))
    MLFLOW_EXPERIMENT_NAME = model_config.get("experiment_name")
    MLFLOW_AUTO_MODEL_NAME = model_config.get("auto_model_name")
    MLFLOW_DR_MODEL_NAME = model_config.get("dr_model_name")
else:
    print(f"❌ Error: No configuration found for {AUTOENCODER_TYPE} model in the YAML file.")
    sys.exit(1)


# Print configuration for verification
print("----------------------------------------------")
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("MLFLOW_TRACKING_USERNAME:", MLFLOW_TRACKING_USERNAME)
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
