#!/usr/bin/env python
"""
MLflow Model Saving Utility

This script saves either a ViT autoencoder or VAE model to MLflow, along with their
corresponding dimensionality reduction model:
- Autoencoder/VAE model: Using PyFunc wrapper with get_latent_features functionality
- Dimensionality reduction model: Using either:
  - Traditional UMAP (joblib) with PyFunc wrapper
  - Neural UMAP (PyTorch) with PyFunc wrapper

The model types are determined by the configuration.
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

# Get UMAP type from config
UMAP_TYPE = config.get("common", {}).get("umap_type", "joblib").lower()
if UMAP_TYPE == "joblib":
    from umap_wrapper import save_umap_model_with_wrapper as save_umap_fn
elif UMAP_TYPE == "neural":
    from neural_umap_wrapper import save_neural_umap_model_with_wrapper as save_umap_fn
else:
    print(f"⚠️ Invalid UMAP_TYPE: {UMAP_TYPE}. Must be 'joblib' or 'neural'.")
    sys.exit(1)

# Get save settings
SAVE_AUTOENCODER = config.get("save", {}).get("autoencoder", "true").lower() == "true"
SAVE_DR = config.get("save", {}).get("dr", "true").lower() == "true"

# MLflow Configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MLFLOW_TRACKING_INSECURE_TLS = os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "")

# Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = MLFLOW_TRACKING_INSECURE_TLS

# Load model-specific configuration based on autoencoder type
model_config_key = "vit" if AUTOENCODER_TYPE == "VIT" else "vae"
model_config = config.get("models", {}).get(model_config_key, {})

if model_config:
    # For the selected model, use values from YAML config
    AUTOENCODER_WEIGHTS_PATH = model_config.get("weights_path")
    AUTOENCODER_CODE_PATH = model_config.get("code_path")
    LATENT_DIM = model_config.get("latent_dim")
    IMAGE_SIZE = tuple(model_config.get("image_size", [512, 512]))
    MLFLOW_EXPERIMENT_NAME = model_config.get("experiment_name")
    MLFLOW_AUTO_MODEL_NAME = model_config.get("auto_model_name")
    
    # Get appropriate UMAP model name based on type
    if UMAP_TYPE == "joblib":
        MLFLOW_DR_MODEL_NAME = model_config.get("dr_model_name")
    else:  # neural
        MLFLOW_DR_MODEL_NAME = model_config.get("dr_neural_model_name")
else:
    print(f"❌ Error: No configuration found for {AUTOENCODER_TYPE} model in the YAML file.")
    sys.exit(1)

# Load dimensionality reduction configuration based on UMAP type
if UMAP_TYPE == "joblib":
    # For traditional UMAP, use the DR weights path from the autoencoder config
    DR_WEIGHTS_PATH = model_config.get("dr_weights_path")
    DR_CONFIG = {
        "name": "SMI_DimRed", 
        "file": DR_WEIGHTS_PATH, 
        "type": "joblib", 
        "input_dim": LATENT_DIM
    }
elif UMAP_TYPE == "neural":
    # For neural UMAP, use the specific neural UMAP fields from model config
    DR_CONFIG = {
        "name": "SMI_DimRed",
        "weights_path": model_config.get("dr_neural_weights_path"),
        "scaler_path": model_config.get("dr_neural_scaler_path"),
        "input_dim": LATENT_DIM  # Use same latent dim as autoencoder
    }

# Configure autoencoder model with a single configuration structure for both types
MODEL_CONFIG = {
    "name": "SMI_Autoencoder",
    "state_dict": AUTOENCODER_WEIGHTS_PATH,
    "python_class": "Autoencoder",
    "python_file": AUTOENCODER_CODE_PATH,
    "type": "torch",
    "latent_dim": LATENT_DIM,
    "image_size": IMAGE_SIZE,
}

# Print configuration for verification
print("----------------------------------------------")
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("MLFLOW_TRACKING_USERNAME:", MLFLOW_TRACKING_USERNAME)
print("AUTOENCODER_TYPE:", AUTOENCODER_TYPE)
print("UMAP_TYPE:", UMAP_TYPE)
print("SAVE_AUTOENCODER:", SAVE_AUTOENCODER)
print("SAVE_DR:", SAVE_DR)
print("MLFLOW_EXPERIMENT_NAME:", MLFLOW_EXPERIMENT_NAME)
print("MLFLOW_AUTO_MODEL_NAME:", MLFLOW_AUTO_MODEL_NAME)
print("MLFLOW_DR_MODEL_NAME:", MLFLOW_DR_MODEL_NAME)
print("AUTOENCODER_WEIGHTS_PATH:", AUTOENCODER_WEIGHTS_PATH)
print("AUTOENCODER_CODE_PATH:", AUTOENCODER_CODE_PATH)
print("LATENT_DIM:", LATENT_DIM)
print("IMAGE_SIZE:", IMAGE_SIZE)
if UMAP_TYPE == "joblib":
    print("DR_WEIGHTS_PATH:", DR_WEIGHTS_PATH)
elif UMAP_TYPE == "neural":
    print("NEURAL_UMAP_WEIGHTS_PATH:", DR_CONFIG["weights_path"])
    print("NEURAL_UMAP_SCALER_PATH:", DR_CONFIG["scaler_path"])
    print("NEURAL_UMAP_INPUT_DIM:", DR_CONFIG["input_dim"])
print("----------------------------------------------")

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

        print("\nModels to save:")
        if SAVE_AUTOENCODER:
            print(f"  {AUTOENCODER_TYPE} model: {MLFLOW_AUTO_MODEL_NAME}")
        if SAVE_DR:
            print(f"  {UMAP_TYPE.capitalize()} dimensionality reduction model: {MLFLOW_DR_MODEL_NAME}")
        
        if not (SAVE_AUTOENCODER or SAVE_DR):
            print("  No models selected for saving. Check the 'save' section in mlflow_config.yaml.")
            sys.exit(0)

        # Initialize success trackers
        auto_success = False
        dr_success = False

        # Save autoencoder model with PyFunc wrapper if enabled
        if SAVE_AUTOENCODER:
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
        else:
            print(f"\n⚠️ Skipping {AUTOENCODER_TYPE} model: disabled in configuration")

        # Save dimensionality reduction model with wrapper if enabled
        if SAVE_DR:
            if UMAP_TYPE == "joblib" and DR_CONFIG["file"]:
                dr_name, dr_run_id = save_umap_fn(
                    DR_CONFIG,
                    MLFLOW_TRACKING_URI,
                    MLFLOW_EXPERIMENT_NAME,
                    MLFLOW_DR_MODEL_NAME,
                )
                dr_success = bool(dr_name)
            elif UMAP_TYPE == "neural" and DR_CONFIG["weights_path"]:
                dr_name, dr_run_id = save_umap_fn(
                    DR_CONFIG,
                    MLFLOW_TRACKING_URI,
                    MLFLOW_EXPERIMENT_NAME,
                    MLFLOW_DR_MODEL_NAME,
                )
                dr_success = bool(dr_name)
            else:
                print(f"\n⚠️ Skipping {UMAP_TYPE} dimensionality reduction model: missing configuration")
        else:
            print(f"\n⚠️ Skipping {UMAP_TYPE} dimensionality reduction model: disabled in configuration")

        # Report results
        print("\n---------- SUMMARY ----------")
        if SAVE_AUTOENCODER and SAVE_DR:
            if auto_success and dr_success:
                print(f"\n✅ Both models saved successfully!")
            elif auto_success:
                print(f"\n✅ {AUTOENCODER_TYPE} model saved successfully!")
                print(f"❌ {UMAP_TYPE.capitalize()} dimensionality reduction model failed to save.")
            elif dr_success:
                print(f"\n✅ {UMAP_TYPE.capitalize()} dimensionality reduction model saved successfully!")
                print(f"❌ {AUTOENCODER_TYPE} model failed to save.")
            else:
                print(f"\n❌ Failed to save both models.")
        elif SAVE_AUTOENCODER:
            if auto_success:
                print(f"\n✅ {AUTOENCODER_TYPE} model saved successfully!")
            else:
                print(f"\n❌ {AUTOENCODER_TYPE} model failed to save.")
        elif SAVE_DR:
            if dr_success:
                print(f"\n✅ {UMAP_TYPE.capitalize()} dimensionality reduction model saved successfully!")
            else:
                print(f"\n❌ {UMAP_TYPE.capitalize()} dimensionality reduction model failed to save.")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()