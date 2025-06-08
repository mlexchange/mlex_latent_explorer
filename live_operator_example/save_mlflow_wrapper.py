#!/usr/bin/env python
"""
MLflow Model Saving Utility

This script saves autoencoder and dimensionality reduction models to MLflow:
- Autoencoder model: Using PyFunc wrapper with get_latent_features functionality
- Dimensionality reduction model: Using PyFunc wrapper (as in the original script)
"""

import os
import sys
import time
import json
from datetime import datetime
import tempfile
import shutil

# Fix transformers compatibility BEFORE any imports
os.environ["TRANSFORMERS_USE_TORCH_EXPORT"] = "0"

import mlflow
import numpy as np
import torch
import joblib

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(dotenv_path='../.env') 


import torch
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device

# MLflow Configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
# Names for logging experiment and model
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "smi_exp")
MLFLOW_AUTO_MODEL_NAME = os.getenv("MLFLOW_AUTO_MODEL_NAME", "smi_auto")
MLFLOW_DR_MODEL_NAME = os.getenv("MLFLOW_DR_MODEL_NAME", "smi_dr")

# Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# Load source files
AUTOENCODER_WEIGHTS_PATH= os.getenv("AUTOENCODER_WEIGHTS_PATH")
AUTOENCODER_CODE_PATH= os.getenv("AUTOENCODER_CODE_PATH")
DR_WEIGHTS_PATH= os.getenv("DR_WEIGHTS_PATH")
LATENT_DIM= os.getenv("LATENT_DIM")
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


def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


def load_vit_model_from_npz(npz_path, latent_dim=64):
    """Load the ViT model from an NPZ file"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ weights file not found: {npz_path}")
    
    # Import the model class dynamically
    import importlib.util
    import sys
    
    # Get the full path to the module
    module_path = os.path.abspath(MODEL_CONFIG["python_file"])
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(MODEL_CONFIG["python_file"]).replace('.py', '')
    
    # Add the directory to sys.path if it's not already there
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    # Check if the module is already imported
    if module_name in sys.modules:
        ViT = sys.modules[module_name]
    else:
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        ViT = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = ViT
        spec.loader.exec_module(ViT)
    
    # Create model instance
    print(f"Initializing Autoencoder with latent_dim={latent_dim}")
    model = ViT.Autoencoder(latent_dim=latent_dim)
    
    # Load weights
    print(f"Loading weights from {npz_path}")
    weights = np.load(npz_path, allow_pickle=True)
    
    # Check for state_dict key
    if 'state_dict' in weights.files:
        state_dict = weights['state_dict'].item()
    else:
        state_dict = {k: weights[k] for k in weights.files}
    
    # Convert numpy arrays to PyTorch tensors
    print("Converting weights to PyTorch tensors")
    torch_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            torch_state_dict[k] = torch.from_numpy(v)
        else:
            torch_state_dict[k] = v
    
    # Load state dict
    try:
        model.load_state_dict(torch_state_dict, strict=True)
        print("✓ Model weights loaded successfully (strict)")
    except Exception as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        
        result = model.load_state_dict(torch_state_dict, strict=False)
        if result.missing_keys:
            print(f"Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys[:5]}...")
    
    # Set model to eval mode
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


# PyFunc wrapper for ViT Autoencoder
class VitAutoencoderWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for ViT Autoencoder with direct model access and latent features functionality
    """
    
    def __init__(self, latent_dim=64):
        self.model = None
        # Explicitly convert to integer to avoid type issues
        self.latent_dim = int(latent_dim) if latent_dim is not None else 64
        
    def load_context(self, context):
        """Load ViT model from context artifacts"""
        import torch
        import os
        import sys
        import importlib.util
        import numpy as np
        from torchvision import transforms
        from PIL import Image

        # Add compatibility patch for torch if needed
        if not hasattr(torch, "get_default_device"):
            def get_default_device():
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.get_default_device = get_default_device
        
        # Get the model code path
        model_code_path = context.artifacts.get("model_code")
        if not model_code_path or not os.path.exists(model_code_path):
            raise FileNotFoundError(f"Model code not found at {model_code_path}")
        
        # Add directory to sys.path if needed
        model_dir = os.path.dirname(model_code_path)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        # Import module
        module_name = os.path.basename(model_code_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, model_code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get latent dimension
        latent_dim = self.latent_dim
        print(f"Latent Dimension is: {latent_dim}")
        
        # Create model instance with explicit integer for latent_dim
        self.model = module.Autoencoder(latent_dim=int(latent_dim))
        
        # Load weights from NPZ file
        weights_path = context.artifacts.get("weights_path")
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        # Load weights
        weights = np.load(weights_path, allow_pickle=True)
        
        # Check for state_dict key
        if 'state_dict' in weights.files:
            state_dict = weights['state_dict'].item()
        else:
            state_dict = {k: weights[k] for k in weights.files}
        
        # Convert numpy arrays to PyTorch tensors
        torch_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                torch_state_dict[k] = torch.from_numpy(v)
            else:
                torch_state_dict[k] = v
        
        # Load state dict
        try:
            self.model.load_state_dict(torch_state_dict, strict=True)
            print("✓ Model weights loaded successfully (strict)")
        except Exception as e:
            print(f"⚠️ Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            
            result = self.model.load_state_dict(torch_state_dict, strict=False)
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
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to smaller dimensions to save memory
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Convert image to PyTorch tensor (0-1 range)
            transforms.Normalize((0.0,), (1.0,)),  # Normalize tensor to have mean 0 and std 1
        ])
        
        print(f"✓ ViT model loaded successfully with latent_dim={latent_dim}")
    
    def predict(self, context, model_input):
        """
        Standard predict method (required by MLflow)
        
        This method processes the input through the autoencoder model and returns both
        the reconstruction and the latent features from the encoder.
        
        Args:
            context: MLflow context
            model_input: Input data as 2D/3D numpy array (H,W) or (H,W,C)
            
        Returns:
            Dictionary with reconstruction and latent features
        """
        import numpy as np
        import torch
        from PIL import Image
        
        if self.model is None:
            raise RuntimeError("ViT model not loaded. Call load_context first.")
        
        # Validate input
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Input must be a numpy array, got {type(model_input)}")
            
        # Check input dimensions
        if not (len(model_input.shape) == 2 or len(model_input.shape) == 3):
            raise ValueError(f"Input must be a 2D or 3D array (H,W) or (H,W,C), got shape {model_input.shape}")
        
        
        # Check and handle input dtype
        if model_input.dtype == np.uint8:
            # uint8 is already the preferred format, no conversion needed
            img_array = model_input
        elif model_input.dtype == np.float32:
            # Convert float32 to uint8 with appropriate scaling
            if model_input.max() <= 1.0:
                # Scale from 0-1 to 0-255
                img_array = (model_input * 255).astype(np.uint8)
            else:
                # Clip to 0-255 range and convert
                img_array = np.clip(model_input, 0, 255).astype(np.uint8)
        else:
            # Raise exception for unsupported dtypes
            raise ValueError(f"Input must be uint8 or float32, got {model_input.dtype}")
        
        try:
            # Convert numpy array to PIL Image
            # PIL.Image.fromarray handles both 2D and 3D arrays automatically
            pil_image = Image.fromarray(img_array)

            # Apply transformations (resize, convert to tensor, normalize)
            tensor = self.transform(pil_image)
            
            # Add batch dimension and move to device
            tensor = tensor.unsqueeze(0).to(self.device)
            
        except Exception as e:
            raise ValueError(f"Failed to process input image: {e}")
        
        # Process with model
        with torch.no_grad():
            # Get reconstruction
            reconstruction = self.model(tensor)
            reconstruction_np = reconstruction.cpu().numpy()
            
            # Get latent features directly from the encoder
            latent, _ = self.model.encoder(tensor)
            latent_features = latent.cpu().numpy()
        
        # Return results
        return {
            "reconstruction": reconstruction_np,
            "latent_features": latent_features
        }

def save_vit_model_with_wrapper(model_name=None):
    """Save autoencoder model using PyFunc wrapper with latent features functionality"""
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Set model name
    if model_name is None:
        model_name = f"{MODEL_CONFIG['name']}_v{datetime.now().strftime('%Y%m%d')}"
    
    print(f"\nSaving autoencoder model with PyFunc wrapper as: {model_name}")
    
    start_time = time.time()
    
    with mlflow.start_run(run_name=f"auto_model_wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        
        # Check file size and existence
        if not os.path.exists(MODEL_CONFIG["state_dict"]):
            print(f"❌ Error: Weights file not found at {MODEL_CONFIG['state_dict']}")
            return None, None
            
        if not os.path.exists(MODEL_CONFIG["python_file"]):
            print(f"❌ Error: Model code file not found at {MODEL_CONFIG['python_file']}")
            return None, None
            
        npz_size = get_file_size_mb(MODEL_CONFIG["state_dict"])
        print(f"\nNPZ file size: {npz_size:.1f} MB")
        
        try:
            # Get latent dimension
            latent_dim = MODEL_CONFIG.get("latent_dim", 64)
            
            # Create model wrapper with latent dimension
            vit_wrapper = VitAutoencoderWrapper(latent_dim=latent_dim)
            
            # Log model information
            mlflow.log_params({
                "model_name": MODEL_CONFIG["name"],
                "model_type": MODEL_CONFIG["type"],
                "python_class": MODEL_CONFIG["python_class"],
                "latent_dim": latent_dim,
                "npz_size_mb": npz_size,
                "using_wrapper": True,
            })
            
            # Set tags
            mlflow.set_tags({
                "model_type": "autoencoder"
            })
            
            # Create artifacts dictionary - only include files
            artifacts = {
                "weights_path": MODEL_CONFIG["state_dict"],
                "model_code": MODEL_CONFIG["python_file"]
            }
            
            # Define explicit requirements
            pip_requirements = ["torch==2.2.2", "numpy", "mlflow==2.14.3", "Pillow", "torchvision"]
            
            # Log the autoencoder model with PyFunc wrapper
            print("\nLogging autoencoder model with PyFunc wrapper to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=vit_wrapper,
                artifacts=artifacts,
                registered_model_name=model_name,
                pip_requirements=pip_requirements
            )
            
            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)
            
            print(f"\n✅ Autoencoder model saved with PyFunc wrapper in {total_time:.1f}s!")
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
            
            return model_name, run.info.run_id
            
        except Exception as e:
            print(f"\n❌ Error saving autoencoder model with wrapper: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# UMAP ModelWrapper class (keep original implementation)
class UMAPModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for UMAP model with direct model access"""
    
    def __init__(self):
        self.model = None
    
    def load_context(self, context):
        """Load UMAP model from context artifacts"""
        import joblib
        
        # Load UMAP model
        umap_path = context.artifacts.get("umap_model")
        if not umap_path or not os.path.exists(umap_path):
            raise FileNotFoundError(f"UMAP model not found in artifacts")
        
        print(f"Loading UMAP model from {umap_path}")
        self.model = joblib.load(umap_path)
        print("✓ UMAP model loaded successfully")
    
    def predict(self, context, model_input):
        """
        Standard predict method for UMAP model
        
        Args:
            context: MLflow context
            model_input: Input data as numpy array
            
        Returns:
            Dictionary with UMAP coordinates
        """
        import numpy as np
        
        if self.model is None:
            raise RuntimeError("UMAP model not loaded. Call load_context first.")
        
        # Validate input
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Input must be a numpy array, got {type(model_input)}")
            
        # Check input dimensions
        if len(model_input.shape) != 2 or model_input.shape[0] != 1:
            raise ValueError(f"Input must be a 2D array with shape (1, latent_dim), got shape {model_input.shape}")
        
        # Apply UMAP transformation
        umap_coords = self.model.transform(model_input)
        
        # Return results
        return {
            "umap_coords": umap_coords
        }


def save_umap_model(model_name=None):
    """Save dimensionality reduction model with a direct access wrapper and a registered name"""
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Set model name
    if model_name is None:
        model_name = f"{JOBLIB_CONFIG['name']}_v{datetime.now().strftime('%Y%m%d')}"
    
    print(f"\nSaving dimensionality reduction model as: {model_name}")
    
    start_time = time.time()
    
    with mlflow.start_run(run_name=f"dr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        
        # Check file existence
        if not os.path.exists(JOBLIB_CONFIG["file"]):
            print(f"⚠️ Dimensionality reduction model file not found at {JOBLIB_CONFIG['file']}")
            return None, None
            
        # Check file size
        joblib_size = get_file_size_mb(JOBLIB_CONFIG["file"])
        print(f"\nFile size: {joblib_size:.1f} MB")
        
        try:
            # Create model wrapper
            umap_wrapper = UMAPModelWrapper()
            
            # Log parameters
            mlflow.log_params({
                "model_name": JOBLIB_CONFIG["name"],
                "model_type": JOBLIB_CONFIG["type"],
                "joblib_size_mb": joblib_size,
                "using_wrapper": True,
            })
            
            # Set tags
            mlflow.set_tags({
                "model_type": "dimension_reduction"
            })
            
            # Create artifacts dictionary
            artifacts = {
                "umap_model": JOBLIB_CONFIG["file"]
            }
            
            # Define explicit requirements
            pip_requirements = ["numpy", "scikit-learn", "joblib", "mlflow==2.14.3", "umap-learn"]
            
            # Log the dimensionality reduction model wrapper
            try:
                print("\nLogging dimensionality reduction model wrapper to MLflow...")
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=umap_wrapper,
                    artifacts=artifacts,
                    registered_model_name=model_name,
                    pip_requirements=pip_requirements
                )
                print(f"✓ Dimensionality reduction model wrapper registered as '{model_name}'")
            except Exception as e:
                print(f"⚠️ Error logging dimensionality reduction model wrapper: {e}")
                import traceback
                traceback.print_exc()
                return None, None
            
            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)
            
            print(f"\n✅ Dimensionality reduction model saved successfully in {total_time:.1f}s!")
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
            
            return model_name, run.info.run_id
            
        except Exception as e:
            print(f"\n❌ Error saving dimensionality reduction model: {e}")
            import traceback
            traceback.print_exc()
            return None, None


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
        
        # Save autoencoder model with PyFunc wrapper (replacing direct log)
        auto_name, auto_run_id = save_vit_model_with_wrapper(MLFLOW_AUTO_MODEL_NAME)
        
        # Save dimensionality reduction model with wrapper (keep as original)
        dr_name, dr_run_id = save_umap_model(MLFLOW_DR_MODEL_NAME)
        
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
        import traceback
        traceback.print_exc()