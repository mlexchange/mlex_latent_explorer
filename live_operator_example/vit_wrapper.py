#!/usr/bin/env python
"""
ViT Autoencoder wrapper for MLflow.
Provides functionality to load a ViT autoencoder model and register it with MLflow.
"""

import os
import sys
import time
import traceback
import importlib.util
from datetime import datetime

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import mlflow


def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


# Add compatibility patch for torch if needed
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device


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


def save_vit_model_with_wrapper(model_config, tracking_uri, experiment_name, model_name=None):
    """
    Save autoencoder model using PyFunc wrapper with latent features functionality
    
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
    
    print(f"\nSaving autoencoder model with PyFunc wrapper as: {model_name}")
    
    start_time = time.time()
    
    with mlflow.start_run(run_name=f"auto_model_wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")
        
        # Check file size and existence
        if not os.path.exists(model_config["state_dict"]):
            print(f"❌ Error: Weights file not found at {model_config['state_dict']}")
            return None, None
            
        if not os.path.exists(model_config["python_file"]):
            print(f"❌ Error: Model code file not found at {model_config['python_file']}")
            return None, None
            
        npz_size = get_file_size_mb(model_config["state_dict"])
        print(f"\nNPZ file size: {npz_size:.1f} MB")
        
        try:
            # Get latent dimension
            latent_dim = model_config.get("latent_dim", 64)
            
            # Create model wrapper with latent dimension
            vit_wrapper = VitAutoencoderWrapper(latent_dim=latent_dim)
            
            # Log model information
            mlflow.log_params({
                "model_name": model_config["name"],
                "model_type": model_config["type"],
                "python_class": model_config["python_class"],
                "latent_dim": latent_dim,
                "npz_size_mb": npz_size,
                "using_wrapper": True,
            })
            
            # Set tags
            mlflow.set_tags({
                "exp_type": "live_mode",
                "model_type": "autoencoder"
            })
            
            # Create artifacts dictionary - only include files
            artifacts = {
                "weights_path": model_config["state_dict"],
                "model_code": model_config["python_file"]
            }
            
            # Define explicit requirements
            pip_requirements = ["torch==2.2.2", "numpy", "mlflow==2.22.0", "Pillow", "torchvision"]
            
            # Log the autoencoder model with PyFunc wrapper
            print("\nLogging autoencoder model with PyFunc wrapper to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=vit_wrapper,
                artifacts=artifacts,
                registered_model_name=model_name,
                pip_requirements=pip_requirements,
                code_path=[__file__]  # Include this file's path
            )
            
            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)
            
            print(f"\n✅ Autoencoder model saved with PyFunc wrapper in {total_time:.1f}s!")
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {tracking_uri}")
            
            return model_name, run.info.run_id
            
        except Exception as e:
            print(f"\n❌ Error saving autoencoder model with wrapper: {e}")
            traceback.print_exc()
            return None, None