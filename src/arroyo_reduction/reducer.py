import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import redis
import torch
import torchvision.transforms as transforms
from arroyosas.schemas import RawFrameEvent
from PIL import Image

from src.utils.mlflow_utils import MLflowClient

from .redis_model_store import RedisModelStore

logger = logging.getLogger("arroyo_reduction.reducer")
# logger.propagate = False  # Add this line to prevent propagation to parent loggers

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "kvrocks")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6666))

# Set Numba environment variables to avoid umap illegal instruction error
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")
# Check if JIT is disabled (set by user)
if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    # Apply conservative CPU settings for compatibility
    os.environ.setdefault("NUMBA_CPU_NAME", "generic")
    os.environ.setdefault("NUMBA_CPU_FEATURES", "+neon")
    logger.info("Numba JIT disabled - using conservative CPU settings for compatibility")
# Else let Numba detect CPU features automatically

# message = {
#     "tiled_uri": DATA_TILED_URI,
#     "index": index,
#     "feature_vector": latent_vector.tolist(),
# }

class Reducer(ABC):
    """
    Abstract base class for reducers.
    Reducers are responsible for taking an image, encoding it into a
    latent space, and saving the latent space to a Tiled dataset.
    """

    @abstractmethod
    def reduce(self, message: RawFrameEvent) -> tuple[np.ndarray, dict]:
        """
        Reduce the image to a feature vector and return timing information.
        Returns a tuple of (feature_vector, timing_info_dict)
        """
        pass

class LatentSpaceReducer(Reducer):
    """
    Responsible for taking an image, encoding it into a
    latent space, and reducing it to 2D
    """

    def __init__(self):
        """Initialize the reducer with models from Redis"""
        # Initialize model loading status flags
        self.is_loading_model = False
        self.loading_model_type = None
        
        # Initialize Redis model store
        self.redis_model_store = RedisModelStore(host=REDIS_HOST, port=REDIS_PORT)
        
        # Get model selections from Redis (may include version in "name:version" format)
        self.autoencoder_model_name = self.redis_model_store.get_autoencoder_model()
        self.dimred_model_name = self.redis_model_store.get_dimred_model()
        self.experiment_name = self.redis_model_store.get_experiment_name()
        
        logger.info(f"Using experiment name: {self.experiment_name}")
        logger.info(f"Using autoencoder model: {self.autoencoder_model_name}")
        logger.info(f"Using dimension reduction model: {self.dimred_model_name}")
        
        # Check for CUDA else use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        self.device = device
        
        # Load models from MLflow
        mlflow_client = MLflowClient()
        self.mlflow_client = mlflow_client  # Store for later use
        
        # Set loading flags before loading models
        self._update_loading_state(True, "initial")

        try:
            # Parse model name and version for autoencoder
            if self.autoencoder_model_name and ":" in self.autoencoder_model_name:
                auto_name, auto_version = self.autoencoder_model_name.split(":", 1)
                self.current_torch_model = mlflow_client.load_model(auto_name, version=auto_version)
            else:
                self.current_torch_model = mlflow_client.load_model(self.autoencoder_model_name)
            
            # Parse model name and version for dimred
            if self.dimred_model_name and ":" in self.dimred_model_name:
                dimred_name, dimred_version = self.dimred_model_name.split(":", 1)
                self.current_dim_reduction_model = mlflow_client.load_model(dimred_name, version=dimred_version)
            else:
                self.current_dim_reduction_model = mlflow_client.load_model(self.dimred_model_name)
            
            logger.info("Initial models loaded successfully")
        finally:
            # Reset loading flags
            self._update_loading_state(False)
        
        # Subscribe to model update channel if supported
        self._subscribe_to_model_updates()

    def _update_loading_state(self, is_loading, model_type=None):
        """
        Update loading state both locally and in Redis
        
        Args:
            is_loading (bool): Whether models are currently loading
            model_type (str, optional): Type of model being loaded if is_loading=True
        """
        # Update local state
        self.is_loading_model = is_loading
        self.loading_model_type = model_type if is_loading else None
        
        # Update Redis state
        try:
            if self.redis_model_store and self.redis_model_store.redis_client:
                self.redis_model_store.redis_client.set("model_loading_state", str(is_loading))
                self.redis_model_store.redis_client.set("loading_model_type", str(model_type or ""))
                logger.info(f"Updated loading state in Redis: is_loading={is_loading}, type={model_type or ''}")
        except Exception as e:
            logger.error(f"Error updating loading state in Redis: {e}")

    def reduce(self, message: RawFrameEvent) -> tuple[np.ndarray, dict]:
        """Process an image through the models to get feature vectors with timing information"""
        
        # Initialize timing dictionary
        timing_info = {
            'autoencoder_time': None,
            'dimred_time': None
        }
        
        # Check if models are currently being loading
        if self.is_loading_model:
            logger.info(f"Waiting for {self.loading_model_type} model to finish loading...")
            # Return a placeholder while models are loading
            return None, timing_info
            
        try:
            # Get numpy array from message
            img_array = message.image.array

            # Additional debugging for the image data
            logger.info(f"Get input image shape: {img_array.shape}, dtype: {img_array.dtype}. Image min: {img_array.min()}, max: {img_array.max()}")
            
        except Exception as e:
            logger.error(f"Error in image preparation: {e}")
            return None, timing_info
        
        # Process with autoencoder to get latent features with timing
        try:
            # Start timing autoencoder processing
            autoencoder_start = time.time()
            
            # Pass numpy array directly to model, the predict() API will handle data preprocessing 
            autoencoder_result = self.current_torch_model.predict(img_array)  
            latent_features = autoencoder_result["latent_features"]
            
            # End timing autoencoder processing
            autoencoder_end = time.time()
            timing_info['autoencoder_time'] = autoencoder_end - autoencoder_start
            
            logger.info(f"Latent features shape: {latent_features.shape}, processing time: {timing_info['autoencoder_time']:.4f}s")
            
        except Exception as e:
            logger.error(f"Error in autoencoder processing: {e}")
            return None, timing_info
        
        # Apply dimension reduction directly with latent features with timing
        try:
            # Start timing dimension reduction processing
            dimred_start = time.time()
            
            dimred_result = self.current_dim_reduction_model.predict(latent_features)  
            f_vec = dimred_result["coords"]
            
            # End timing dimension reduction processing
            dimred_end = time.time()
            timing_info['dimred_time'] = dimred_end - dimred_start
            
            logger.info(f"Feature vector shape: {f_vec.shape}, processing time: {timing_info['dimred_time']:.4f}s")
            
            return f_vec, timing_info
        except Exception as e:
            logger.error(f"Error in dimension reduction: {e}")
            return None, timing_info
    
    def _subscribe_to_model_updates(self):
        """
        Subscribe to model update notifications through Redis PubSub
        This runs in a separate thread to listen for model updates
        """
        try:
            import threading
            
            def listen_for_updates():
                """Listen for model updates in a separate thread"""
                try:
                    redis_client = redis.Redis(
                        host=REDIS_HOST, 
                        port=REDIS_PORT, 
                        decode_responses=True
                    )
                    pubsub = redis_client.pubsub()
                    pubsub.subscribe("model_updates")
                    
                    logger.info("Subscribed to model updates channel")
                    
                    # Listen for messages
                    for message in pubsub.listen():
                        if message["type"] == "message":
                            data = message["data"]
                            try:
                                import json
                                update = json.loads(data)
                                self._handle_model_update(update)
                            except Exception as e:
                                logger.error(f"Error processing model update: {e}")
                except Exception as e:
                    logger.error(f"Error in model update listener: {e}")
            
            # Start listener thread
            thread = threading.Thread(target=listen_for_updates, daemon=True)
            thread.start()
            logger.info("Started model update listener thread")
        except Exception as e:
            logger.warning(f"Could not start model update listener: {e}")
    

    def _handle_model_update(self, update):
        """Handle a model update from Redis PubSub with version support"""
        try:
            # NEW: Check if this is an experiment name update
            if update.get("update_type") == "experiment_name":
                new_experiment_name = update.get("experiment_name")
                logger.info(f"Received experiment name update: {new_experiment_name}")
                self.experiment_name = new_experiment_name
                return
            
            # Otherwise handle model updates as before
            model_type = update.get("model_type")
            model_id = update.get("model_name")
            
            if not model_type or not model_id:
                logger.warning(f"Invalid model update: {update}")
                return
            
            # Parse model name and version
            if ":" in model_id:
                model_name, model_version = model_id.split(":", 1)
            else:
                model_name = model_id
                model_version = None
            
            # Check if this is a duplicate update
            if model_type == "autoencoder":
                if self.autoencoder_model_name and ":" in self.autoencoder_model_name:
                    current_name, current_version = self.autoencoder_model_name.split(":", 1)
                else:
                    current_name = self.autoencoder_model_name
                    current_version = None
                
                if model_name == current_name and model_version == current_version:
                    logger.info(f"Ignoring duplicate autoencoder update: {model_id}")
                    self._update_loading_state(False)
                    return
                    
            elif model_type == "dimred":
                if self.dimred_model_name and ":" in self.dimred_model_name:
                    current_name, current_version = self.dimred_model_name.split(":", 1)
                else:
                    current_name = self.dimred_model_name
                    current_version = None
                
                if model_name == current_name and model_version == current_version:
                    logger.info(f"Ignoring duplicate dimred update: {model_id}")
                    self._update_loading_state(False)
                    return
                    
            logger.info(f"Received model update: {model_type} = {model_id}")
            
            # Set loading flags
            self._update_loading_state(True, model_type)
            
            try:
                # Update the appropriate model
                if model_type == "autoencoder":
                    logger.info(f"Loading new autoencoder model: {model_id}...")
                    # DON'T update name yet - wait until model is loaded
                    # self.autoencoder_model_name = model_id  # ← REMOVE THIS
                    
                    # Load model with version if specified
                    if model_version:
                        new_model = self.mlflow_client.load_model(
                            model_name, version=model_version
                        )
                    else:
                        new_model = self.mlflow_client.load_model(model_name)
                    
                    # Only update name AFTER successful load
                    self.current_torch_model = new_model
                    self.autoencoder_model_name = model_id  # ← MOVE HERE
                    logger.info(f"Successfully loaded new autoencoder model: {model_id}")
                    
                elif model_type == "dimred":
                    logger.info(f"Loading new dimension reduction model: {model_id}...")
                    # DON'T update name yet - wait until model is loaded
                    # self.dimred_model_name = model_id  # ← REMOVE THIS
                    
                    # Load model with version if specified
                    if model_version:
                        new_model = self.mlflow_client.load_model(
                            model_name, version=model_version
                        )
                    else:
                        new_model = self.mlflow_client.load_model(model_name)
                    
                    # Only update name AFTER successful load
                    self.current_dim_reduction_model = new_model
                    self.dimred_model_name = model_id  # ← MOVE HERE
                    logger.info(f"Successfully loaded new dimension reduction model: {model_id}")
                else:
                    logger.warning(f"Unknown model type: {model_type}")
            finally:
                # Reset loading flags
                self._update_loading_state(False)
                logger.info(f"Model update complete. Ready to process images.")
        except Exception as e:
            # Ensure we reset flags even if there's an error
            self._update_loading_state(False)                
            logger.error(f"Error handling model update: {e}")