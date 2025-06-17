import logging
import os
import sys
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
    def reduce(self, message: RawFrameEvent) -> np.ndarray:
        """
        Reduce the image to a feature vector.
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
        self.model_store = RedisModelStore(host=REDIS_HOST, port=REDIS_PORT)
        
        # Get model selections from Redis
        self.autoencoder_model_name = self.model_store.get_autoencoder_model()
        self.dimred_model_name = self.model_store.get_dimred_model()
        
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
        self.is_loading_model = True
        self.loading_model_type = "initial"
        
        try:
            self.current_torch_model = mlflow_client.load_model(self.autoencoder_model_name)
            self.current_dim_reduction_model = mlflow_client.load_model(self.dimred_model_name)
            logger.info("Initial models loaded successfully")
        finally:
            # Reset loading flags
            self.is_loading_model = False
            self.loading_model_type = None
        
        # Subscribe to model update channel if supported
        self._subscribe_to_model_updates()

    def reduce(self, message: RawFrameEvent) -> np.ndarray:
        """Process an image through the models to get feature vectors"""
        
        # Check if models are currently being loaded
        if self.is_loading_model:
            logger.info(f"Waiting for {self.loading_model_type} model to finish loading...")
            # Return a placeholder while models are loading
            return np.zeros((1, 2))  # Return empty vector during loading
            
        try:
            # Get numpy array from message
            img_array = message.image.array

            # Additional debugging for the image data
            logger.info(f"Get input image shape: {img_array.shape}, dtype: {img_array.dtype}. Image min: {img_array.min()}, max: {img_array.max()}")
            
        except Exception as e:
            logger.error(f"Error in image preparation: {e}")
            return np.zeros((1, 2))  # Return empty vector on error
        
        # Process with autoencoder to get latent features
        try:
            # Pass numpy array directly to model, the predict() API will handle data preprocessing 
            autoencoder_result = self.current_torch_model.predict(img_array)  
            latent_features = autoencoder_result["latent_features"]
            logger.info(f"Latent features shape: {latent_features.shape}")
            
        except Exception as e:
            logger.error(f"Error in autoencoder processing: {e}")
            return np.zeros((1, 2))  # Return empty vector on error
        
        # Apply dimension reduction directly with latent features
        try:            
            umap_result = self.current_dim_reduction_model.predict(latent_features)  
            f_vec = umap_result["umap_coords"]
            logger.info(f"Feature vector shape: {f_vec.shape}")
            return f_vec
        except Exception as e:
            logger.error(f"Error in dimension reduction: {e}")
            return np.zeros((1, 2))  # Return empty vector on error
            
    
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
        """Handle a model update from Redis PubSub"""
        try:
            model_type = update.get("model_type")
            model_name = update.get("model_name")
            
            if not model_type or not model_name:
                logger.warning(f"Invalid model update: {update}")
                return
            
            # Check if this is a duplicate update for the same model
            if (model_type == "autoencoder" and model_name == self.autoencoder_model_name) or \
            (model_type == "dimred" and model_name == self.dimred_model_name):
                logger.info(f"Ignoring duplicate model update: {model_type} = {model_name} (already loaded)")
                return
                
            logger.info(f"Received model update: {model_type} = {model_name}")
            
            # Set loading flags before updating models
            self.is_loading_model = True
            self.loading_model_type = model_type
            
            try:
                # Update the appropriate model
                if model_type == "autoencoder":
                    logger.info(f"Loading new autoencoder model: {model_name}...")
                    self.autoencoder_model_name = model_name
                    self.current_torch_model = self.mlflow_client.load_model(model_name)
                    logger.info(f"Successfully loaded new autoencoder model: {model_name}")
                elif model_type == "dimred":
                    logger.info(f"Loading new dimension reduction model: {model_name}...")
                    self.dimred_model_name = model_name
                    self.current_dim_reduction_model = self.mlflow_client.load_model(model_name)
                    logger.info(f"Successfully loaded new dimension reduction model: {model_name}")
                else:
                    logger.warning(f"Unknown model type: {model_type}")
            finally:
                # Reset loading flags
                self.is_loading_model = False
                self.loading_model_type = None
                logger.info(f"Model update complete. Ready to process images.")
        except Exception as e:
            # Ensure we reset flags even if there's an error
            self.is_loading_model = False
            self.loading_model_type = None
            logger.error(f"Error handling model update: {e}")

