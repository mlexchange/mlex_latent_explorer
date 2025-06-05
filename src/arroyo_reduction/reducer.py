from abc import ABC, abstractmethod
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import redis

from arroyosas.schemas import RawFrameEvent

from src.utils.mlflow_client import load_model
from .redis_model_store import RedisModelStore

logger = logging.getLogger(__name__)

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

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

class LatentSpaceReducer:
    """
    Responsible for taking an image, encoding it into a
    latent space, and reducing it to 2D
    """

    def __init__(self):
        """Initialize the reducer with models from Redis"""
        # Initialize Redis model store
        self.model_store = RedisModelStore(host=REDIS_HOST, port=REDIS_PORT)
        
        # Get model selections from Redis
        self.autoencoder_model_name = self.model_store.get_autoencoder_model()
        self.dimred_model_name = self.model_store.get_dimred_model()
        
        logger.info(f"Using autoencoder model: {self.autoencoder_model_name}")
        logger.info(f"Using dimension reduction model: {self.dimred_model_name}")
        
        # # Initialize MLflow client
        # self.mlflow_client = MLflowClient()
        
        # Check for CUDA else use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        self.device = device
        
        # Load models from MLflow
        self.current_torch_model = load_model(self.autoencoder_model_name)
        self.current_dim_reduction_model = load_model(self.dimred_model_name)
        self.current_transform = self.get_transform()
        
        # Subscribe to model update channel if supported
        self._subscribe_to_model_updates()

    def reduce(self, message: RawFrameEvent) -> np.ndarray:
        """Process an image through the models to get feature vectors"""
        # 1. Encode the image into a latent space
        pil = Image.fromarray(message.image.array.astype(np.float32))
        tensor = self.current_transform(pil)
        logger.debug("Encoding image into latent space")
        
        # Convert to numpy for PyFunc models
        tensor_np = tensor.numpy()
        
        if self.current_torch_model is None:
            logger.error("No autoencoder model loaded")
            return np.zeros((1, 2))  # Return empty vector if model not loaded
        
        # Process with autoencoder to get latent features using wrapper
        autoencoder_result = self.current_torch_model.predict({"image": tensor_np})
        latent_features = autoencoder_result["latent_features"]
        
        if self.current_dim_reduction_model is None:
            logger.error("No dimension reduction model loaded")
            return latent_features  # Return latent features if no reduction model
        
        # Apply dimension reduction using wrapper
        umap_result = self.current_dim_reduction_model.predict({"latent": latent_features})
        f_vec = umap_result["umap_coords"]
        
        logger.debug(f"Reduced latent space to {f_vec.shape}")
        return f_vec

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Resize to smaller dimensions to save memory
                transforms.ToTensor(),  # Convert image to PyTorch tensor (0-1 range)
                transforms.Normalize(
                    (0.0,), (1.0,)
                ),  # Normalize tensor to have mean 0 and std 1
            ]
        )
    
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
            
            logger.info(f"Received model update: {model_type} = {model_name}")
            
            # Update the appropriate model
            if model_type == "autoencoder":
                self.autoencoder_model_name = model_name
                self.current_torch_model = self.mlflow_client.load_model(model_name)
            elif model_type == "dimred":
                self.dimred_model_name = model_name
                self.current_dim_reduction_model = self.mlflow_client.load_model(model_name)
            else:
                logger.warning(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"Error handling model update: {e}")

