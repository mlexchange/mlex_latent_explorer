import logging
import redis
import os
import json
import threading

logger = logging.getLogger(__name__)

class RedisModelStore:
    """
    Redis integration for model selections that supports both:
    1. Key-Value Store: For storing and retrieving model selections
    2. Pub/Sub: For real-time notification of model changes
    """
    
    # Redis Key Constants
    KEY_AUTOENCODER_MODEL = "selected_mlflow_model"
    KEY_DIMRED_MODEL = "selected_dim_reduction_model"
    
    # Redis Channel Constants
    CHANNEL_MODEL_UPDATES = "model_updates"
    
    def __init__(
        self, 
        host: str = None, 
        port: int = None, 
        password: str = None,
        decode_responses: bool = True
    ):
        """Initialize Redis client for key-value operations"""
        self.host = host or os.getenv("REDIS_HOST", "redis")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.password = password or os.getenv("REDIS_PASSWORD", None)
        
        # Initialize Redis client
        try:
            self.redis_client = redis.Redis(
                host=self.host, 
                port=self.port,
                password=self.password,
                decode_responses=decode_responses
            )
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            self.redis_client = None
            logger.warning(f"Could not connect to Redis: {e}")
    
    # =====================================================================
    # Key-Value Store Methods for Model Selection Storage
    # =====================================================================
    
    def store_autoencoder_model(self, model_name: str) -> bool:
        """Store autoencoder model name in Redis"""
        if self.redis_client is None:
            logger.warning("Redis client not available")
            return False
        
        try:
            logger.info(f"Storing autoencoder model: {model_name}")
            self.redis_client.set(self.KEY_AUTOENCODER_MODEL, model_name)
            
            # Also publish update notification
            self.publish_model_update("autoencoder", model_name)
            
            return True
        except Exception as e:
            logger.error(f"Error storing autoencoder model in Redis: {e}")
            return False
    
    def store_dimred_model(self, model_name: str) -> bool:
        """Store dimension reduction model name in Redis"""
        if self.redis_client is None:
            logger.warning("Redis client not available")
            return False
        
        try:
            logger.info(f"Storing dimension reduction model: {model_name}")
            self.redis_client.set(self.KEY_DIMRED_MODEL, model_name)
            
            # Also publish update notification
            self.publish_model_update("dimred", model_name)
            
            return True
        except Exception as e:
            logger.error(f"Error storing dimension reduction model in Redis: {e}")
            return False
    
    def get_autoencoder_model(self) -> str:
        """Get autoencoder model name from Redis"""
        if self.redis_client is None:
            logger.warning("Redis client not available")
            return None
        
        try:
            model_name = self.redis_client.get(self.KEY_AUTOENCODER_MODEL)
            logger.info(f"Retrieved autoencoder model: {model_name}")
            return model_name
        except Exception as e:
            logger.error(f"Error retrieving autoencoder model from Redis: {e}")
            return None
    
    def get_dimred_model(self) -> str:
        """Get dimension reduction model name from Redis"""
        if self.redis_client is None:
            logger.warning("Redis client not available")
            return None
        
        try:
            model_name = self.redis_client.get(self.KEY_DIMRED_MODEL)
            logger.info(f"Retrieved dimension reduction model: {model_name}")
            return model_name
        except Exception as e:
            logger.error(f"Error retrieving dimension reduction model from Redis: {e}")
            return None
    
    # =====================================================================
    # Pub/Sub Methods for Real-time Model Updates
    # =====================================================================
    
    def publish_model_update(self, model_type: str, model_name: str) -> bool:
        """
        Publish a model update notification to Redis
        
        Args:
            model_type: Type of model ('autoencoder' or 'dimred')
            model_name: Name of the selected model
            
        Returns:
            bool: Success status
        """
        if self.redis_client is None:
            logger.warning("Redis client not available for publishing model update")
            return False
        
        try:
            # Create message payload
            message = {
                "model_type": model_type,
                "model_name": model_name,
                "timestamp": import_time_module().time()
            }
            
            # Publish to Redis channel
            message_json = json.dumps(message)
            result = self.redis_client.publish(self.CHANNEL_MODEL_UPDATES, message_json)
            logger.info(f"Published model update: {message}, received by {result} subscribers")
            return True
        except Exception as e:
            logger.error(f"Error publishing model update to Redis: {e}")
            return False
    
    def subscribe_to_model_updates(self, callback):
        """
        Subscribe to model update notifications from Redis
        
        Args:
            callback: Function to call when a model update is received
        """
        if self.redis_client is None:
            logger.warning("Redis client not available for subscribing to model updates")
            return
        
        # Create a new thread for listening to Redis Pub/Sub
        def listener_thread():
            try:
                # Create a new Redis connection for Pub/Sub
                redis_client = redis.Redis(
                    host=self.host, 
                    port=self.port,
                    password=self.password,
                    decode_responses=True
                )
                pubsub = redis_client.pubsub()
                pubsub.subscribe(self.CHANNEL_MODEL_UPDATES)
                logger.info(f"Subscribed to channel: {self.CHANNEL_MODEL_UPDATES}")
                
                # Listen for messages
                for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = message.get("data")
                            if isinstance(data, str):
                                payload = json.loads(data)
                                logger.debug(f"Received model update: {payload}")
                                callback(payload)
                        except json.JSONDecodeError:
                            logger.warning(f"Received invalid JSON in model update: {message}")
                        except Exception as e:
                            logger.error(f"Error processing model update: {e}")
            except Exception as e:
                logger.error(f"Error in model update listener: {e}")
        
        # Start the listener thread
        thread = threading.Thread(target=listener_thread, daemon=True)
        thread.start()
        logger.info("Started model update listener thread")


def import_time_module():
    """Import time module on demand to avoid circular imports"""
    import time
    return time

