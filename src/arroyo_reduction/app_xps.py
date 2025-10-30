import asyncio
import logging
import os
import sys

import typer
from dynaconf import Dynaconf

from .operator import LatentSpaceOperator
from .publisher import LSEWSResultPublisher
from .redis_model_store import RedisModelStore
from .vector_save import VectorSavePublisher
from .tiled_results_publisher import TiledResultsPublisher
from .xps_websocket_listener import XPSWebSocketListener
from .xps_tiled_local_image_publisher import XPSTiledLocalImagePublisher  # NEW: Add this import

settings = Dynaconf(
    envvar_prefix="",
    settings_files=["settings.yaml", ".secrets.yaml"],
    load_dotenv=True,
)
app = typer.Typer()
logger = logging.getLogger("arroyo_reduction")

# Redis connection info
REDIS_HOST = os.getenv("REDIS_HOST", "kvrocks")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6666))

def setup_logger(logger: logging.Logger, log_level: str = "INFO"):
    formatter = logging.Formatter("%(levelname)s: (%(name)s)  %(message)s ")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level.upper())
    logger.debug("DEBUG LOGGING SET")

setup_logger(logger, settings.logging_level)


@app.command()
async def start() -> None:
    try:
        app_settings = settings.lse_operator
        logger.info("Starting XPS Latent Space Encoder")
        logger.info(f"{settings.lse_operator}")
        
        # Initialize the WebSocket publisher first (so it's available for connections)
        ws_publisher = LSEWSResultPublisher.from_settings(app_settings.ws_publisher)
        asyncio.create_task(ws_publisher.start())

        # Initialize the VectorSavePublisher for saving vectors to SQLite
        vector_save_publisher = VectorSavePublisher(db_path=app_settings.vector_save.db_path)
        asyncio.create_task(vector_save_publisher.start())
        
        # Initialize the TiledResultsPublisher for saving vectors to Tiled
        tiled_publisher = TiledResultsPublisher.from_settings(app_settings.tiled_publisher)
        asyncio.create_task(tiled_publisher.start())
        
        # NEW: Initialize the XPS-specific local image publisher
        xps_local_image_publisher = XPSTiledLocalImagePublisher.from_settings(app_settings.local_image_publisher)
        asyncio.create_task(xps_local_image_publisher.start())
        
        # Initialize Redis model store instead of direct Redis client
        logger.info("Initializing Redis Model Store")
        redis_model_store = RedisModelStore(host=REDIS_HOST, port=REDIS_PORT)
        
        # Wait for model selection in Redis before starting listener
        logger.info("Waiting for models to be selected in the UI...")
        
        # Poll Redis for model selections using the model store
        while True:
            try:
                # Check if both models are selected using the model store
                autoencoder_model = redis_model_store.get_autoencoder_model()
                dimred_model = redis_model_store.get_dimred_model()
                
                if autoencoder_model and dimred_model:
                    logger.info(f"Models selected - starting processing:")
                    logger.info(f"  Autoencoder: {autoencoder_model}")
                    logger.info(f"  Dimension Reduction: {dimred_model}")
                    break
                
                # Wait before checking again
                await asyncio.sleep(2)
                logger.debug("Still waiting for model selection...")
                
            except Exception as e:
                logger.error(f"Error checking Redis: {e}")
                await asyncio.sleep(5)  # Longer delay on error
        
        # Models are selected, now create operator and start listening
        operator = LatentSpaceOperator.from_settings(app_settings, settings.lse_reducer)
        operator.add_publisher(ws_publisher)
        operator.add_publisher(vector_save_publisher)
        operator.add_publisher(tiled_publisher)
        operator.add_publisher(xps_local_image_publisher)  # NEW: Add this line
        
        # Use from_settings() pattern (consistent with ZMQFrameListener)
        listener = XPSWebSocketListener.from_settings(app_settings.xps_listener, operator)
        
        # Start the listener
        logger.info("Starting to listen for XPS messages")
        await listener.start()
        
    except Exception as e:
        logger.critical(f"Fatal error in XPS application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(start())