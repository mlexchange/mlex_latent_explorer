import asyncio
import logging
import os
import sys

from arroyosas.zmq import ZMQFrameListener
from dynaconf import Dynaconf
import typer
import redis

from .operator import LatentSpaceOperator
from .publisher import LSEWSResultPublisher

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
    # logger.setLevel(log_level)
    logger.setLevel(log_level.upper())
    logger.debug("DEBUG LOGGING SET")

setup_logger(logger, settings.logging_level)


@app.command()
async def start() -> None:
    try:
        app_settings = settings.lse_operator
        logger.info("Getting settings")
        logger.info(f"{settings.lse_operator}")

        logger.info("Starting ZMQ PubSub Listener")
        logger.info(f"ZMQPubSubListener settings: {app_settings}")
        
        # Initialize the WebSocket publisher first (so it's available for connections)
        ws_publisher = LSEWSResultPublisher.from_settings(app_settings.ws_publisher)
        publisher_task = asyncio.create_task(ws_publisher.start())
        
        # Wait for model selection in Redis before starting listener
        logger.info("Waiting for models to be selected in the UI...")
        
        # Create Redis client
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # Poll Redis for model selections
        while True:
            try:
                # Check if both models are selected
                autoencoder_model = redis_client.get("selected_mlflow_model")
                dimred_model = redis_client.get("selected_dim_reduction_model")
                
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
        
        listener = ZMQFrameListener.from_settings(app_settings.listener, operator)
        
        # Start the listener
        logger.info("Starting to listen for messages from arroyo_sim")
        await listener.start()
        
    except Exception as e:
        logger.critical(f"Fatal error in main application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(start())