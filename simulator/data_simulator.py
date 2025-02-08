import asyncio
import json
import logging
import os

import websockets
from dotenv import load_dotenv
from tiled.client import from_uri

load_dotenv(".env")

DATA_TILED_URI = os.getenv("DEFAULT_TILED_URI")
DATA_TILED_API_KEY = os.getenv("DATA_TILED_KEY")

RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
# Only append "/api/v1/metadata/" if it's not already in the string
if "/api/v1/metadata/" not in RESULTS_TILED_URI:
    RESULTS_TILED_URI = RESULTS_TILED_URI.rstrip("/") + "/api/v1/metadata/"

RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", None)

WEBSOCKET_PORT = os.getenv("WEBSOCKET_PORT", "8765")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "localhost")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_tiled_list(tiled_uri, tiled_api_key=None):
    client = from_uri(tiled_uri, api_key=tiled_api_key)
    return list(client.keys())


async def client_main():
    """
    Connect to the existing WebSocket server elsewhere,
    send messages, then close the connection.
    """

    logger.info("Preparing messages to send...")
    num_messages = 20

    data_list = get_tiled_list(DATA_TILED_URI, DATA_TILED_API_KEY)[0:num_messages]
    feature_vector_list = get_tiled_list(RESULTS_TILED_URI, RESULTS_TILED_API_KEY)[
        -num_messages:
    ]

    # Construct the WebSocket URI (e.g., ws://localhost:8765)
    uri = f"ws://{WEBSOCKET_URL}:{WEBSOCKET_PORT}"
    logger.info(f"Connecting to {uri}...")

    # Connect to the server
    async with websockets.connect(uri) as websocket:
        logger.info("Successfully connected to the server!")

        for data_uri, latent_vector_uri in zip(data_list, feature_vector_list):
            message = {
                "root_uri": DATA_TILED_URI,
                "data_uri": data_uri,
                "feature_vector_uri": latent_vector_uri,
            }
            logger.info(f"Sending message: {message}")

            # Send the message (as JSON) to the server
            await websocket.send(json.dumps(message))

            await asyncio.sleep(1)

    logger.info("All messages sent; connection closed.")


if __name__ == "__main__":
    asyncio.run(client_main())
