import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from tiled.client import from_uri

load_dotenv(".env")

DATA_TILED_URI = "http://tiled:8000/api/v1/metadata"
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_num_frames(tiled_uri, tiled_api_key=None):
    client = from_uri(tiled_uri, api_key=tiled_api_key)
    return client.shape


# Define how many images per label to include
label_mapping = {
    1: 10,
    2: 3,
    3: 1,
    4: 15,
    5: 1,
    6: 5,
    7: 7,
    8: 2,
    9: 4,
    0: 6,
}

label_names = {
    1: "ones",
    2: "twos",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    0: "zero",
}


async def stream():
    """
    Connect to the existing WebSocket server elsewhere,
    send messages, then close the connection.
    """

    logger.info("Preparing messages to send...")
    time.sleep(3)

    for digit in range(10):
        tiled_uri = f"{DATA_TILED_URI}/mnist/{label_names[digit]}"
        num_imgs = label_mapping[digit]
        if num_imgs > 1:
            for i in range(num_imgs):
                message = {
                    "tiled_url": f"{tiled_uri}?slice={i}",  # be compatible with LatentSpaceEvent
                    "index": i,
                    "feature_vector": [digit, i],  # Example feature vector
                }
                logger.info(f"Sending message: {message}")

                # Send the message (as JSON) to the server
                yield message

                await asyncio.sleep(0.05)
        else:
            message = {
                "tiled_url": f"{tiled_uri}",  # be compatible with LatentSpaceEvent
                "index": 0,
                "feature_vector": [digit, 0],  # Example feature vector
            }
            logger.info(f"Sending message: {message}")

            # Send the message (as JSON) to the server
            yield message

            await asyncio.sleep(0.05)

    logger.info("All messages sent; connection closed.")
