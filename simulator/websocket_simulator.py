import asyncio
import logging
import os
import sys

import numpy as np
import typer

from src.arroyo_reduction.publisher import LSEWSResultPublisher
from src.arroyo_reduction.schemas import LatentSpaceEvent

from .data_simulator import stream
from .data_simulator_mnist import mnist_stream
from .tiled_ingestor_mnist import ingest_mnist_to_tiled

app = typer.Typer()
logger = logging.getLogger("arroyo_reduction")
SIMULATION_TYPE = os.getenv("SIMULATION_TYPE", "default").lower()  # default or mnist


def setup_logger(logger: logging.Logger, log_level: str = "INFO"):
    formatter = logging.Formatter("%(levelname)s: (%(name)s)  %(message)s ")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.setLevel(log_level)
    logger.setLevel(log_level.upper())
    logger.debug("DEBUG LOGGING SET")


setup_logger(logger)


def get_feature_vectors(num_messages):
    return 5 * np.random.rand(num_messages, 2)


class DummyWSPublisher(LSEWSResultPublisher):

    def __init__(self, host="0.0.0.0", port=8765, path="/ws"):
        super().__init__(host, port, path)
        logger.info("DummyWSPublisher initialized")

    async def start(self) -> None:
        logger.info("DummyWSPublisher started, but does nothing.")
        await super().start()


@app.command()
def start() -> None:
    async def main():
        ws_publisher = DummyWSPublisher()
        asyncio.create_task(ws_publisher.start())
        while True:
            if SIMULATION_TYPE == "mnist":
                logger.info("Starting MNIST simulation...")
                gen = mnist_stream()
            else:
                gen = stream()
                logger.info("Starting default simulation...")
            while True:
                # Simulate receiving data
                message = await anext(gen, None)
                if message is None:
                    break
                await ws_publisher.publish(LatentSpaceEvent(**message))

    asyncio.run(main())


if __name__ == "__main__":
    if SIMULATION_TYPE == "mnist":
        logger.info("Running MNIST simulation...")
        ingestion = ingest_mnist_to_tiled(
            tiled_uri=os.getenv("RESULTS_TILED_URI", "http://tiled:8000/api/v1"),
            api_key=os.getenv("RESULTS_TILED_API_KEY", None),
        )
        if not ingestion:
            sys.exit(1)

    app()
