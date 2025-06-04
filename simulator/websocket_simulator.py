import asyncio
import logging

import numpy as np
import typer

from .data_simulator import stream
from src.arroyo_reduction.publisher import LSEWSResultPublisher
from src.arroyo_reduction.schemas import LatentSpaceEvent

app = typer.Typer()
logger = logging.getLogger("arroyo_reduction")



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
            gen = stream()
            while True:
                # Simulate receiving data
                message = await anext(gen, None)
                if message is None:
                    break
                await ws_publisher.publish(LatentSpaceEvent(**message))

    asyncio.run(main())


if __name__ == "__main__":
    app()