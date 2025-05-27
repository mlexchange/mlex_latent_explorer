import asyncio
import logging

from arroyosas.zmq import ZMQFrameListener
from dynaconf import Dynaconf
import typer

from .operator import LatentSpaceOperator
from .publisher import LSEWSResultPublisher


settings = Dynaconf(
    envvar_prefix="",
    settings_files=["arroyo_settings.yaml", ".arroyo_secrets.yaml"],
    load_dotenv=True,
)
app = typer.Typer()
logger = logging.getLogger("lse_arroyo")



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
    app_settings = settings.lse_operator
    logger.info("Getting settings")
    logger.info(f"{settings.lse_operator}")

    logger.info("Starting ZMQ PubSub Listener")
    logger.info(f"ZMQPubSubListener settings: {app_settings}")
    operator = LatentSpaceOperator.from_settings(app_settings, settings.lse_reducer)

    ws_publisher = LSEWSResultPublisher.from_settings(app_settings.ws_publisher)

    operator.add_publisher(ws_publisher)

    listener = ZMQFrameListener.from_settings(app_settings.listener, operator)
    await asyncio.gather(listener.start(), ws_publisher.start())


if __name__ == "__main__":
    asyncio.run(start())