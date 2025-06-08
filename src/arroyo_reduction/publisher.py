import asyncio
import json
import logging
from typing import Union

# import msgpack
# import numpy as np
import websockets
from arroyopy.publisher import Publisher
from arroyosas.schemas import SASStart, SASStop
from .schemas import LatentSpaceEvent


logger = logging.getLogger("arroyo_reduction.publisher")


class LSEWSResultPublisher(Publisher):
    """
    A publisher class for sending dimensionality reduction information

    """

    websocket_server = None
    connected_clients = set()
    current_start_message = None

    def __init__(self, host: str = "localhost", port: int = 8765, path="/lse"):

        super().__init__()
        self.host = host
        self.port = port
        self.path = path
        logger.info(f"Initialized LSEWSResultPublisher on {self.host}:{self.port}{self.path}")

    async def start(
        self,
    ):
        # Use partial to bind `self` while matching the expected handler signature
        server = await websockets.serve(
            self.websocket_handler,
            self.host,
            self.port,
        )
        logger.info(f"Websocket server started at ws://{self.host}:{self.port}")
        await server.wait_closed()

    async def publish(self, message: LatentSpaceEvent) -> None:
        if self.connected_clients:  # Only send if there are clients connected
            asyncio.gather(
                *(self.publish_ws(client, message) for client in self.connected_clients)
            )

    async def publish_ws(
        self,
        client,
        message: Union[LatentSpaceEvent | SASStart | SASStop],
    ) -> None:
        if isinstance(message, SASStop):
            # logger.info(f"WS Sending Stop {message}")
            # self.current_start_message = None
            # await client.send(json.dumps(message.model_dump()))
            return

        if isinstance(message, SASStart):
            # self.current_start_message = message
            # logger.info(f"WS Sending Start {message}")
            # await client.send(json.dumps(message.model_dump()))
            return

        if isinstance(message, LatentSpaceEvent):
            # send image data separately to client memory issues
           
            logger.debug(f"WS Sending LatentSpaceEvent {message.feature_vector}")
            await client.send(message.model_dump_json())

    async def websocket_handler(self, websocket):
        logger.info(f"New connection from {websocket.remote_address}")
        
        self.connected_clients.add(websocket)
        try:
            # Keep the connection open and do nothing until the client disconnects
            await websocket.wait_closed()
        finally:
            # Remove the client when it disconnects
            self.connected_clients.remove(websocket)
            logger.info("Client disconnected")

    @classmethod
    def from_settings(cls, settings: dict) -> "LSEWSResultPublisher":
        return cls(settings.host, settings.port)
