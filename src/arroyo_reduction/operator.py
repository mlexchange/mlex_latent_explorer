import asyncio
import logging

import msgpack
import zmq
from arroyopy.operator import Operator
from arroyopy.schemas import  Start, Stop
from arroyosas.schemas import RawFrameEvent, SASMessage

from .reducer import Reducer, LatentSpaceReducer
from .schemas import LatentSpaceEvent

logger = logging.getLogger("arroyo_reduction.operator")


class LatentSpaceOperator(Operator):
    def __init__(self, proxy_socket: zmq.Socket, reducer: Reducer):
        super().__init__()
        self.proxy_socket = proxy_socket
        self.reducer = reducer

    async def process(self, message: SASMessage) -> None:
        # logger.debug("message recvd")
        if isinstance(message, Start):
            logger.info("Received Start Message")
            await self.publish(message)
        elif isinstance(message, RawFrameEvent):
            result = await self.dispatch(message)
            if result is not None:  # Only publish if we got a valid result
                await self.publish(result)
        elif isinstance(message, Stop):
            logger.info("Received Stop Message")
            await self.publish(message)
        else:
            logger.warning(f"Unknown message type: {type(message)}")
        return None

    async def dispatch(self, message: RawFrameEvent) -> LatentSpaceEvent:
        try:
            # Check if models are being loaded
            if hasattr(self.reducer, 'is_loading_model') and self.reducer.is_loading_model:
                loading_type = self.reducer.loading_model_type or "unknown"
                logger.info(f"Waiting for {loading_type} model to finish loading before processing frame {message.frame_number}...")
                # Return None to indicate we should skip this frame
                return None
                
            feature_vector = await asyncio.to_thread(self.reducer.reduce, message)
            response = LatentSpaceEvent(
                tiled_url=message.tiled_url,
                feature_vector=feature_vector[0].tolist(),
                index=message.frame_number,
            )
            return response
        except Exception as e:
            logger.error(f"Error sending message to broker {e}")
            return None

    async def dispatch_workers(self, message: RawFrameEvent) -> LatentSpaceEvent:
        """Dispatch the message to the worker and return the response. This is applicable
        when the reducer is setup to run in a zqm req/rep worker pool. Currently unsupported."""

        try:
            message = message.model_dump()
            message = msgpack.packb(message, use_bin_type=True)
            await self.proxy_socket.send(message)
            # logger.debug("sent frame to broker")
            response = await self.proxy_socket.recv()
            if response == b"ERROR":
                logger.debug("Worker reported an error")
                return None
            # logger.debug("response from broker")
            return LatentSpaceEvent(**msgpack.unpackb(response))
        except Exception as e:
            logger.error(f"Error sending message to broker {e}")

    @classmethod
    def from_settings(cls, settings, reducer_settings=None):
        # Connect to the ZMQ Router/Dealer as a client
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDHWM, 10000)  # Allow up to 10,000 messages
        socket.setsockopt(zmq.RCVHWM, 10000)
        # socket.connect(settings.zmq_broker.router_address)
        # logger.info(f"Connected to broker at {settings.zmq_broker.router_address}")
        reducer = LatentSpaceReducer()
        return cls(socket, reducer)
