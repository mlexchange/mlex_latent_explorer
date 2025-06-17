import asyncio
import logging
import os

import msgpack
import redis
import zmq
from arroyopy.operator import Operator
from arroyopy.schemas import Start, Stop
from arroyosas.schemas import RawFrameEvent, SASMessage

from .reducer import LatentSpaceReducer, Reducer
from .schemas import LatentSpaceEvent

logger = logging.getLogger("arroyo_reduction.operator")


class LatentSpaceOperator(Operator):
    def __init__(self, proxy_socket: zmq.Socket, reducer: Reducer):
        super().__init__()
        self.proxy_socket = proxy_socket
        self.reducer = reducer
        
        # Initialize Redis client once during initialization
        try:
            self.redis_host = os.getenv("REDIS_HOST", "kvrocks")
            self.redis_port = int(os.getenv("REDIS_PORT", 6666))
            self.redis_client = redis.Redis(
                host=self.redis_host, 
                port=self.redis_port, 
                decode_responses=True
            )
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None

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
            # Use the instance's Redis client instead of creating a new one each time
            if self.redis_client is not None:
                # Check if processing is disabled (by checking if models are set)
                autoencoder_model = self.redis_client.get("selected_mlflow_model")
                dimred_model = self.redis_client.get("selected_dim_reduction_model")
                
                if not autoencoder_model or not dimred_model:
                    logger.info(f"In offline mode - skipping frame {message.frame_number}")
                    return None
            else:
                # Redis client couldn't be initialized, log a warning but continue processing
                logger.debug("Redis client not available, proceeding with processing")
                
            # Existing loading check
            if hasattr(self.reducer, 'is_loading_model') and self.reducer.is_loading_model:
                loading_type = self.reducer.loading_model_type or "unknown"
                logger.info(f"Waiting for {loading_type} model to finish loading before processing frame {message.frame_number}...")
                return None
                
            feature_vector = await asyncio.to_thread(self.reducer.reduce, message)
            
            # Get the current model names from the reducer
            current_autoencoder = self.reducer.autoencoder_model_name
            current_dimred = self.reducer.dimred_model_name
            
            response = LatentSpaceEvent(
                tiled_url=message.tiled_url,
                feature_vector=feature_vector[0].tolist(),
                index=message.frame_number,
                autoencoder_model=current_autoencoder,  # Add autoencoder model name
                dimred_model=current_dimred,            # Add dimension reduction model name
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