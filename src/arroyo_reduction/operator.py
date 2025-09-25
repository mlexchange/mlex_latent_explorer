import asyncio
import logging
import os
import time

import msgpack
import redis
import zmq
from arroyopy.operator import Operator
from arroyopy.schemas import Start, Stop
from arroyosas.schemas import RawFrameEvent, SASMessage

from .reducer import LatentSpaceReducer, Reducer
from .schemas import LatentSpaceEvent
from .redis_model_store import RedisModelStore

logger = logging.getLogger("arroyo_reduction.operator")


class LatentSpaceOperator(Operator):
    def __init__(self, proxy_socket: zmq.Socket, reducer: Reducer):
        super().__init__()
        self.proxy_socket = proxy_socket
        self.reducer = reducer
        
        # Initialize RedisModelStore instead of direct Redis client
        try:
            self.redis_host = os.getenv("REDIS_HOST", "kvrocks")
            self.redis_port = int(os.getenv("REDIS_PORT", 6666))
            self.redis_model_store = RedisModelStore(host=self.redis_host, port=self.redis_port)
            logger.info(f"Connected to Redis Model Store at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis Model Store: {e}")
            self.redis_model_store = None
            
        # Dictionary to store mappings from remote to local Tiled URLs
        self.local_image_urls = {}

    async def process(self, message: SASMessage) -> None:
        # logger.debug("message recvd")
        if isinstance(message, Start):
            logger.info("Received Start Message")
            await self.publish(message)
        elif isinstance(message, RawFrameEvent):
            # Process the RawFrameEvent message normally
            # Note: All publishers will automatically receive this message through self.publish
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
            # Use the RedisModelStore instead of direct Redis client
            if self.redis_model_store is not None:
                # Check if processing is disabled (by checking if models are set)
                autoencoder_model = self.redis_model_store.get_autoencoder_model()
                dimred_model = self.redis_model_store.get_dimred_model()
                
                if not autoencoder_model or not dimred_model:
                    logger.info(f"In offline mode - skipping frame {message.frame_number}")
                    return None
            else:
                # Model store couldn't be initialized, log a warning but continue processing
                logger.debug("Redis Model Store not available, proceeding with processing")
                
            # Existing loading check
            if hasattr(self.reducer, 'is_loading_model') and self.reducer.is_loading_model:
                loading_type = self.reducer.loading_model_type or "unknown"
                logger.info(f"Waiting for {loading_type} model to finish loading before processing frame {message.frame_number}...")
                return None
            
            # Record timing information
            start_time = time.time()
            
            # Pass message to reducer with timing information tracking
            feature_vector, timing_info = await asyncio.to_thread(self.reducer.reduce, message)
            
            # Calculate total processing time
            end_time = time.time()
            total_processing_time = end_time - start_time
            
            if feature_vector is None:
                logger.info(f"Skipping frame {message.frame_number} due to processing error or model transition")
                return None
            
            # Get the current model names from the reducer
            current_autoencoder = self.reducer.autoencoder_model_name
            current_dimred = self.reducer.dimred_model_name
            
            # Get the local Tiled URL if available
            local_tiled_url = self.get_local_url(message.tiled_url)
            
            # If we have a local URL, log it
            if local_tiled_url:
                logger.debug(f"Using local Tiled URL: {local_tiled_url} instead of {message.tiled_url}")
            
            response = LatentSpaceEvent(
                # Use local URL if available, otherwise use the original
                tiled_url=local_tiled_url or message.tiled_url,
                # Store the original URL regardless
                original_tiled_url=message.tiled_url,
                feature_vector=feature_vector[0].tolist(),
                index=message.frame_number,
                autoencoder_model=current_autoencoder,
                dimred_model=current_dimred,
                timestamp=start_time,
                total_processing_time=total_processing_time,
                autoencoder_time=timing_info.get('autoencoder_time'),
                dimred_time=timing_info.get('dimred_time'),
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

    def get_local_url(self, original_url):
        """Get the local URL for an original Tiled URL, if available."""
        # First check our cache
        if original_url in self.local_image_urls:
            return self.local_image_urls[original_url]
            
        # If not in cache, try to get it from the TiledLocalImagePublisher
        for publisher in self.publishers:
            if hasattr(publisher, 'get_local_url_for'):
                local_url = publisher.get_local_url_for(original_url)
                if local_url:
                    # Cache the result for future use
                    self.local_image_urls[original_url] = local_url
                    return local_url
                    
        # If no local URL is available, return None
        return None

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