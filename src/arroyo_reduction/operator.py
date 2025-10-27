import asyncio
import logging
import os
import time

from arroyopy.operator import Operator
from arroyopy.schemas import Start, Stop
from arroyosas.schemas import RawFrameEvent, SASMessage

from .reducer import LatentSpaceReducer, Reducer
from .schemas import LatentSpaceEvent
from .redis_model_store import RedisModelStore  # Import the RedisModelStore class

logger = logging.getLogger("arroyo_reduction.operator")


class LatentSpaceOperator(Operator):
    def __init__(self, reducer: Reducer):
        super().__init__()
        self.reducer = reducer
        
        # NEW: Track if flush was already sent
        self._flush_sent = False
        
        # Initialize RedisModelStore instead of direct Redis client
        try:
            self.redis_host = os.getenv("REDIS_HOST", "kvrocks")
            self.redis_port = int(os.getenv("REDIS_PORT", 6666))
            self.redis_model_store = RedisModelStore(host=self.redis_host, port=self.redis_port)
            logger.info(f"Connected to Redis Model Store at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis Model Store: {e}")
            self.redis_model_store = None

    async def process(self, message: SASMessage) -> None:
        # logger.debug("message recvd")
        if isinstance(message, Start):
            logger.info("Received Start Message")
            await self.publish(message)
        elif isinstance(message, RawFrameEvent):
            # NEW: Check if models are selected before publishing RawFrameEvent
            if self.redis_model_store is not None:
                autoencoder_model = self.redis_model_store.get_autoencoder_model()
                dimred_model = self.redis_model_store.get_dimred_model()
                
                if not autoencoder_model or not dimred_model:
                    logger.info(f"In offline mode - skipping write image {message.frame_number}")
                else:
                    await self.publish(message)
            else:
                # If redis not available, publish anyway (default behavior)
                await self.publish(message)

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
                    # NEW: Send flush only once when entering offline mode
                    if not self._flush_sent:
                        flush_event = LatentSpaceEvent(
                            tiled_url="FLUSH_SIGNAL",
                            feature_vector=[],
                            index=-1,
                            autoencoder_model="",
                            dimred_model="",
                            experiment_name="",
                            timestamp=time.time()
                        )
                        await self.publish(flush_event)
                        self._flush_sent = True
                        logger.info("Sent flush signal when entering offline mode")
                    
                    logger.info(f"In offline mode - skipping dispatch frame {message.frame_number}")
                    return None
                else:
                    # NEW: Reset flush flag when back in live mode
                    self._flush_sent = False
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
            
            # NEW: Get experiment name from the reducer
            experiment_name = self.reducer.experiment_name
            
            response = LatentSpaceEvent(
                tiled_url=message.tiled_url,
                feature_vector=feature_vector[0].tolist(),
                index=message.frame_number,
                autoencoder_model=current_autoencoder,  # Add autoencoder model name
                dimred_model=current_dimred,            # Add dimension reduction model name
                experiment_name=experiment_name,        # NEW: Add experiment name
                timestamp=start_time,                   # Add start timestamp
                total_processing_time=total_processing_time,  # Add total processing time
                autoencoder_time=timing_info.get('autoencoder_time'),  # Add autoencoder processing time
                dimred_time=timing_info.get('dimred_time'),            # Add dimension reduction processing time
            )
            return response
        except Exception as e:
            logger.error(f"Error sending message to broker {e}")
            return None


    @classmethod
    def from_settings(cls, settings, reducer_settings=None):

        # socket.connect(settings.zmq_broker.router_address)
        # logger.info(f"Connected to broker at {settings.zmq_broker.router_address}")
        reducer = LatentSpaceReducer()
        return cls(reducer)