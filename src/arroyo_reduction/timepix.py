import asyncio
from datetime import datetime
import logging
import os
import uuid

import numpy as np
import msgpack
import pytz
import zmq.asyncio

from arroyopy.operator import Operator
from arroyopy.zmq import ZMQListener
from arroyosas.schemas import RawFrameEvent, SerializableNumpyArrayModel
# from arroyoxps.schemas import NumpyArrayModel, XPSImageInfo, XPSRawEvent, XPSStart, XPSStop
from .redis_model_store import RedisModelStore


logger = logging.getLogger(__name__)
CALIFORNIA_TZ = pytz.timezone("US/Pacific")


def setup_zmq(settings: dict) -> zmq.asyncio.Socket:
    ctx = zmq.asyncio.Context()
    lv_zmq_socket = ctx.socket(zmq.SUB)
    lv_zmq_socket.setsockopt(zmq.RCVHWM, 100000)
    logger.info(
        f"binding to: {settings.zmq_pub_address}:{settings.zmq_pub_port}"
    )
    lv_zmq_socket.connect(
        f"{settings.zmq_pub_address}:{settings.zmq_pub_port}"
    )
    lv_zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
    return lv_zmq_socket


class XPSTimepixZMQListener(ZMQListener):
    stop_signal = False

    def __init__(
        self,
        zmq_socket: zmq.asyncio.Socket,
        operator: Operator,
        redis_model_store=None,
    ):
        super().__init__(zmq_socket=zmq_socket, operator=operator)
        self.redis_model_store = redis_model_store
        self.current_uuid = str(uuid.uuid4())
        self.frame_counter = 0
        self.tiled_base_uri = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")
        self.tiled_prefix = "beamlines/bl931/processed"
        
        # Initialize experiment name from Redis
        self.experiment_name = "default_experiment"
        if self.redis_model_store:
            try:
                redis_name = self.redis_model_store.get_experiment_name()
                if redis_name:
                    self.experiment_name = redis_name
                    logger.info(f"Initialized with experiment name: {self.experiment_name}")
            except Exception as e:
                logger.error(f"Error initializing experiment name from Redis: {e}")
            
            # Subscribe to updates (both model and experiment name changes)
            self.redis_model_store.subscribe_to_updates(self._handle_redis_update)
            logger.info("Subscribed to Redis updates")

    def _handle_redis_update(self, payload: dict):
        """
        Callback for Redis pub/sub updates.
        Handles both model updates and experiment name changes.
        
        Args:
            payload: Dict with either:
                - Model update: {"model_type": str, "model_name": str, "timestamp": float}
                - Experiment update: {"update_type": "experiment_name", "experiment_name": str, "timestamp": float}
        """
        try:
            # Check if this is an experiment name update
            if payload.get("update_type") == "experiment_name":
                new_experiment_name = payload.get("experiment_name")
                if new_experiment_name and new_experiment_name != self.experiment_name:
                    logger.info(f"Experiment name updated: {self.experiment_name} -> {new_experiment_name}")
                    self.experiment_name = new_experiment_name
                    # Generate new UUID for new experiment
                    self.current_uuid = str(uuid.uuid4())
                    self.frame_counter = 0
                    logger.info(f"Generated new UUID for experiment: {self.current_uuid}")
            # Could also handle model updates here if needed in the future
            elif payload.get("model_type"):
                logger.debug(f"Model update received: {payload.get('model_type')} -> {payload.get('model_name')}")
        except Exception as e:
            logger.error(f"Error handling Redis update: {e}")

    async def start(self):
        logger.info("Listener started")
        while True:
            try:
                self.frame_counter += 1
                if self.stop_signal:
                    logger.info("Stopping listener.")
                    break
                metadata_msg_packed = await self.zmq_socket.recv()
                raw_message = await self.zmq_socket.recv()
                # print(raw_message[0:300])
                try:
                    metadata = msgpack.unpackb(metadata_msg_packed)
                except Exception as e:
                    logger.error(f"Error unpacking message: {e}")
                    continue


                # Must be an event with an image
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(f"event: {metadata.keys()}")
           
                await self.operator.process(
                    await self._build_event(raw_message, metadata)
                )
                logger.debug("event processed")
            except Exception as e:
                logger.error(e)
   
    @classmethod
    def from_settings(
        cls, settings: dict, operator: Operator
    ) -> "XPSTimepixZMQListener":
        zmq_socket = setup_zmq(settings)
        redis_host = os.getenv("REDIS_HOST", "kvrocks")
        redis_port = int(os.getenv("REDIS_PORT", 6666))
        redis_model_store = RedisModelStore(host=redis_host, port=redis_port)
        return cls(zmq_socket=zmq_socket, operator=operator, redis_model_store=redis_model_store)            

    async def _build_event(
        self,
        image: bytes,
        metadata: dict,

    ) -> RawFrameEvent:
        
        shape = tuple(metadata["shape"])
        dtype = metadata["dtype"]
        flush_number = metadata.get("flush_number")

        # tiled_url = await self._build_tiled_url(shape,flush_number)
        tiled_url = await self._build_tiled_url(shape, self.frame_counter)
        array_received = np.frombuffer(image, dtype=dtype).reshape(shape)
        frame_event = RawFrameEvent(
            image=SerializableNumpyArrayModel(array=array_received),
            frame_number=self.frame_counter,
            tiled_url=tiled_url,
        )
        return frame_event

    def _get_experiment_name(self):
        """
        Get the current experiment name from cache.
        The value is kept up-to-date via Redis pub/sub subscription.

        Returns:
            str: Current experiment name
        """
        return self.experiment_name


    async def _build_tiled_url(self,shape, flush_number) -> str:
        # Get experiment name from cache (updated via Redis pub/sub)
        experiment_name = self._get_experiment_name()
        logger.debug(f"Using experiment name: {experiment_name}")

        # REMOVED: Get USER from environment

        # Get current date components for Year/Month/Day hierarchy
        now = datetime.now(CALIFORNIA_TZ)
        year_str = str(now.year)
        month_str = f"{now.month:02d}"
        day_str = f"{now.day:02d}"

        # Construct tiled_url pointing to the new structure
        # OLD: {prefix}/lse_live_results/{USER}/{YYYY}/{MM}/{DD}/{experiment_name}/{UUID}/xps_averaged_heatmaps
        # NEW: {prefix}/lse_live_results/{YYYY}/{MM}/{DD}/{experiment_name}/{UUID}/xps_averaged_heatmaps
        prefix_path = f"{self.tiled_prefix}/" if self.tiled_prefix else ""

        tiled_url = (
            f"{self.tiled_base_uri}/api/v1/array/full/{prefix_path}"
            f"lse_live_results/{year_str}/{month_str}/{day_str}/{experiment_name}/{self.current_uuid}/xps_averaged_heatmaps"
            f"?slice={flush_number}:{flush_number+1},0:{shape[0]},0:{shape[1]}"
        )
        return tiled_url

if __name__ == "__main__":
    from arroyoxps.log_utils import setup_logger  # noqa: F401

    class DummyOperator:
        async def process(self, event: XPSRawEvent):
            logger.info(
                f"Dummy operator received event with image shape: {event.image.array.shape}"
            )


    setup_logger(logger)
    zmq_socket = setup_zmq()
    listener = XPSTimepixZMQListener(zmq_socket=zmq_socket, operator=DummyOperator())
    asyncio.run(listener.start())