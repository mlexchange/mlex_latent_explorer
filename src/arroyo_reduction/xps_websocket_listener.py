import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime

import msgpack
import numpy as np
import pytz
import websockets
from arroyopy.listener import Listener
from arroyopy.operator import Operator
from arroyosas.schemas import RawFrameEvent, SerializableNumpyArrayModel

from .redis_model_store import RedisModelStore

logger = logging.getLogger("arroyo_reduction.xps_websocket_listener")

# Read base URL from environment variable
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")

# Timezone for daily run ID
CALIFORNIA_TZ = pytz.timezone("US/Pacific")

# Finds a full UUID36 anywhere in a string
UUID_PATTERN = re.compile(
    r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
)


class XPSWebSocketListener(Listener):
    """Listen to XPS websocket and process shot_mean data"""

    def __init__(
        self, operator: Operator, websocket_url: str, tiled_prefix: str = None
    ):
        self.operator = operator
        self.websocket_url = websocket_url
        self.should_stop = False
        self.current_uuid = None
        self.tiled_base_uri = RESULTS_TILED_URI
        self.frame_counter = 0
        self.tiled_prefix = tiled_prefix or "beamlines/bl931/processed"

        # Initialize Redis model store to get experiment name
        try:
            redis_host = os.getenv("REDIS_HOST", "kvrocks")
            redis_port = int(os.getenv("REDIS_PORT", 6666))
            self.redis_model_store = RedisModelStore(host=redis_host, port=redis_port)
            logger.info("Connected to Redis Model Store for experiment name")
        except Exception as e:
            logger.warning(f"Could not connect to Redis Model Store: {e}")
            self.redis_model_store = None

    def _get_experiment_name(self):
        """
        Synchronous helper to get experiment name from Redis.
        This will be called via asyncio.to_thread() to avoid blocking the event loop.

        Returns:
            str: Experiment name or "default_experiment" if not available
        """
        if self.redis_model_store is None:
            return "default_experiment"

        try:
            redis_experiment_name = self.redis_model_store.get_experiment_name()
            if redis_experiment_name:
                return redis_experiment_name
            else:
                logger.warning("No experiment name in Redis, using default")
                return "default_experiment"
        except Exception as e:
            logger.error(f"Error getting experiment name from Redis: {e}")
            return "default_experiment"

    async def start(self):
        """Connect to XPS websocket and listen for messages"""
        logger.info(f"XPS WebSocket listener starting on {self.websocket_url}")
        logger.info(f"Using Tiled base URI: {self.tiled_base_uri}")
        logger.info(f"Using Tiled prefix: {self.tiled_prefix}")

        while not self.should_stop:
            # Wait for models to be selected before connecting
            if self.redis_model_store:
                autoencoder_model = self.redis_model_store.get_autoencoder_model()
                dimred_model = self.redis_model_store.get_dimred_model()
                if not autoencoder_model or not dimred_model:
                    logger.debug(
                        "Models not selected - waiting before connecting to XPS websocket..."
                    )
                    await asyncio.sleep(2)
                    continue

            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    logger.info("Connected to XPS websocket")

                    async for message in websocket:
                        if self.should_stop:
                            break
                        try:
                            await self._handle_message(message)
                        except Exception as e:
                            logger.exception(f"Error processing message: {e}")

            except websockets.ConnectionClosed:
                logger.warning("XPS websocket connection closed, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in XPS websocket connection: {e}")
                await asyncio.sleep(5)

    async def _handle_message(self, message):
        """Parse XPS message and extract shot_mean for processing"""
        # First message is JSON metadata
        if isinstance(message, str):
            data = json.loads(message)
            logger.debug(f"Received XPS metadata: {data}")

            msg_type = data.get("msg_type")

            if msg_type == "start":
                scan_name = data.get("scan_name", "")
                # Extract UUID36 if present anywhere in scan_name
                # (e.g. LabVIEW sends "temp name <uuid>").
                # If not found (e.g. Timepix "acquisition_..." with no full UUID),
                # generate a fresh one so all sources use the same id format.
                match = UUID_PATTERN.search(scan_name)
                if match:
                    self.current_uuid = match.group()
                    logger.info(
                        f"Starting new XPS run, extracted UUID from "
                        f"'{scan_name}': {self.current_uuid}"
                    )
                else:
                    self.current_uuid = str(uuid.uuid4())
                    logger.info(
                        f"scan_name '{scan_name}' contains no UUID36 — "
                        f"generated new UUID: {self.current_uuid}"
                    )
                self.frame_counter = 0

            elif msg_type == "stop":
                logger.info(
                    f"XPS run stopped: scan_name={data.get('scan_name')} "
                    f"frames={self.frame_counter}"
                )
                self.current_uuid = None
                self.frame_counter = 0

            return

        # Second message is msgpack with images
        data = msgpack.unpackb(message)

        shot_mean_bytes = data.get("shot_mean")
        width = data.get("width")
        height = data.get("height")
        shot_num = data.get("shot_num", 0)

        if not shot_mean_bytes or not width or not height:
            logger.warning("Received XPS message without shot_mean data")
            return

        # Convert bytes to numpy array
        shot_mean = np.frombuffer(shot_mean_bytes, dtype=np.uint8).reshape(
            width, height
        )

        logger.debug(f"Received shot_mean for shot {shot_num}: shape {shot_mean.shape}")

        # Get experiment name from Redis (non-blocking)
        experiment_name = await asyncio.to_thread(self._get_experiment_name)
        logger.debug(f"Using experiment name: {experiment_name}")

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
            f"?slice={self.frame_counter}:{self.frame_counter+1},0:{height},0:{width}"
        )

        logger.debug(f"Constructed tiled_url: {tiled_url}")

        # Create RawFrameEvent
        frame_event = RawFrameEvent(
            image=SerializableNumpyArrayModel(array=shot_mean),
            frame_number=self.frame_counter,
            tiled_url=tiled_url,
        )

        # Increment frame counter
        self.frame_counter += 1

        # Process through operator
        await self.operator.process(frame_event)

    async def stop(self):
        """Stop the listener"""
        logger.info("Stopping XPS websocket listener")
        self.should_stop = True

    @classmethod
    def from_settings(
        cls, settings: dict, operator: Operator
    ) -> "XPSWebSocketListener":
        """Create listener from settings"""
        websocket_url = settings.websocket_url
        tiled_prefix = settings.get("tiled_prefix", "beamlines/bl931/processed")
        logger.info(f"Listening for XPS frames on {websocket_url}")
        return cls(operator, websocket_url, tiled_prefix)


def xps_ws_listener_factory(
    operator, websocket_url: str, tiled_prefix: str = "beamlines/bl931/processed"
):
    return XPSWebSocketListener(operator, websocket_url, tiled_prefix)
