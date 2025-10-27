import asyncio
import json
import logging
import msgpack
import numpy as np
import os
import websockets
import uuid
from arroyopy.listener import Listener
from arroyopy.operator import Operator
from arroyosas.schemas import RawFrameEvent, SerializableNumpyArrayModel

logger = logging.getLogger("arroyo_reduction.xps_websocket_listener")

# Read base URL from environment variable
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")


class XPSWebSocketListener(Listener):
    """Listen to XPS websocket and process shot_mean data"""
    
    def __init__(self, operator: Operator, websocket_url: str, tiled_prefix: str = None):
        self.operator = operator
        self.websocket_url = websocket_url
        self.should_stop = False
        self.current_uuid = None
        self.tiled_base_uri = RESULTS_TILED_URI
        self.frame_counter = 0
        self.tiled_prefix = tiled_prefix or "beamlines/bl931/processed"
        
    async def start(self):
        """Connect to XPS websocket and listen for messages"""
        logger.info(f"XPS WebSocket listener starting on {self.websocket_url}")
        logger.info(f"Using Tiled base URI: {self.tiled_base_uri}")
        logger.info(f"Using Tiled prefix: {self.tiled_prefix}")
        
        while not self.should_stop:
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
        # First message is JSON metadata (skip)
        if isinstance(message, str):
            data = json.loads(message)
            logger.debug(f"Received XPS metadata: {data}")
            return
        
        # Second message is msgpack with images
        data = msgpack.unpackb(message)
        
        shot_mean_bytes = data.get('shot_mean')
        width = data.get('width')
        height = data.get('height')
        shot_num = data.get('shot_num', 0)
        
        if not shot_mean_bytes or not width or not height:
            logger.warning("Received XPS message without shot_mean data")
            return
        
        # Generate new UUID when shot_num is 1 (start of new run)
        if shot_num == 1:
            self.current_uuid = str(uuid.uuid4())
            self.frame_counter = 0
            logger.info(f"Starting new XPS run with UUID: {self.current_uuid}")
        
        # Convert bytes to numpy array
        shot_mean = np.frombuffer(shot_mean_bytes, dtype=np.uint8).reshape(width, height)
        
        logger.debug(f"Received shot_mean for shot {shot_num}: shape {shot_mean.shape}")
        
        # Construct tiled_url for the 3D array
        prefix_path = f"{self.tiled_prefix}/" if self.tiled_prefix else ""
        # URL format for accessing slice of 3D array: array[frame:frame+1, 0:height, 0:width]
        tiled_url = (
            f"{self.tiled_base_uri}/api/v1/array/full/{prefix_path}"
            f"live_data_cache/{self.current_uuid}/xps_averaged_heatmaps"
            f"?slice={self.frame_counter}:{self.frame_counter+1},0:{height},0:{width}"
        )
        
        # Create RawFrameEvent
        frame_event = RawFrameEvent(
            image=SerializableNumpyArrayModel(array=shot_mean),
            frame_number=self.frame_counter,
            tiled_url=tiled_url
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
    def from_settings(cls, settings: dict, operator: Operator) -> "XPSWebSocketListener":
        """Create listener from settings"""
        websocket_url = settings.websocket_url
        tiled_prefix = settings.get("tiled_prefix", "beamlines/bl931/processed")
        logger.info(f"Listening for XPS frames on {websocket_url}")
        return cls(operator, websocket_url, tiled_prefix)