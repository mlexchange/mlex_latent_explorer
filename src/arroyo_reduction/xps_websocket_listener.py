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
    
    def __init__(self, operator: Operator, websocket_url: str):
        self.operator = operator
        self.websocket_url = websocket_url
        self.should_stop = False
        self.current_uuid = None  # Track current run UUID
        self.tiled_base_uri = RESULTS_TILED_URI  # Store base URI
        
    async def start(self):
        """Connect to XPS websocket and listen for messages"""
        logger.info(f"XPS WebSocket listen loop started on {self.websocket_url}")
        logger.info(f"Using Tiled base URI: {self.tiled_base_uri}")
        
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
        # First message is JSON with metadata (skip it)
        if isinstance(message, str):
            data = json.loads(message)
            logger.debug(f"Received XPS metadata: {data}")
            return
        
        # Second message is msgpack with images
        data = msgpack.unpackb(message)
        
        # Extract shot_mean (the averaged heatmap)
        shot_mean_bytes = data.get('shot_mean')
        width = data.get('width')
        height = data.get('height')
        shot_num = data.get('shot_num', 0)
        
        if not shot_mean_bytes or not width or not height:
            logger.warning("Received XPS message without shot_mean data")
            return
        
        # Generate new UUID when frame_number is 0 (start of new run)
        if shot_num == 1:
            self.current_uuid = str(uuid.uuid4())
            logger.info(f"Starting new XPS run with UUID: {self.current_uuid}")
        
        # Convert uint8 bytes back to numpy array
        shot_mean = np.frombuffer(shot_mean_bytes, dtype=np.uint8).reshape(width, height)
        
        logger.debug(f"Received shot_mean for shot {shot_num}: shape {shot_mean.shape}")
        
        # Define the target tiled_url where the image should be saved
        # Format: base_url/api/v1/array/full/beamlines/bl931/processed/live_data_cache/UUID/xps_averaged_heatmaps/frame_N
        tiled_url = f"{self.tiled_base_uri}/api/v1/array/full/beamlines/bl931/processed/live_data_cache/{self.current_uuid}/xps_averaged_heatmaps/frame_{shot_num}?slice=0:1,0:{height},0:{width}"
        
        # Create a RawFrameEvent compatible with the operator
        frame_event = RawFrameEvent(
            image=SerializableNumpyArrayModel(array=shot_mean),
            frame_number=shot_num,
            tiled_url=tiled_url
        )
        
        # Process through operator
        await self.operator.process(frame_event)
    
    async def stop(self):
        """Stop the listener"""
        logger.info("Stopping XPS websocket listener")
        self.should_stop = True
    
    @classmethod
    def from_settings(cls, settings: dict, operator: Operator) -> "XPSWebSocketListener":
        """Create listener from settings (consistent with ZMQFrameListener)"""
        websocket_url = settings.websocket_url
        logger.info(f"##### Listening for XPS frames on {websocket_url}")
        return cls(operator, websocket_url)