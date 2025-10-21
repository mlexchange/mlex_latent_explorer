import asyncio
import json
import logging
import msgpack
import numpy as np
import websockets
from arroyopy.listener import Listener
from arroyopy.operator import Operator
from arroyosas.schemas import RawFrameEvent, NumpyArrayModel

logger = logging.getLogger("arroyo_reduction.xps_websocket_listener")


class XPSWebSocketListener(Listener):  # ← Inherit from Listener
    """Listen to XPS websocket and process shot_mean data"""
    
    def __init__(self, operator: Operator, websocket_url: str):  # ← Operator first (like ZMQ)
        self.operator = operator
        self.websocket_url = websocket_url
        self.should_stop = False
        
    async def start(self):
        """Connect to XPS websocket and listen for messages"""
        logger.info(f"XPS WebSocket listen loop started on {self.websocket_url}")
        
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
        
        # Convert uint8 bytes back to numpy array
        shot_mean = np.frombuffer(shot_mean_bytes, dtype=np.uint8)
        
        logger.debug(f"Received shot_mean for shot {shot_num}: shape {shot_mean.shape}")
        
        # Create a RawFrameEvent compatible with the operator
        frame_event = RawFrameEvent(
            image=NumpyArrayModel(array=shot_mean),
            frame_number=shot_num,
            tiled_url=f"xps_shot_{shot_num}"
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