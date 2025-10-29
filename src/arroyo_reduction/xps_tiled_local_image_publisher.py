import asyncio
import logging
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pytz
from arroyopy.publisher import Publisher
from arroyosas.schemas import SASStop, RawFrameEvent
from tiled.client import from_uri
from tiled.client.array import ArrayClient

logger = logging.getLogger("arroyo_reduction.xps_tiled_local_image_publisher")

# API key for Tiled authentication
LOCAL_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")
# Timezone for date components
CALIFORNIA_TZ = pytz.timezone('US/Pacific')


class XPSTiledLocalImagePublisher(Publisher):
    """Publisher that saves XPS raw images to a local Tiled server using patch for incremental updates."""

    def __init__(self, tiled_api_key=None, tiled_prefix=None):
        super().__init__()
        self.tiled_api_key = tiled_api_key or LOCAL_TILED_API_KEY
        self.tiled_prefix = tiled_prefix
        self.client = None
        self.tiled_base_uri = None

        # Dictionary to track array clients for each UUID
        self.array_clients = {}  # {uuid: ArrayClient}

        logger.info(f"Initialized XPSTiledLocalImagePublisher with patch-based updates")
        logger.info(f"Using tiled prefix: {self.tiled_prefix}")

    async def start(self):
        """Initialize - actual connection happens on first publish."""
        logger.info("XPSTiledLocalImagePublisher ready")

    def _get_client(self, url):
        """Get or create Tiled client from URL."""
        if self.client is None:
            # Extract base URI from the URL
            parsed_url = urlparse(url)
            self.tiled_base_uri = f"{parsed_url.scheme}://{parsed_url.netloc}"
            logger.info(f"Connecting to Tiled server at {self.tiled_base_uri}")
            self.client = from_uri(self.tiled_base_uri, api_key=self.tiled_api_key)
        return self.client

    def _extract_uuid_from_url(self, url):
        """Extract UUID from tiled_url."""
        uuid_pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
        match = re.search(uuid_pattern, url)
        if match:
            return match.group(1)
        return None

    def _parse_path_from_url(self, url):
        """Extract the container path from tiled_url.
        
        Example URL (old): http://tiled:8000/api/v1/array/full/prefix/lse_live_results/user/daily_run/exp/uuid/xps_averaged_heatmaps?slice=...
        Example URL (new): http://tiled:8000/api/v1/array/full/prefix/lse_live_results/user/2025/01/15/exp/uuid/xps_averaged_heatmaps?slice=...
        Returns: ['prefix', 'lse_live_results', 'user', '2025', '01', '15', 'exp', 'uuid']
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Remove '/api/v1/array/full/' prefix and '/xps_averaged_heatmaps' suffix
        path = path.replace('/api/v1/array/full/', '')
        
        # Remove 'xps_averaged_heatmaps' from the end if present
        if path.endswith('/xps_averaged_heatmaps'):
            path = path[:-len('/xps_averaged_heatmaps')]
        
        # Split into segments and filter out empty strings
        segments = [s for s in path.split('/') if s]
        
        return segments

    def _get_or_create_array_client(self, tiled_url, first_frame_image):
        """Get or create an array client for the UUID in the URL."""
        uuid = self._extract_uuid_from_url(tiled_url)
        if not uuid:
            logger.error(f"Could not extract UUID from URL: {tiled_url}")
            return None

        # Check if we already have an array client for this UUID
        if uuid in self.array_clients:
            return self.array_clients[uuid]

        # Create new array client
        try:
            client = self._get_client(tiled_url)
            
            # Parse the path from tiled_url to get all container segments
            path_segments = self._parse_path_from_url(tiled_url)
            
            logger.info(f"Parsed path segments from URL: {path_segments}")
            
            # Navigate/create the container hierarchy
            container = client
            for segment in path_segments:
                if segment:
                    if segment in container:
                        container = container[segment]
                    else:
                        logger.info(f"Creating container: {segment}")
                        container = container.create_container(segment)
            
            # Create the 3D array with first frame: (1, height, width)
            initial_array = first_frame_image[None, :, :]
            
            logger.info(f"Creating 3D array with shape: {initial_array.shape}, dtype: {initial_array.dtype}")
            
            array_client = container.write_array(
                initial_array,
                key="xps_averaged_heatmaps"
            )
            
            # Cache the array client
            self.array_clients[uuid] = array_client
            logger.info(f"Created new 3D array client for UUID: {uuid} at path: {'/'.join(path_segments)}/xps_averaged_heatmaps")
            
            return array_client
            
        except Exception as e:
            logger.error(f"Error creating array client: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _patch_tiled_array(self, array_client: ArrayClient, array: np.ndarray):
        """Append new 2D frame to existing 3D Tiled array using patch."""
        try:
            shape = array_client.shape  # Current shape: (n_frames, height, width)
            num_frames = shape[0]
            
            logger.info(f"[PATCH] Current shape: {shape}, num_frames: {num_frames}")
            
            # Ensure array is 2D (height, width)
            if array.ndim != 2:
                logger.error(f"Expected 2D array, got shape: {array.shape}")
                return
            
            # Add dimension to make it (1, height, width) for patching
            frame_to_append = array[None, :, :]
            logger.info(f"[PATCH] Frame to append shape: {frame_to_append.shape}, dtype: {frame_to_append.dtype}")
            
            # Only specify offset for the growing dimension (axis 0)
            offset = (num_frames,)
            logger.info(f"[PATCH] Patching at offset: {offset}")
            
            # Patch the array - extend=True allows growing along axis 0
            array_client.patch(frame_to_append, offset=offset, extend=True)
            
            # Verify the patch worked
            new_shape = array_client.shape
            logger.info(f"[PATCH SUCCESS] New shape: {new_shape} (expected: ({num_frames + 1}, {shape[1]}, {shape[2]}))")
            
        except Exception as e:
            logger.error(f"[PATCH ERROR] Error patching Tiled array: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def publish(self, message):
        """Publish a message to Tiled server."""
        if isinstance(message, SASStop):
            logger.info("Received Stop message")
            self.array_clients.clear()
            return

        if not isinstance(message, RawFrameEvent):
            return

        try:
            # Extract image data from message
            image_array = message.image.array  # Shape: (height, width)
            tiled_url = message.tiled_url
            frame_number = message.frame_number if hasattr(message, 'frame_number') else 0

            logger.info(f"[PUBLISH] Received frame {frame_number}, shape={image_array.shape}, dtype={image_array.dtype}")

            # Get or create array client for this UUID
            array_client = await asyncio.to_thread(
                self._get_or_create_array_client,
                tiled_url,
                image_array
            )

            if not array_client:
                logger.warning(f"Failed to get array client for frame {frame_number}")
                return None

            # For the first frame (frame 0), we already wrote initial data, so skip patch
            if frame_number == 0:
                logger.info(f"[FRAME 0] Initialized 3D array for UUID")
                return tiled_url

            # Patch (append) the new frame
            logger.info(f"[FRAME {frame_number}] About to patch...")
            await asyncio.to_thread(
                self._patch_tiled_array,
                array_client,
                image_array
            )

            logger.info(f"[FRAME {frame_number}] Successfully patched to 3D array")
            return tiled_url

        except Exception as e:
            logger.error(f"Error publishing XPS data to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def from_settings(cls, settings):
        """Create an XPSTiledLocalImagePublisher from settings."""
        return cls(tiled_prefix=settings.get("tiled_prefix"))