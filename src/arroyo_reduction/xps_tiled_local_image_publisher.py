import asyncio
import logging
import os
from urllib.parse import urlparse

from arroyopy.publisher import Publisher
from arroyosas.schemas import SASStop, RawFrameEvent
from tiled.client import from_uri

logger = logging.getLogger("arroyo_reduction.xps_tiled_local_image_publisher")

# API key for Tiled authentication
LOCAL_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")


class XPSTiledLocalImagePublisher(Publisher):
    """Publisher that saves XPS raw images to a local Tiled server directly to the target path."""

    def __init__(self, tiled_api_key=None):
        super().__init__()
        self.tiled_api_key = tiled_api_key or LOCAL_TILED_API_KEY
        self.client = None
        self.tiled_base_uri = None  # Will be extracted from the first URL

        # Dictionary to track saved images to avoid duplicates
        self.saved_images = set()

        logger.info(f"Initialized XPSTiledLocalImagePublisher")

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

    def _extract_path_from_url(self, url):
        """
        Extract the storage path from the tiled URL.
        
        Input: http://{base_url}/api/v1/array/full/beamlines/bl931/processed/live_data_cache/UUID/xps_averaged_heatmaps/frame_0?slice=...
        Output: ['beamlines', 'bl931', 'processed', 'live_data_cache', 'UUID', 'xps_averaged_heatmaps', 'frame_0']
        """
        try:
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            
            # Find the position of "array/full" to extract the path after it
            array_full_idx = -1
            for i, part in enumerate(path_parts):
                if part == "array" and i < len(path_parts) - 1 and path_parts[i+1] == "full":
                    array_full_idx = i + 1
                    break
            
            if array_full_idx >= 0:
                # Get everything after "array/full"
                storage_path = path_parts[array_full_idx+1:]
                # Remove empty strings
                storage_path = [p for p in storage_path if p]
                return storage_path
            
            return []
        except Exception as e:
            logger.error(f"Error extracting path from URL {url}: {e}")
            return []

    def _save_image_sync(self, image_array, tiled_url):
        """Save XPS image to Tiled server synchronously using the path from tiled_url."""
        try:
            # Get or create client from URL
            client = self._get_client(tiled_url)
            
            if client is None:
                logger.error("Could not initialize client")
                return False

            # Extract the full path from the URL
            path_parts = self._extract_path_from_url(tiled_url)
            
            if not path_parts:
                logger.error(f"Could not extract path from URL: {tiled_url}")
                return False
            
            # Create a unique key for this image to avoid duplicates
            image_key = '/'.join(path_parts)
            
            # Skip if we've already saved this image
            if image_key in self.saved_images:
                logger.debug(f"Image already saved: {image_key}")
                return True

            # Navigate/create the path structure
            current = client
            
            # Navigate through all path components except the last one (which is the array name)
            for component in path_parts[:-1]:
                if component not in current:
                    logger.info(f"Creating container: {component}")
                    current = current.create_container(component)
                else:
                    current = current[component]
            
            # Save the image with the last component as the key
            array_name = path_parts[-1]
            logger.info(f"Saving XPS image to {'/'.join(path_parts)}")
            
            if array_name not in current:
                current.write_array(image_array, key=array_name)
            else:
                logger.debug(f"Array {array_name} already exists, skipping write")

            # Mark as saved
            self.saved_images.add(image_key)
            return True

        except Exception as e:
            logger.error(f"Error saving XPS image to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def publish(self, message):
        """Publish a message to Tiled server."""
        if isinstance(message, SASStop):
            logger.info("Received Stop message")
            return

        if not isinstance(message, RawFrameEvent):
            return

        try:
            # Extract image data from message
            image_array = message.image.array
            tiled_url = message.tiled_url
            frame_number = message.frame_number if hasattr(message, 'frame_number') else 0

            # Save the image in a separate thread
            success = await asyncio.to_thread(
                self._save_image_sync, 
                image_array, 
                tiled_url
            )

            if success:
                logger.info(f"XPS image saved successfully to: {tiled_url}")
                return tiled_url
            else:
                logger.warning(f"Failed to save XPS image from frame {frame_number}")
                return None

        except Exception as e:
            logger.error(f"Error publishing XPS data to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def from_settings(cls, settings):
        """Create an XPSTiledLocalImagePublisher from settings."""
        return cls()

    def get_local_url_for(self, original_url):
        """
        For XPS, the original URL is already the local URL, so just return it if saved.
        """
        if not original_url:
            return None

        # Extract path and check if we saved it
        path_parts = self._extract_path_from_url(original_url)
        if not path_parts:
            return None
        
        image_key = '/'.join(path_parts)
        
        if image_key in self.saved_images:
            return original_url
        
        return None