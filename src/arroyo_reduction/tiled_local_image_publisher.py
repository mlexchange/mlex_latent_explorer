import asyncio
import logging
import os
import re
from urllib.parse import urlparse, parse_qs

import numpy as np
from arroyopy.publisher import Publisher
from arroyosas.schemas import SASStop, RawFrameEvent
from tiled.client import from_uri

logger = logging.getLogger("arroyo_reduction.tiled_local_image_publisher")

# Environment variables for local Tiled connections
LOCAL_TILED_URI = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")
LOCAL_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

class TiledLocalImagePublisher(Publisher):
    """Publisher that saves raw images to a local Tiled server cache."""

    def __init__(self, tiled_uri=None, tiled_api_key=None, container_name="live_cache"):
        super().__init__()
        self.tiled_uri = tiled_uri or LOCAL_TILED_URI
        self.tiled_api_key = tiled_api_key or LOCAL_TILED_API_KEY
        self.container_name = container_name
        self.client = None
        self.root_container = None
        
        # Dictionary to track saved images to avoid duplicates
        self.saved_images = set()
        
        logger.info(f"Initialized TiledLocalImagePublisher for container: {container_name}")

    async def start(self):
        """Connect to Tiled server and initialize containers."""
        try:
            # Run the entire initialization in a separate thread
            await asyncio.to_thread(self._start_sync)
        except Exception as e:
            logger.error(f"Failed to initialize Tiled client: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _start_sync(self):
        """Synchronous implementation of start() to be run in a thread."""
        try:
            self.client = from_uri(self.tiled_uri, api_key=self.tiled_api_key)
            
            # Create live_cache container if it doesn't exist
            if self.container_name not in self.client:
                logger.info(f"Creating container: {self.container_name}")
                self.root_container = self.client.create_container(self.container_name)
            else:
                logger.info(f"Using existing container: {self.container_name}")
                self.root_container = self.client[self.container_name]
                
            logger.info(f"Connected to Tiled server at {self.tiled_uri}")
            logger.info(f"Using container: {self.container_name}")
        except Exception as e:
            logger.error(f"Error in _start_sync: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise so the async method can catch it
    
    def _parse_tiled_url(self, url):
        """
        Parse a Tiled URL to extract components and frame index.
        Handles both production and testing formats:
        
        Production: http://domain/api/v1/array/full/UUID/streams/primary/pil2M_image?slice=1:2,0:1679,0:1475
        Testing: http://domain/api/v1/array/full/container/image?slice=0:1,0:1679,0:1475
        """
        try:
            # Parse the URL to extract components
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            query_params = parse_qs(parsed_url.query)
            
            # Extract slice information and frame index
            slice_param = query_params.get('slice', [''])[0]
            frame_index = 0
            image_h = 0
            image_w = 0
            
            if slice_param:
                slice_parts = slice_param.split(',')
                if len(slice_parts) >= 1 and ':' in slice_parts[0]:
                    # Extract frame index from "X:X+1" format
                    frame_start = int(slice_parts[0].split(':')[0])
                    frame_index = frame_start
                
                # Try to extract image dimensions from slice
                if len(slice_parts) >= 3:
                    if ':' in slice_parts[1]:
                        try:
                            image_h = int(slice_parts[1].split(':')[1])
                        except (ValueError, IndexError):
                            pass
                    
                    if ':' in slice_parts[2]:
                        try:
                            image_w = int(slice_parts[2].split(':')[1])
                        except (ValueError, IndexError):
                            pass
            
            # Check for UUID in the URL to determine if it's production format
            uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
            uuid_match = re.search(uuid_pattern, url)
            
            if uuid_match:
                # Production format: With UUID
                uuid = uuid_match.group(1)
                
                # Find position of uuid in path
                uuid_idx = -1
                for i, part in enumerate(path_parts):
                    if uuid in part:
                        uuid_idx = i
                        break
                
                if uuid_idx >= 0:
                    # Extract the path components after "array/full"
                    array_full_idx = -1
                    for i, part in enumerate(path_parts):
                        if part == "array" and i < len(path_parts) - 1 and path_parts[i+1] == "full":
                            array_full_idx = i + 1
                            break
                    
                    if array_full_idx >= 0:
                        # Get path after array/full but before and after UUID
                        remaining_path = path_parts[array_full_idx+1:]  # Skip "full"
                        
                        # Find UUID position in remaining path
                        uuid_pos = -1
                        for i, part in enumerate(remaining_path):
                            if uuid in part:
                                uuid_pos = i
                                break
                        
                        if uuid_pos >= 0:
                            # path structure: [uuid, 'streams', 'primary', 'pil2M_image']
                            path_after_uuid = remaining_path[uuid_pos+1:]  # Skip UUID
                            storage_path = f"{uuid}/{'/'.join(path_after_uuid)}"
                            
                            return {
                                "has_uuid": True,
                                "storage_path": storage_path,
                                "frame_index": frame_index,
                                "uuid": uuid,
                                "image_h": image_h,
                                "image_w": image_w,
                                "path_after_uuid": path_after_uuid,
                            }
            
            # Testing format: No UUID, simpler format
            # Extract the path after "array/full"
            array_full_idx = -1
            for i, part in enumerate(path_parts):
                if part == "array" and i < len(path_parts) - 1 and path_parts[i+1] == "full":
                    array_full_idx = i + 1
                    break
            
            if array_full_idx >= 0:
                # Get the parts after "array/full"
                remaining_path = path_parts[array_full_idx+1:]
                
                if len(remaining_path) >= 1:
                    storage_path = '/'.join(remaining_path)
                    
                    return {
                        "has_uuid": False,
                        "storage_path": storage_path,
                        "frame_index": frame_index,
                        "image_h": image_h,
                        "image_w": image_w
                    }
            
            # Default fallback if parsing fails
            return {
                "has_uuid": False,
                "storage_path": "unknown",
                "frame_index": frame_index,
                "image_h": image_h,
                "image_w": image_w
            }
            
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            # Provide a fallback structure so we can still save the image
            return {
                "has_uuid": False,
                "storage_path": "unknown",
                "frame_index": 0,
                "image_h": 0,
                "image_w": 0
            }
    
    def _save_image_sync(self, image_array, url_info):
        """Save image to Tiled server synchronously."""
        try:
            if self.root_container is None:
                logger.error("Container not initialized, cannot save image")
                return False
                
            storage_path = url_info["storage_path"]
            frame_index = url_info["frame_index"]
            
            # Create a unique key for this image to avoid duplicates
            image_key = f"{storage_path}_{frame_index}"
            
            # Skip if we've already saved this image
            if image_key in self.saved_images:
                logger.debug(f"Image already saved: {image_key}")
                return True
            
            if url_info["has_uuid"]:
                # Production format: save to UUID/streams/primary/pil2M_image/frameX
                uuid = url_info["uuid"]
                path_after_uuid = url_info["path_after_uuid"]
                
                # Create containers: live_cache/UUID/streams/primary/pil2M_image/frameX
                current = self.root_container
                
                # Create UUID container
                if uuid not in current:
                    current = current.create_container(uuid)
                else:
                    current = current[uuid]
                
                # Create path after UUID (streams/primary/pil2M_image)
                for component in path_after_uuid:
                    if not component:  # Skip empty components
                        continue
                    if component not in current:
                        current = current.create_container(component)
                    else:
                        current = current[component]
                
                # Create frame container
                frame_name = f"frame{frame_index}"
                
                # Save the image directly in the frame container
                logger.info(f"Saving production image to {self.container_name}/{uuid}/{'/'.join(path_after_uuid)}/{frame_name}")
                if frame_name not in current:
                    current.write_array(image_array,key=frame_name)
            else:
                # Save the image directly
                current = self.root_container
                path_components = storage_path.split('/')
                logger.info(f"image_name:{path_components[-1]}")
                for component in path_components[:-1]:
                    if not component:  # Skip empty components
                        continue
                    if component not in current:
                        current = current.create_container(component)
                    else:
                        current = current[component]
                logger.info(f"Saving test image to {self.container_name}/{storage_path}")
                if path_components[-1] not in current:
                    current.write_array(image_array,key=path_components[-1])
            
            # Mark as saved
            self.saved_images.add(image_key)
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to Tiled: {e}")
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
            
            # Parse the URL to get storage path and frame index
            url_info = self._parse_tiled_url(tiled_url)
            
            # Use frame_number from message if available
            if hasattr(message, 'frame_number') and message.frame_number is not None:
                url_info["frame_index"] = message.frame_number
            
            # Save the image in a separate thread
            success = await asyncio.to_thread(
                self._save_image_sync, 
                image_array, 
                url_info
            )
            
            if success:
                # Get the local URL using our utility function
                local_url = self.get_local_url_for(tiled_url)
                logger.info(f"Image saved successfully, local URL: {local_url}")
                return local_url
            else:
                logger.warning(f"Failed to save image from frame {frame_number}")
                return None
                
        except Exception as e:
            logger.error(f"Error publishing to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    @classmethod
    def from_settings(cls, settings):
        """Create a TiledLocalImagePublisher from settings."""
        return cls(
            container_name=settings.get("container_name", "live_data_cache")
        )
        
    def get_local_url_for(self, original_url):
        """
        Convert an original Tiled URL to its local equivalent.
        Returns None if the URL cannot be converted or the image hasn't been cached.
        """
        if not original_url:
            return None
            
        # Parse the URL to extract path components and frame index
        url_info = self._parse_tiled_url(original_url)
        
        # Check if we have saved this image
        image_key = f"{url_info['storage_path']}_{url_info['frame_index']}"
        if image_key not in self.saved_images:
            return None
            
        # Construct the local URL
        storage_path = url_info["storage_path"]
        frame_index = url_info["frame_index"]
        image_h = url_info.get("image_h", 1679) 
        image_w = url_info.get("image_w", 1475)
        
        if url_info["has_uuid"]:
            # Production format: add frame path
            uuid = url_info["uuid"]
            path_after_uuid = url_info["path_after_uuid"]
            local_path = f"{self.container_name}/{uuid}/{'/'.join(path_after_uuid)}/frame{frame_index}"
            local_url = f"{self.tiled_uri}/api/v1/array/full/{local_path}?slice=0:1,0:{image_h},0:{image_w}"
        else:
            # Testing format: simple path
            local_url = f"{self.tiled_uri}/api/v1/array/full/{self.container_name}/{storage_path}?slice=0:1,0:{image_h},0:{image_w}"
        
        return local_url