import asyncio
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from arroyopy.publisher import Publisher
from arroyosas.schemas import SASStop
from tiled.client import from_uri

from .schemas import LatentSpaceEvent

logger = logging.getLogger("arroyo_reduction.tiled_results_publisher")

# Environment variables for Tiled connections
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

# Constants
# Timezone for log timestamps
CALIFORNIA_TZ = pytz.timezone('US/Pacific')
# Daily run ID that all instances will use
DAILY_RUN_ID = f"daily_run_{datetime.now(CALIFORNIA_TZ).strftime('%Y-%m-%d')}"
# Regex pattern to extract UUID from tiled_url
UUID_PATTERN = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"

class TiledResultsPublisher(Publisher):
    """Publisher that saves latent space vectors to a Tiled server."""

    def __init__(self, tiled_uri=None, tiled_api_key=None, root_segments=None):
        super().__init__()
        self.tiled_uri = tiled_uri or RESULTS_TILED_URI
        self.tiled_api_key = tiled_api_key or RESULTS_TILED_API_KEY
        self.root_segments = root_segments or ["lse_live_results"]
        self.client = None
        self.root_container = None
        self.daily_container = None
        
        # Dictionary to store DataFrames by UUID
        self.uuid_dataframes = {}
        # Set to track UUIDs that already exist in Tiled
        self.existing_uuids = set()
        # Default table name if no UUID is available
        self.default_table_name = "feature_vectors"
        # Keep track of the current UUID
        self.current_uuid = None
        
        logger.info(f"Initialized publisher with UUID-based table grouping")

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
            
            # Navigate to the root container and create the daily run container inside it
            self._setup_containers_sync()
            
            # List all existing tables in the daily container
            if self.daily_container is not None:
                table_keys = list(self.daily_container)
                logger.info(f"Found {len(table_keys)} existing tables in daily container")
                
                # Add all existing tables to our set of existing UUIDs
                self.existing_uuids.update(table_keys)
                logger.info(f"Tracking {len(self.existing_uuids)} existing UUIDs")
                
                # Log some examples of existing UUIDs for debugging
                if self.existing_uuids:
                    examples = list(self.existing_uuids)[:3]
                    logger.info(f"Examples of existing UUIDs: {', '.join(examples)}")
                    
            logger.info(f"Connected to Tiled server at {self.tiled_uri}")
            logger.info(f"Using container path: {'/'.join(self.root_segments)}/{DAILY_RUN_ID}")
        except Exception as e:
            logger.error(f"Error in _start_sync: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise so the async method can catch it
    
    def _extract_uuid_from_url(self, url):
        """Extract UUID from tiled_url."""
        if not url:
            return self.default_table_name
        
        # Log the URL for debugging
        logger.debug(f"Extracting UUID from URL: {url}")
        
        match = re.search(UUID_PATTERN, url)
        if match:
            uuid = match.group(1)
            logger.debug(f"Extracted UUID: {uuid}")
            return uuid
        
        logger.debug(f"No UUID found in URL, using default: {self.default_table_name}")
        return self.default_table_name
    
    def _setup_containers_sync(self):
        """Set up the container structure (synchronous version)."""
        try:
            # Navigate through root_segments
            container = self.client
            for segment in self.root_segments:
                if segment in container:
                    logger.info(f"Using existing container: {segment}")
                    container = container[segment]
                else:
                    logger.info(f"Creating container: {segment}")
                    container = container.create_container(segment)
            
            # Store reference to the root container
            self.root_container = container
            
            # Now create the daily run container inside the root container
            if DAILY_RUN_ID not in self.root_container:
                logger.info(f"Creating daily container: {DAILY_RUN_ID}")
                self.root_container.create_container(DAILY_RUN_ID)
            else:
                logger.info(f"Using existing daily container: {DAILY_RUN_ID}")
            
            # Store reference to daily container
            self.daily_container = self.root_container[DAILY_RUN_ID]
            
        except Exception as e:
            logger.error(f"Error setting up containers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise to propagate the error
    
    async def publish(self, message):
        """Publish a message to Tiled server."""
        if isinstance(message, SASStop):
            logger.info("Received Stop message, writing any remaining data to Tiled")
            await self.stop()
            return
        
        if not isinstance(message, LatentSpaceEvent):
            return

        try:
            # Run the entire publish operation in a separate thread
            uuid_to_write = await asyncio.to_thread(self._publish_sync, message)
            
            # If there's a UUID to write, write it
            if uuid_to_write:
                await self.write_table_to_tiled(uuid_to_write)
                
        except Exception as e:
            logger.error(f"Error publishing to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _publish_sync(self, message):
        """Synchronous implementation of publish() to be run in a thread."""
        try:
            # Ensure daily container exists
            if self.daily_container is None:
                logger.error("Daily container not initialized, cannot publish")
                return None

            # Format vector and metadata
            vector = np.array(message.feature_vector, dtype=np.float32)
            if vector.ndim == 1:
                # Extract UUID from tiled_url
                tiled_url = getattr(message, "tiled_url", None)
                uuid = self._extract_uuid_from_url(tiled_url)
                
                # Check if this UUID already exists
                if uuid in self.existing_uuids:
                    logger.debug(f"Skipping vector for existing UUID: {uuid}")
                    return None
                
                # Check if this is a new UUID
                uuid_to_write = None
                
                if self.current_uuid is not None and uuid != self.current_uuid and self.current_uuid in self.uuid_dataframes:
                    # We have a new UUID, so write the data for the previous UUID (if it's not an existing UUID)
                    if self.current_uuid not in self.existing_uuids and not self.uuid_dataframes[self.current_uuid].empty:
                        logger.info(f"New UUID detected, marking previous UUID for writing: {self.current_uuid}")
                        uuid_to_write = self.current_uuid
                
                # Update current UUID
                self.current_uuid = uuid
                
                # Initialize tracking for this UUID if needed
                if uuid not in self.uuid_dataframes:
                    self.uuid_dataframes[uuid] = pd.DataFrame()
                
                # Create a record with metadata and the vector
                record = {
                    "tiled_url": tiled_url,
                    "autoencoder_model": getattr(message, "autoencoder_model", None),
                    "dimred_model": getattr(message, "dimred_model", None),
                    "timestamp": getattr(message, "timestamp", time.time()),
                    "total_processing_time": getattr(message, "total_processing_time", None),
                    "autoencoder_time": getattr(message, "autoencoder_time", None),
                    "dimred_time": getattr(message, "dimred_time", None)
                }
                
                # Add vector elements as columns (limit to first 20 to keep it manageable)
                for i, val in enumerate(vector[:20]):
                    record[f"feature_{i}"] = float(val)
                
                # Append to DataFrame for this UUID
                new_row = pd.DataFrame([record])
                self.uuid_dataframes[uuid] = pd.concat([self.uuid_dataframes[uuid], new_row], ignore_index=True)
                
                logger.debug(f"Added vector to table '{uuid}'")
                
                return uuid_to_write
            else:
                logger.warning(f"Received vector with unexpected dimensions: {vector.shape}")
                return None
        except Exception as e:
            logger.error(f"Error in _publish_sync: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def write_table_to_tiled(self, table_key):
        """Write the collected vectors for a specific UUID to Tiled."""
        try:
            # Run the write operation in a separate thread
            await asyncio.to_thread(self._write_table_to_tiled_sync, table_key)
        except Exception as e:
            logger.error(f"Error in write_table_to_tiled for {table_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _write_table_to_tiled_sync(self, table_key):
        """Synchronous implementation of write_table_to_tiled to be run in a thread."""
        try:
            # Check if this UUID already exists
            if table_key in self.existing_uuids:
                logger.info(f"Skipping write for existing UUID: {table_key}")
                return
            
            # Get the DataFrame for this UUID
            df = self.uuid_dataframes.get(table_key)
            if df is None:
                logger.warning(f"No DataFrame found for {table_key}")
                return
                
            # Log DataFrame info for debugging
            logger.info(f"Writing {len(df)} vectors to new table '{table_key}'")
            
            # Check if DataFrame is empty
            if df.empty:
                logger.warning(f"DataFrame for {table_key} is empty, nothing to write")
                return
            
            # Simply write the DataFrame to Tiled
            try:
                # Use write_dataframe with the UUID as the key
                self.daily_container.write_dataframe(df, key=table_key)
                
                logger.info(f"Successfully wrote {len(df)} vectors to '{table_key}'")
                
                # Add this UUID to our set of existing UUIDs
                self.existing_uuids.add(table_key)
                
                # Clear the DataFrame for this UUID
                self.uuid_dataframes[table_key] = pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Error writing DataFrame for {table_key}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"Error in _write_table_to_tiled_sync for {table_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def stop(self):
        """Write any remaining data for new UUIDs before stopping."""
        try:
            # Run the stopping operation in a separate thread to get UUID to write
            uuid_to_write = await asyncio.to_thread(self._stop_sync)
            
            # If there's a UUID to write, write it
            if uuid_to_write:
                logger.info(f"Writing final data for UUID: {uuid_to_write}")
                await self.write_table_to_tiled(uuid_to_write)
                
            logger.info("Publisher stopped")
        except Exception as e:
            logger.error(f"Error stopping publisher: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _stop_sync(self):
        """Synchronous implementation of stop() to be run in a thread.
        
        Returns:
            str or None: UUID that needs to be written, or None if no writing needed
        """
        try:
            logger.info("Publisher stopping, checking if current UUID needs writing")
            
            # Check if the current UUID needs writing
            if (self.current_uuid is not None and 
                self.current_uuid not in self.existing_uuids and 
                self.current_uuid in self.uuid_dataframes and 
                not self.uuid_dataframes[self.current_uuid].empty):
                
                return self.current_uuid
            
            return None
        except Exception as e:
            logger.error(f"Error in _stop_sync: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def from_settings(cls, settings):
        """Create a TiledResultsPublisher from settings."""
        return cls(root_segments=settings.get("root_segments"))