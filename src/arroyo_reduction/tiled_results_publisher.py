import asyncio
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz  # Added import for timezone
from arroyopy.publisher import Publisher
from tiled.client import from_uri

from .schemas import LatentSpaceEvent

logger = logging.getLogger("arroyo_reduction.tiled_results_publisher")

# Environment variables for Tiled connections
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

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
        
        # Create a daily run ID using California timezone
        california_tz = pytz.timezone('US/Pacific')  # Added for California timezone
        today = datetime.now(california_tz).strftime("%Y-%m-%d")  # Modified to use California timezone
        self.run_id = f"daily_run_{today}"
        
        # Regex pattern to extract UUID from tiled_url
        self.uuid_pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
        
        logger.info(f"Initialized publisher with UUID-based table grouping")

    async def start(self):
        """Connect to Tiled server and initialize containers."""
        try:
            self.client = from_uri(self.tiled_uri, api_key=self.tiled_api_key)
            
            # Navigate to the root container and create the daily run container inside it
            await self._setup_containers()
            
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
            logger.info(f"Using container path: {'/'.join(self.root_segments)}/{self.run_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Tiled client: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _extract_uuid_from_url(self, url):
        """Extract UUID from tiled_url."""
        if not url:
            return self.default_table_name
        
        # Log the URL for debugging
        logger.debug(f"Extracting UUID from URL: {url}")
        
        match = re.search(self.uuid_pattern, url)
        if match:
            uuid = match.group(1)
            logger.debug(f"Extracted UUID: {uuid}")
            return uuid
        
        logger.debug(f"No UUID found in URL, using default: {self.default_table_name}")
        return self.default_table_name
    
    async def _setup_containers(self):
        """Set up the container structure."""
        try:
            # Navigate through root_segments
            container = self.client
            for segment in self.root_segments:
                if segment not in container:
                    logger.info(f"Creating container: {segment}")
                    container = container.create_container(segment)
                else:
                    logger.info(f"Using existing container: {segment}")
                    container = container[segment]
            
            # Store reference to the root container
            self.root_container = container
            
            # Now create the daily run container inside the root container
            if self.run_id not in self.root_container:
                logger.info(f"Creating daily container: {self.run_id}")
                self.root_container.create_container(self.run_id)
            else:
                logger.info(f"Using existing daily container: {self.run_id}")
            
            # Store reference to daily container
            self.daily_container = self.root_container[self.run_id]
            
        except Exception as e:
            logger.error(f"Error setting up containers: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def publish(self, message):
        """Publish a message to Tiled server."""
        if not isinstance(message, LatentSpaceEvent):
            return

        try:
            # Ensure daily container exists
            if self.daily_container is None:
                logger.error("Daily container not initialized, cannot publish")
                return

            # Format vector and metadata
            vector = np.array(message.feature_vector, dtype=np.float32)
            if vector.ndim == 1:
                # Extract UUID from tiled_url
                tiled_url = getattr(message, "tiled_url", None)
                uuid = self._extract_uuid_from_url(tiled_url)
                
                # Check if this UUID already exists
                if uuid in self.existing_uuids:
                    logger.debug(f"Skipping vector for existing UUID: {uuid}")
                    return
                
                # Check if this is a new UUID
                if self.current_uuid is not None and uuid != self.current_uuid and self.current_uuid in self.uuid_dataframes:
                    # We have a new UUID, so write the data for the previous UUID (if it's not an existing UUID)
                    if self.current_uuid not in self.existing_uuids and not self.uuid_dataframes[self.current_uuid].empty:
                        logger.info(f"New UUID detected, writing data for previous UUID: {self.current_uuid}")
                        await self._write_table_to_tiled(self.current_uuid)
                
                # Update current UUID
                self.current_uuid = uuid
                
                # If this UUID already exists, skip further processing
                if uuid in self.existing_uuids:
                    return
                
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
                
            else:
                logger.warning(f"Received vector with unexpected dimensions: {vector.shape}")
                
        except Exception as e:
            logger.error(f"Error publishing to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _write_table_to_tiled(self, table_key):
        """Write the collected vectors for a specific UUID to Tiled."""
        try:
            # Check if this UUID already exists
            if table_key in self.existing_uuids:
                logger.info(f"Skipping write for existing UUID: {table_key}")
                return
            
            # Get the DataFrame for this UUID
            df = self.uuid_dataframes[table_key]
            
            # Log DataFrame info for debugging
            logger.info(f"Writing {len(df)} vectors to new table '{table_key}'")
            
            # Check if DataFrame is empty
            if df.empty:
                logger.warning(f"DataFrame for {table_key} is empty, nothing to write")
                return
            
            # Simply write the DataFrame to Tiled
            try:
                # Use write_dataframe with the UUID as the key
                await asyncio.to_thread(
                    self.daily_container.write_dataframe,
                    df,
                    key=table_key
                )
                
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
            logger.error(f"Error in _write_table_to_tiled for {table_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def stop(self):
        """Write any remaining data for new UUIDs before stopping."""
        logger.info("Publisher stopping, writing any remaining data for new UUIDs")
        
        # Write data for all UUIDs that have accumulated data and don't already exist
        for uuid in list(self.uuid_dataframes.keys()):
            if uuid not in self.existing_uuids and not self.uuid_dataframes[uuid].empty:
                logger.info(f"Writing remaining data for new UUID: {uuid}")
                await self._write_table_to_tiled(uuid)
        
        logger.info("Publisher stopped")

    @classmethod
    def from_settings(cls, settings):
        """Create a TiledResultsPublisher from settings."""
        return cls(root_segments=settings.get("root_segments"))