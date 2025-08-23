import hashlib
import logging
import os

import httpx
from humanhash import humanize
from tiled.client import from_uri

RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

logger = logging.getLogger("lse.data_utils")


class TiledDataLoader:
    def __init__(self, data_tiled_uri, data_tiled_api_key):
        self.data_tiled_uri = data_tiled_uri
        self.data_tiled_api_key = data_tiled_api_key
        self.refresh_data_client()

    def refresh_data_client(self):
        try:
            self.data_client = from_uri(
                self.data_tiled_uri,
                api_key=self.data_tiled_api_key,
                timeout=httpx.Timeout(30.0),
            )
        except Exception as e:
            logger.warning(f"Error connecting to Tiled: {e}")
            self.data_client = None

    def check_dataloader_ready(self):
        """
        Check if the data client is available and ready to be used.
        If base_only is True, only check the base uri.
        """
        if self.data_client is None:
            # Try refreshing once
            self.refresh_data_client()
            return False if self.data_client is None else True
        else:
            try:
                headers = {"Authorization": f"Bearer {self.data_tiled_api_key}"}
                httpx.get(self.data_tiled_uri, headers=headers)
            except Exception as e:
                logger.warning(f"Error connecting to Tiled: {e}")
                return False
        return True

    def prepare_project_container(self, user, project_name):
        """
        Prepare a project container in the data store
        """
        last_container = self.data_client
        for part in [user, project_name]:
            if part in last_container.keys():
                last_container = last_container[part]
            else:
                last_container = last_container.create_container(key=part)
        return last_container

    def get_data_by_trimmed_uri(self, trimmed_uri, slice=None):
        """
        Retrieve data by a trimmed uri (not containing the base uri) and slice id
        """
        if slice is None:
            return self.data_client[trimmed_uri]
        else:
            return self.data_client[trimmed_uri][slice]

    def get_metadata_by_trimmed_uri(self, trimmed_uri):
        """
        Retrieve metadata by a trimmed uri (not containing the base uri)
        """
        return self.data_client[trimmed_uri].metadata


tiled_results = TiledDataLoader(
    data_tiled_uri=RESULTS_TILED_URI, data_tiled_api_key=RESULTS_TILED_API_KEY
)


def hash_list_of_strings(strings_list):
    """
    Produces a hash of a list of strings.
    """
    concatenated = "".join(strings_list)
    digest = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()
    return humanize(digest)


def get_available_experiment_uuids():
    """
    Retrieve all available experiment UUIDs from Tiled that were written
    by tiled_results_publisher.py
    
    Returns:
        list: List of dictionaries with {label: uuid, value: uuid}
    """
    try:
        # Check if tiled_results client is available
        if not tiled_results.check_dataloader_ready():
            logger.warning("Tiled results client not available")
            return []
        
        # Get the daily run container
        container = tiled_results.data_client
        
        # Navigate to the lse_live_results/daily_run_YYYY-MM-DD path
        if "lse_live_results" in container:
            container = container["lse_live_results"]
            
            # Find the most recent daily run container
            daily_runs = sorted([k for k in container.keys() if k.startswith("daily_run_")], reverse=True)
            
            if not daily_runs:
                logger.warning("No daily run containers found")
                return []
                
            # Get the latest daily run
            latest_run = daily_runs[0]
            container = container[latest_run]
            
            # Get all UUIDs (these are the keys in the container)
            uuids = list(container.keys())
            
            # Format as dropdown options
            return [{"label": uuid, "value": uuid} for uuid in uuids]
            
        else:
            logger.warning("lse_live_results container not found")
            return []
            
    except Exception as e:
        logger.error(f"Error retrieving experiment UUIDs: {e}")
        return []
