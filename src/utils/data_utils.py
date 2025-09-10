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


def get_daily_containers():
    """
    Retrieve all available daily containers from Tiled
    
    Returns:
        list: List of dictionaries with {label: container_name, value: container_name}
    """
    try:
        # Check if tiled_results client is available
        if not tiled_results.check_dataloader_ready():
            logger.warning("Tiled results client not available")
            return []
        
        # Get the daily run container
        container = tiled_results.data_client
        
        # Navigate to the lse_live_results path
        if "lse_live_results" in container:
            container = container["lse_live_results"]
            
            # Find all daily run containers and sort in reverse chronological order
            daily_runs = sorted([k for k in container.keys() if k.startswith("daily_run_")], reverse=True)
            
            if not daily_runs:
                logger.warning("No daily run containers found")
                return []
                
            # Format as dropdown options with human-readable labels
            return [{"label": format_container_name(run), "value": run} for run in daily_runs]
            
        else:
            logger.warning("lse_live_results container not found")
            return []
            
    except Exception as e:
        logger.error(f"Error retrieving daily containers: {e}")
        return []

def format_container_name(container_name):
    """Format a container name for display in the dropdown"""
    if container_name.startswith("daily_run_"):
        # Extract the date part
        date_str = container_name[10:]  # Skip "daily_run_"
        return f"Daily Run {date_str}"
    return container_name

def get_uuids_in_container(container_name):
    """
    Retrieve all available experiment UUIDs from a specific daily container
    
    Args:
        container_name (str): The name of the daily container
        
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
        
        # Navigate to the lse_live_results/container_name path
        if "lse_live_results" not in container:
            logger.warning("lse_live_results container not found")
            return []
            
        container = container["lse_live_results"]
        
        if container_name not in container:
            logger.warning(f"Container {container_name} not found")
            return []
            
        # Get all UUIDs in this container
        uuids = list(container[container_name].keys())
        
        # Format as dropdown options
        return [{"label": uuid, "value": uuid} for uuid in uuids]
            
    except Exception as e:
        logger.error(f"Error retrieving UUIDs from container {container_name}: {e}")
        return []
