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


# ============= UPDATED/NEW FUNCTIONS FOR USER HIERARCHY =============

def get_daily_containers(beamline_path=None):
    """
    Retrieve all available daily containers for the current user from Tiled
    
    Args:
        beamline_path (str, optional): Path to beamline prefix (e.g., "beamlines/bl931/processed")
    
    Returns:
        list: List of dictionaries with {label: container_name, value: container_name}
    """
    try:
        # Get username from environment
        username = os.getenv("USER", "default_user")
        
        # Check if tiled_results client is available
        if not tiled_results.check_dataloader_ready():
            logger.warning("Tiled results client not available")
            return []
        
        # Get the root container
        container = tiled_results.data_client
        
        # NEW: Navigate to beamline path if provided
        if beamline_path:
            path_parts = beamline_path.split('/')
            for part in path_parts:
                if part and part in container:
                    container = container[part]
                elif part:
                    logger.warning(f"Beamline path segment '{part}' not found")
                    return []
        
        # Navigate to the lse_live_results/username path
        if "lse_live_results" in container:
            container = container["lse_live_results"]
            
            # Check if user container exists
            if username not in container:
                logger.warning(f"User {username} not found in lse_live_results")
                return []
            
            user_container = container[username]
            
            # Find all daily run containers and sort in reverse chronological order
            daily_runs = sorted([k for k in user_container.keys() if k.startswith("daily_run_")], reverse=True)
            
            if not daily_runs:
                logger.warning(f"No daily run containers found for user {username}")
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


def get_experiment_names_in_container(container_name, beamline_path=None):
    """
    Retrieve all available experiment names from a specific daily container for the current user
    
    Args:
        container_name (str): The name of the daily container (e.g., "daily_run_2025-08-20")
        beamline_path (str, optional): Path to beamline prefix (e.g., "beamlines/bl931/processed")
        
    Returns:
        list: List of dictionaries with {label: experiment_name, value: experiment_name}
    """
    try:
        # Get username from environment
        username = os.getenv("USER", "default_user")
        
        # Check if tiled_results client is available
        if not tiled_results.check_dataloader_ready():
            logger.warning("Tiled results client not available")
            return []
        
        # Get the root container
        container = tiled_results.data_client
        
        # NEW: Navigate to beamline path if provided
        if beamline_path:
            path_parts = beamline_path.split('/')
            for part in path_parts:
                if part and part in container:
                    container = container[part]
                elif part:
                    logger.warning(f"Beamline path segment '{part}' not found")
                    return []
        
        # Navigate to the lse_live_results/username/container_name path
        if "lse_live_results" not in container:
            logger.warning("lse_live_results container not found")
            return []
            
        container = container["lse_live_results"]
        
        if username not in container:
            logger.warning(f"User {username} not found")
            return []
        
        user_container = container[username]
        
        if container_name not in user_container:
            logger.warning(f"Container {container_name} not found for user {username}")
            return []
            
        daily_container = user_container[container_name]
        
        # Get all experiment names (containers) in this daily container
        experiment_names = [key for key in daily_container.keys()]
        
        if not experiment_names:
            logger.warning(f"No experiment names found in {username}/{container_name}")
            return []
        
        # Format as dropdown options
        return [{"label": name, "value": name} for name in experiment_names]
            
    except Exception as e:
        logger.error(f"Error retrieving experiment names from {container_name}: {e}")
        return []


def get_uuids_in_experiment(container_name, experiment_name, beamline_path=None):
    """
    Retrieve all available experiment UUIDs from a specific experiment for the current user
    
    Args:
        container_name (str): The name of the daily container (e.g., "daily_run_2025-08-20")
        experiment_name (str): The name of the experiment (user-entered name)
        beamline_path (str, optional): Path to beamline prefix (e.g., "beamlines/bl931/processed")
        
    Returns:
        list: List of dictionaries with {label: uuid, value: uuid}
    """
    try:
        # Get username from environment
        username = os.getenv("USER", "default_user")
        
        # Check if tiled_results client is available
        if not tiled_results.check_dataloader_ready():
            logger.warning("Tiled results client not available")
            return []
        
        # Get the root container
        container = tiled_results.data_client
        
        # NEW: Navigate to beamline path if provided
        if beamline_path:
            path_parts = beamline_path.split('/')
            for part in path_parts:
                if part and part in container:
                    container = container[part]
                elif part:
                    logger.warning(f"Beamline path segment '{part}' not found")
                    return []
        
        # Navigate to the lse_live_results/username/container_name/experiment_name path
        if "lse_live_results" not in container:
            logger.warning("lse_live_results container not found")
            return []
            
        container = container["lse_live_results"]
        
        if username not in container:
            logger.warning(f"User {username} not found")
            return []
        
        user_container = container[username]
        
        if container_name not in user_container:
            logger.warning(f"Container {container_name} not found for user {username}")
            return []
            
        daily_container = user_container[container_name]
        
        if experiment_name not in daily_container:
            logger.warning(f"Experiment {experiment_name} not found in {username}/{container_name}")
            return []
        
        experiment_container = daily_container[experiment_name]
        
        # Get all UUIDs (tables) in this experiment
        uuids = list(experiment_container.keys())
        
        if not uuids:
            logger.warning(f"No UUIDs found in {username}/{container_name}/{experiment_name}")
            return []
        
        # Format as dropdown options
        return [{"label": uuid, "value": uuid} for uuid in uuids]
            
    except Exception as e:
        logger.error(f"Error retrieving UUIDs from {container_name}/{experiment_name}: {e}")
        return []


def get_experiment_dataframe(container_name, experiment_name, uuid, beamline_path=None):
    """
    Retrieve the DataFrame for a specific UUID from an experiment
    
    Args:
        container_name (str): The name of the daily container (e.g., "daily_run_2025-08-20")
        experiment_name (str): The name of the experiment
        uuid (str): The UUID of the specific table to retrieve
        beamline_path (str, optional): Path to beamline prefix (e.g., "beamlines/bl931/processed")
        
    Returns:
        pandas.DataFrame or None: The DataFrame containing the experiment data, or None if not found
    """
    try:
        # Get username from environment
        username = os.getenv("USER", "default_user")
        
        # Check if tiled_results client is available
        if not tiled_results.check_dataloader_ready():
            logger.warning("Tiled results client not available")
            return None
        
        # Get the root container
        container = tiled_results.data_client
        
        # NEW: Navigate to beamline path if provided
        if beamline_path:
            path_parts = beamline_path.split('/')
            for part in path_parts:
                if part and part in container:
                    container = container[part]
                elif part:
                    logger.warning(f"Beamline path segment '{part}' not found")
                    return None
        
        # Navigate to the lse_live_results/username/container_name/experiment_name path
        if "lse_live_results" not in container:
            logger.warning("lse_live_results container not found")
            return None
            
        container = container["lse_live_results"]
        
        if username not in container:
            logger.warning(f"User {username} not found")
            return None
        
        user_container = container[username]
        
        if container_name not in user_container:
            logger.warning(f"Container {container_name} not found for user {username}")
            return None
            
        daily_container = user_container[container_name]
        
        if experiment_name not in daily_container:
            logger.warning(f"Experiment {experiment_name} not found in {username}/{container_name}")
            return None
        
        experiment_container = daily_container[experiment_name]
        
        if uuid not in experiment_container:
            logger.warning(f"UUID {uuid} not found in {username}/{container_name}/{experiment_name}")
            return None
        
        # Get and return the DataFrame
        df = experiment_container[uuid].read()
        
        if df is not None and not df.empty:
            logger.info(f"Successfully loaded DataFrame with shape {df.shape} from {username}/{container_name}/{experiment_name}/{uuid}")
        else:
            logger.warning(f"DataFrame is empty for {username}/{container_name}/{experiment_name}/{uuid}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving DataFrame from {container_name}/{experiment_name}/{uuid}: {e}")
        return None