import logging
import traceback
import numpy as np
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject

from src.utils.data_utils import get_available_experiment_uuids, tiled_results
from src.utils.plot_utils import (
    generate_scatter_data, 
    plot_empty_heatmap, 
    plot_empty_scatter,
    generate_notification
)

logger = logging.getLogger("lse.replay")


@callback(
    Output("experiment-uuid-dropdown", "options"),
    Output("experiment-uuid-dropdown", "value"),
    Input("refresh-experiment-uuids", "n_clicks"),
    Input("sidebar", "active_item"),  # Also load when sidebar tab is selected
    prevent_initial_call=True,
)
def load_experiment_uuids(n_clicks, active_item):
    """Load available experiment UUIDs from Tiled"""
    uuids = get_available_experiment_uuids()
    
    if uuids:
        return uuids, uuids[0]["value"]
    else:
        return [], None


@callback(
    Output("load-experiment-button", "disabled"),
    Input("experiment-uuid-dropdown", "value"),
)
def toggle_load_button(selected_uuid):
    """Enable/disable load button based on UUID selection"""
    return selected_uuid is None


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Output({"base_id": "file-manager", "name": "data-project-dict"}, "data", allow_duplicate=True),
    Output("live-indices", "data", allow_duplicate=True),  # Add live-indices output
    Input("load-experiment-button", "n_clicks"),
    State("experiment-uuid-dropdown", "value"),
    prevent_initial_call=True,
)
def load_experiment_replay(n_clicks, selected_uuid):
    """Load feature vectors from selected experiment UUID and display in scatter plot"""
    if not n_clicks or not selected_uuid:
        raise PreventUpdate
    
    # Initialize a minimal data project dictionary with replay_mode flag
    data_project_dict = {
        "root_uri": "",
        "data_type": "tiled",
        "datasets": [],
        "project_id": f"replay_{selected_uuid}",
        "replay_mode": True  # Flag to prevent update_data_overview from running
    }
    
    # Initialize an empty list for the indices
    live_indices = []
    
    try:
        # Connect to Tiled and load feature vectors
        logger.info(f"Loading experiment UUID: {selected_uuid}")
        
        if not tiled_results.check_dataloader_ready():
            return no_update, f"Could not connect to Tiled server", data_project_dict, live_indices

        # Build the path to the selected experiment table
        container = tiled_results.data_client
        
        # Navigate to lse_live_results/daily_run_YYYY-MM-DD/selected_uuid
        if "lse_live_results" not in container:
            return no_update, "lse_live_results container not found in Tiled", data_project_dict, live_indices
            
        container = container["lse_live_results"]
        
        # Find the most recent daily run container
        daily_runs = sorted([k for k in container.keys() if k.startswith("daily_run_")], reverse=True)
        
        if not daily_runs:
            return no_update, "No experiment data found", data_project_dict, live_indices
            
        # Get the latest daily run containing our UUID
        df = None
        run_used = None
        for run in daily_runs:
            try:
                if selected_uuid in container[run]:
                    df = container[run][selected_uuid].read()
                    run_used = run
                    logger.info(f"Loaded DataFrame with shape: {df.shape} from run: {run}")
                    break
            except Exception as e:
                logger.warning(f"Error loading data from {run}/{selected_uuid}: {e}")
                continue
        
        if df is None:
            return no_update, f"Could not load data for experiment {selected_uuid}", data_project_dict, live_indices
        
        # Extract feature vectors from DataFrame
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            return no_update, "No feature vectors found in the selected data", data_project_dict, live_indices
            
        # Create numpy array with feature vectors
        import numpy as np
        feature_vectors = df[feature_cols].values
        
        # Log sample of feature vectors (first 3 rows)
        sample_size = min(3, feature_vectors.shape[0])
        logger.info(f"Sample feature vectors (first {sample_size} rows):")
        for i, sample in enumerate(feature_vectors[:sample_size]):
            logger.info(f"Row {i}: {sample}")
        
        # Extract tiled_urls for reference
        tiled_urls = df['tiled_url'].tolist() if 'tiled_url' in df.columns else []
        
        # Extract slice indices from URLs
        if tiled_urls:
            import re
            
            # Initialize live_indices as a list of the same length as feature_vectors
            live_indices = [-1] * len(feature_vectors)
            
            # Extract slice indices from URLs
            for i, url in enumerate(tiled_urls):
                # Look for slice parameter in the URL
                slice_match = re.search(r'slice=(\d+):', url)
                if slice_match:
                    slice_index = int(slice_match.group(1))
                    live_indices[i] = slice_index
                    
                    # Log some samples for debugging
                    if i < 3:
                        logger.info(f"URL {i}: Extracted slice index {slice_index} from {url}")
            
            # Log the slice indices
            logger.info(f"Extracted {sum(1 for idx in live_indices if idx >= 0)} slice indices")
            logger.info(f"Sample indices: {live_indices[:5]}")
        
        # Log some sample URLs
        if tiled_urls:
            url_sample_size = min(3, len(tiled_urls))
            logger.info(f"Sample tiled_urls (first {url_sample_size}):")
            for i, url in enumerate(tiled_urls[:url_sample_size]):
                logger.info(f"URL {i}: {url}")
        else:
            logger.info("No tiled_urls found in DataFrame")
        
        # Create a scatter plot from the feature vectors
        n_components = 2  # Default to 2D
        if feature_vectors.shape[1] > 2:
            n_components = min(3, feature_vectors.shape[1])
            
        scatter_fig = generate_scatter_data(feature_vectors, n_components)
        
        # Let's try to create a more complete data project dictionary
        # Extract information from the first URL if available
        if tiled_urls:
            try:
                from urllib.parse import urlparse
                
                # Parse the first URL
                first_url = urlparse(tiled_urls[0])
                
                # Extract base URI
                base_uri = f"{first_url.scheme}://{first_url.netloc}"
                logger.info(f"Base URI extracted: {base_uri}")
                
                # Extract path components
                path = first_url.path
                logger.info(f"Path from URL: {path}")
                
                # Extract slice info from query
                query = first_url.query
                logger.info(f"Query from URL: {query}")
                
                # Try to create a useful dataset structure
                # Typically path will be like /api/v1/array/full/smi/raw/UUID/primary/data/pil2M_image
                parts = path.split('/api/v1/')
                if len(parts) > 1:
                    uri_path = parts[1]
                    logger.info(f"Path after /api/v1/: {uri_path}")
                    
                    if uri_path.startswith(('array/full/', 'metadata/')):
                        uri_path = uri_path.split('/', 2)[2]  # Get path after array/full/ or metadata/
                        logger.info(f"Final URI path: {uri_path}")
                    
                    # Create a dataset with this path
                    datasets = [
                        {
                            "uri": uri_path,
                            "cumulative_data_count": len(df)
                        }
                    ]
                    
                    # Create data project with the extracted components
                    data_project_dict = {
                        "root_uri": f"{base_uri}/api/v1/metadata",
                        "data_type": "tiled",
                        "datasets": datasets,
                        "project_id": f"replay_{selected_uuid}",
                        "replay_mode": True  # Keep the replay_mode flag
                    }
                else:
                    # Fallback if parsing fails - use the initialized data_project_dict
                    pass
            except Exception as e:
                logger.warning(f"Error parsing URL for data project: {e}")
                # Fallback to the initialized data_project_dict
                pass
        else:
            # No URLs available - use the initialized data_project_dict
            pass
        
        # Log the final data project dictionary
        import json
        logger.info(f"Created data project dictionary:")
        logger.info(json.dumps(data_project_dict, indent=2))
        
        # Success message
        stats_text = f"Loaded {len(df)} feature vectors from experiment {selected_uuid}"
        
        return scatter_fig, stats_text, data_project_dict, live_indices
            
    except Exception as e:
        logger.error(f"Error loading experiment: {e}")
        logger.error(traceback.format_exc())
        return no_update, f"Error loading experiment: {str(e)}", data_project_dict, live_indices