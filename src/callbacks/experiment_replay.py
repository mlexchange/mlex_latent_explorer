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
    Input("replay-data-percentage", "value"),
    State("replay-buffer", "data"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def filter_experiment_by_percentage(percentage_range, replay_buffer, data_project_dict):
    """Filter experiment data by percentage range"""
    # Check if buffer is empty or doesn't have feature_vectors
    if not replay_buffer or "feature_vectors" not in replay_buffer:
        raise PreventUpdate
        
    # Check if we're in replay mode
    if not data_project_dict or not data_project_dict.get("replay_mode", False):
        raise PreventUpdate
        
    try:
        # Get the feature vectors from the replay buffer
        feature_vectors_full = np.array(replay_buffer["feature_vectors"])
        
        total_points = len(feature_vectors_full)
        if total_points == 0:
            raise PreventUpdate
            
        # Calculate the range indices based on percentage
        start_percent, end_percent = percentage_range
        
        # Handle full range (0-100) the same way as partial ranges
        start_idx = int((start_percent / 100) * total_points)
        end_idx = int((end_percent / 100) * total_points)
        
        # Ensure at least one point is shown
        end_idx = max(end_idx, start_idx + 1)
        
        # Create filtered data
        filtered_vectors = feature_vectors_full[start_idx:end_idx]
        
        # Get experiment UUID
        experiment_uuid = replay_buffer.get("uuid", "unknown")
        
        # Rebuild the scatter plot if we have vectors
        if len(filtered_vectors) > 0:
            n_components = filtered_vectors.shape[1]
            
            # Create a new scatter plot
            scatter_fig = generate_scatter_data(filtered_vectors, n_components)
            
            # Update stats text
            if start_percent == 0 and end_percent == 100:
                stats_text = f"Showing all {len(filtered_vectors)} feature vectors from experiment {experiment_uuid}"
            else:
                stats_text = f"Showing {len(filtered_vectors)} feature vectors ({start_percent}% - {end_percent}% of data) from experiment {experiment_uuid}"
            
            return scatter_fig, stats_text
                
        # If we get here, we couldn't create a valid plot
        return no_update, f"No data available in the selected range ({start_percent}% - {end_percent}%)"
        
    except Exception as e:
        logger.error(f"Error filtering experiment data: {e}")
        logger.error(traceback.format_exc())
        return no_update, f"Error filtering data: {str(e)}"


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Output({"base_id": "file-manager", "name": "data-project-dict"}, "data", allow_duplicate=True),
    Output("replay-buffer", "data"),  # Use our new replay-buffer with structured format
    Output("replay-data-percentage", "value"),  # Reset the slider
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
    
    # Initialize an empty structured replay buffer
    replay_buffer = {
        "feature_vectors": [],
        "live_indices": [],
        "uuid": selected_uuid
    }
    
    try:
        # Connect to Tiled and load feature vectors
        logger.info(f"Loading experiment UUID: {selected_uuid}")
        
        if not tiled_results.check_dataloader_ready():
            return no_update, f"Could not connect to Tiled server", data_project_dict, replay_buffer, [0, 100]

        # Build the path to the selected experiment table
        container = tiled_results.data_client
        
        # Navigate to lse_live_results/daily_run_YYYY-MM-DD/selected_uuid
        if "lse_live_results" not in container:
            return no_update, "lse_live_results container not found in Tiled", data_project_dict, replay_buffer, [0, 100]
            
        container = container["lse_live_results"]
        
        # Find the most recent daily run container
        daily_runs = sorted([k for k in container.keys() if k.startswith("daily_run_")], reverse=True)
        
        if not daily_runs:
            return no_update, "No experiment data found", data_project_dict, replay_buffer, [0, 100]
            
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
            return no_update, f"Could not load data for experiment {selected_uuid}", data_project_dict, replay_buffer, [0, 100]
        
        # Extract feature vectors from DataFrame
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            return no_update, "No feature vectors found in the selected data", data_project_dict, replay_buffer, [0, 100]
            
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
        live_indices = []
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
        
        # Create a scatter plot from the feature vectors
        n_components = 2  # Default to 2D
        if feature_vectors.shape[1] > 2:
            n_components = min(3, feature_vectors.shape[1])
            
        scatter_fig = generate_scatter_data(feature_vectors, n_components)
        
        # Store data in the replay buffer
        replay_buffer = {
            "feature_vectors": feature_vectors.tolist(),
            "live_indices": live_indices,
            "uuid": selected_uuid
        }
        
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
        
        return scatter_fig, stats_text, data_project_dict, replay_buffer, [0, 100]
            
    except Exception as e:
        logger.error(f"Error loading experiment: {e}")
        logger.error(traceback.format_exc())
        return no_update, f"Error loading experiment: {str(e)}", data_project_dict, replay_buffer, [0, 100]