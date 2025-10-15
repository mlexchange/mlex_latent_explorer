import logging
import traceback
import numpy as np
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject

from src.utils.data_utils import (
    get_daily_containers, 
    get_experiment_names_in_container,  
    get_uuids_in_experiment,
    get_experiment_dataframe,  # Add this import
    tiled_results
)
from src.utils.plot_utils import (
    generate_scatter_data, 
    plot_empty_heatmap, 
    plot_empty_scatter,
    generate_notification
)

logger = logging.getLogger("lse.replay")

@callback(
    Output("daily-container-dropdown", "options"),
    Output("daily-container-dropdown", "value"),
    Input("refresh-daily-containers", "n_clicks"),
    prevent_initial_call=True,
)
def load_daily_containers(n_clicks):
    """Load available daily containers from Tiled"""
    if n_clicks is None:
        raise PreventUpdate
        
    # CHANGED: Now calls get_daily_containers() without username parameter
    containers = get_daily_containers()
    
    if containers:
        return containers, containers[0]["value"]
    else:
        return [], None
    
@callback(
    Output("daily-container-dropdown", "options", allow_duplicate=True),
    Input("sidebar", "active_item"),
    prevent_initial_call="initial_duplicate",
)
def load_containers_on_render(active_item):
    """Load daily containers when the page is first loaded"""
    # CHANGED: Now calls get_daily_containers() without username parameter
    containers = get_daily_containers()
    return containers if containers else []

@callback(
    Output("experiment-name-dropdown", "options"),
    Output("experiment-name-dropdown", "value"),
    Input("daily-container-dropdown", "value"),
    Input("refresh-experiment-names", "n_clicks"),
    prevent_initial_call=True,
)
def load_experiment_names(selected_container, refresh_clicks):
    """Load available experiment names from selected daily container"""
    if not selected_container:
        return [], None
        
    # CHANGED: Now calls get_experiment_names_in_container() without username parameter
    experiment_names = get_experiment_names_in_container(selected_container)
    
    if experiment_names:
        return experiment_names, experiment_names[0]["value"]
    else:
        return [], None

@callback(
    Output("experiment-uuid-dropdown", "options"),
    Output("experiment-uuid-dropdown", "value"),
    Input("experiment-name-dropdown", "value"),  
    Input("refresh-experiment-uuids", "n_clicks"),
    State("daily-container-dropdown", "value"),
    prevent_initial_call=True,
)
def load_experiment_uuids(selected_experiment, refresh_clicks, selected_container):
    """Load available experiment UUIDs from selected experiment"""
    if not selected_container or not selected_experiment:
        return [], None
        
    # CHANGED: Now calls get_uuids_in_experiment() without username parameter
    uuids = get_uuids_in_experiment(selected_container, selected_experiment)
    
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
    Output("replay-data-range", "max", allow_duplicate=True),
    Input("replay-data-range", "value"),
    State("replay-buffer", "data"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def filter_experiment_by_range(range_value, replay_buffer, data_project_dict):
    """Filter experiment data by index range"""
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
            
        # Make sure range_value contains integer indices
        if range_value is None:
            start_idx, end_idx = 0, total_points
        else:
            # Ensure we have integers for slicing
            start_idx = int(range_value[0])
            end_idx = int(range_value[1])
        
        # Get experiment UUID and container
        experiment_uuid = replay_buffer.get("uuid", "unknown")
        container = replay_buffer.get("container", "unknown")
        
        # Check for invalid range conditions
        if start_idx >= end_idx or start_idx >= total_points or end_idx <= 0:
            # Return an empty scatter plot for invalid ranges
            empty_fig = plot_empty_scatter()
            return empty_fig, total_points  # Remove marks from return
        
        # Ensure end_idx doesn't exceed total_points
        end_idx = min(end_idx, total_points)
        
        # Create filtered data
        filtered_vectors = feature_vectors_full[start_idx:end_idx]
        
        # Rebuild the scatter plot if we have vectors
        if len(filtered_vectors) > 0:
            n_components = filtered_vectors.shape[1]
            
            # Create a new scatter plot
            scatter_fig = generate_scatter_data(filtered_vectors, n_components)
            
            return scatter_fig, total_points
                
        # If we get here, we couldn't create a valid plot (empty filtered_vectors)
        empty_fig = plot_empty_scatter()
        return empty_fig, total_points 
        
    except Exception as e:
        logger.error(f"Error filtering experiment data: {e}")
        logger.error(traceback.format_exc())
        # Return an empty plot on error instead of no_update
        empty_fig = plot_empty_scatter()
        return empty_fig, 100 

@callback(
    Output("scatter", "figure", allow_duplicate=True),
    # Removed stats-div output
    Output({"base_id": "file-manager", "name": "data-project-dict"}, "data", allow_duplicate=True),
    Output("replay-buffer", "data"),
    Output("replay-data-range", "value"),
    Output("replay-data-range", "max"),
    Output("replay-data-range", "marks", allow_duplicate=True),
    Input("load-experiment-button", "n_clicks"),
    State("daily-container-dropdown", "value"),
    State("experiment-name-dropdown", "value"),  
    State("experiment-uuid-dropdown", "value"),
    prevent_initial_call=True,
)
def load_experiment_replay(n_clicks, selected_container, selected_experiment, selected_uuid):
    """Load feature vectors from selected experiment UUID and display in scatter plot"""
    if not n_clicks or not selected_container or not selected_experiment or not selected_uuid:
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
        "uuid": selected_uuid,
        "container": selected_container,
        "experiment_name": selected_experiment
    }
    
    try:
        # Connect to Tiled and load feature vectors
        logger.info(f"Loading experiment UUID: {selected_uuid} from container: {selected_container}, experiment: {selected_experiment}")
        
        # CHANGED: Use the new function to get the DataFrame directly
        df = get_experiment_dataframe(selected_container, selected_experiment, selected_uuid)
        
        if df is None or df.empty:
            default_marks = {i: str(i) for i in range(0, 101, 20)}
            return no_update, data_project_dict, replay_buffer, [0, 100], 100, default_marks
        
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        
        # Extract feature vectors from DataFrame
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            default_marks = {i: str(i) for i in range(0, 101, 20)}
            return no_update, data_project_dict, replay_buffer, [0, 100], 100, default_marks
            
        # Create numpy array with feature vectors
        feature_vectors = df[feature_cols].values
        
        # Log sample of feature vectors (first 3 rows)
        sample_size = min(3, feature_vectors.shape[0])
        logger.info(f"Sample feature vectors (first {sample_size} rows):")
        for i, sample in enumerate(feature_vectors[:sample_size]):
            logger.info(f"Row {i}: {sample}")
        
        # Extract tiled_urls for reference
        tiled_urls = df['tiled_url'].tolist() if 'tiled_url' in df.columns else []
        
        # Check if this is the special "feature_vectors" case
        is_feature_vectors_special_case = (selected_uuid == "feature_vectors")
        logger.info(f"Is feature_vectors special case: {is_feature_vectors_special_case}")
        
        # Extract slice indices from URLs
        live_indices = []
        if tiled_urls:
            import re
            
            if is_feature_vectors_special_case:
                # Special case: Each image is in its own container
                # live_indices should be sequential [0, 1, 2, ...] because each scatter point
                # maps to a dataset, and each dataset has only 1 image at index 0
                logger.info("Processing feature_vectors special case - using sequential indices")
                live_indices = list(range(len(df)))
                logger.info(f"Set live_indices to sequential: {live_indices[:5]}...")
            else:
                # Normal case: Extract slice indices from URLs
                # Initialize live_indices as a list of the same length as feature_vectors
                live_indices = [-1] * len(feature_vectors)
                
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
            "uuid": selected_uuid,
            "container": selected_container,
            "experiment_name": selected_experiment,
            "is_feature_vectors_special_case": is_feature_vectors_special_case  # Add flag
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
                
                if is_feature_vectors_special_case:
                    # Special case: Each image is in its own container
                    # URL example: /api/v1/metadata/Nafion_p25_30_run_1/local_images_1758157540_0170
                    # We need to create one dataset per image
                    
                    logger.info("Processing feature_vectors special case - creating datasets")
                    
                    datasets = []
                    for i, url in enumerate(tiled_urls):
                        # Extract the container path from each URL
                        parsed_url = urlparse(url)
                        url_path = parsed_url.path
                        
                        # Remove /api/v1/metadata/ or /api/v1/array/full/ prefix
                        if '/api/v1/metadata/' in url_path:
                            container_uri = url_path.split('/api/v1/metadata/')[-1]
                        elif '/api/v1/array/full/' in url_path:
                            container_uri = url_path.split('/api/v1/array/full/')[-1]
                        else:
                            # Fallback: try to extract path after /api/v1/
                            parts = url_path.split('/api/v1/')
                            if len(parts) > 1:
                                container_uri = parts[1]
                                # Remove metadata/ or array/full/ prefix if present
                                if container_uri.startswith('metadata/'):
                                    container_uri = container_uri.replace('metadata/', '', 1)
                                elif container_uri.startswith('array/full/'):
                                    container_uri = container_uri.replace('array/full/', '', 1)
                            else:
                                logger.warning(f"Could not parse URL: {url}")
                                continue
                        
                        # Add dataset for this specific image container
                        datasets.append({
                            "uri": container_uri,
                            "cumulative_data_count": i + 1
                        })
                        
                        # Log first few for debugging
                        if i < 3:
                            logger.info(f"Dataset {i}: uri={container_uri}")
                    
                    logger.info(f"Created {len(datasets)} datasets for special case")
                    
                    # Create data project with multiple datasets
                    data_project_dict = {
                        "root_uri": f"{base_uri}/api/v1/metadata",
                        "data_type": "tiled",
                        "datasets": datasets,
                        "project_id": f"replay_{selected_uuid}",
                        "replay_mode": True,
                        "is_feature_vectors_special_case": True  # Add flag
                    }
                else:
                    # Normal case: single dataset with slice-based access
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
                            "replay_mode": True
                        }
                    else:
                        # Fallback if parsing fails - use the initialized data_project_dict
                        pass
            except Exception as e:
                logger.warning(f"Error parsing URL for data project: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback to the initialized data_project_dict
                pass
        else:
            # No URLs available - use the initialized data_project_dict
            pass
        
        # Set total points as the data length
        total_points = len(feature_vectors)
        
        # Create integer range for slider
        initial_range = [0, int(total_points)]
        
        # Create integer marks for slider
        mark_interval = max(1, total_points // 5)  # Create ~5 marks
        marks = {i: str(i) for i in range(0, total_points+1, mark_interval)}
        
        # Return all outputs with data length as max and showing all data initially
        return scatter_fig, data_project_dict, replay_buffer, initial_range, total_points, marks
            
    except Exception as e:
        logger.error(f"Error loading experiment: {e}")
        logger.error(traceback.format_exc())
        default_marks = {i: str(i) for i in range(0, 101, 20)}
        return no_update, data_project_dict, replay_buffer, [0, 100], 100, default_marks