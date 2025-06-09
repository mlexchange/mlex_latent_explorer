import json
import logging
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from dash import (Input, Output, Patch, State, callback, callback_context,
                  no_update)
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from src.callbacks.execute import redis_client
from src.utils.mlflow_utils import MLflowClient
from src.utils.plot_utils import (generate_notification, generate_scatter_data,
                                  plot_empty_heatmap, plot_empty_scatter)

logger = logging.getLogger("lse.live_mode")
mlflow_client = MLflowClient()

@callback(
    Output("live-model-dialog", "is_open"),
    Output("live-autoencoder-dropdown", "options"),
    Output("live-dimred-dropdown", "options"),
    Output("live-autoencoder-dropdown", "value"),
    Output("live-dimred-dropdown", "value"),
    Input("go-live", "n_clicks"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def show_model_selection_dialog(n_clicks, last_selected_models):
    if n_clicks is not None and n_clicks % 2 == 1:
        # Get model options filtered by type
        autoencoder_options = mlflow_client.get_mlflow_models_live(model_type=None)
        dimred_options = mlflow_client.get_mlflow_models_live(model_type=None)
        
        # Set default values from previous selection if available
        autoencoder_default = None
        dimred_default = None
        
        if last_selected_models is not None:
            autoencoder_default = last_selected_models.get("autoencoder")
            dimred_default = last_selected_models.get("dimred")
       
        return True, autoencoder_options, dimred_options, autoencoder_default, dimred_default
    return False, [], [], None, None


@callback(
    Output("live-model-dialog", "is_open", allow_duplicate=True),
    Output("selected-live-models", "data"),
    Output("data-selection-controls", "style"),
    Output("dimension-reduction-controls", "style"),
    Output("clustering-controls", "style"),
    Output("data-overview-card", "style"),
    Output("live-mode-models", "style"),
    Output("live-mode-autoencoder-dropdown", "options"),
    Output("live-mode-autoencoder-dropdown", "value"),
    Output("live-mode-dimred-dropdown", "options"),
    Output("live-mode-dimred-dropdown", "value"),
    Output("sidebar", "active_item"),
    Output("image-card", "style"),
    Output("scatter", "style"),
    Output("heatmap", "style"),
    Output("go-live", "style"),
    Output("tooltip-go-live", "children"),
    Output("pause-button", "style"),
    Output("scatter", "figure", allow_duplicate=True),
    Output("heatmap", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Output("buffer", "data", allow_duplicate=True),
    Output("live-indices", "data", allow_duplicate=True),
    Output("model-loading-spinner", "style", allow_duplicate=True),  # Add spinner output
    Output("in-model-transition", "data", allow_duplicate=True),  # Add transition state output
    Input("live-model-continue", "n_clicks"),
    State("live-autoencoder-dropdown", "value"),
    State("live-dimred-dropdown", "value"),
    State("live-autoencoder-dropdown", "options"),
    State("live-dimred-dropdown", "options"),
    prevent_initial_call=True,
)
def handle_model_continue(continue_clicks, selected_autoencoder, selected_dimred, 
                          autoencoder_options, dimred_options):
    if continue_clicks:
        # Close dialog and save selected models
        selected_models = {"autoencoder": selected_autoencoder, "dimred": selected_dimred}
        
        # Show loading spinner and set transition state to True
        spinner_style = {
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0, 0, 0, 0.7)",
            "zIndex": 9998,
            "display": "block"  # Show the spinner
        }
        
        # Transition to live mode UI
        return (
            False,  # Close dialog
            selected_models,  # Save selected models
            {"display": "none"},  # Hide data selection controls
            {"display": "none"},  # Hide dimension reduction controls
            {"display": "none"},  # Hide clustering controls
            {"display": "none"},  # Hide data overview card
            {},  # Show live mode models section
            autoencoder_options,  # Copy options from dialog to sidebar
            selected_autoencoder,  # Set selected autoencoder
            dimred_options,  # Copy options from dialog to sidebar
            selected_dimred,  # Set selected dimension reduction model
            ["item-1"],  # Active sidebar item
            {"width": "100%", "height": "85vh"},  # Image card style
            {"height": "65vh"},  # Scatter style
            {"height": "65vh"},  # Heatmap style
            {  # Go-live button style
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "white",
                "background-color": "#D57800",
                "border": "0px",
            },
            "Go to Offline Mode",  # Go-live tooltip
            {  # Pause button style
                "display": "flex",
                "font-size": "1.5rem",
                "padding": "5px",
            },
            plot_empty_scatter(),  # Clear scatter when continuing
            plot_empty_heatmap(),  # Clear heatmap when continuing
            "Number of images selected: 0",  # Reset stats text
            {},  # Clear buffer
            [],  # Clear indices
            spinner_style,  # Show the loading spinner
            True,  # Set transition state to True
        )
    raise PreventUpdate


@callback(
    Output("live-model-dialog", "is_open", allow_duplicate=True),
    Output("go-live", "n_clicks"),
    Output("live-mode-canceled", "data"),
    Input("live-model-cancel", "n_clicks"),
    State("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def handle_model_cancel(cancel_clicks, go_live_clicks):
    """
    Handle the cancel button click in the model selection dialog.
    Return to the state before "Go to Live Mode" was clicked.
    """
    if cancel_clicks and go_live_clicks is not None and go_live_clicks % 2 == 1:
        # Set n_clicks to an even number (as if we're in offline mode)
        # Also set a flag to indicate cancellation occurred
        return False, go_live_clicks - 1, True
    raise PreventUpdate


@callback(
    Output("live-model-continue", "disabled"),
    Input("live-autoencoder-dropdown", "value"),
    Input("live-dimred-dropdown", "value"),
)
def toggle_continue_button(selected_autoencoder, selected_dimred):
    return selected_autoencoder is None or selected_dimred is None


@callback(
    Output("show-clusters", "value", allow_duplicate=True),
    Output("show-feature-vectors", "value", allow_duplicate=True),
    Output("data-selection-controls", "style", allow_duplicate=True),
    Output("dimension-reduction-controls", "style", allow_duplicate=True),
    Output("clustering-controls", "style", allow_duplicate=True),
    Output("data-overview-card", "style", allow_duplicate=True),
    Output("live-mode-models", "style", allow_duplicate=True),
    Output("sidebar", "active_item", allow_duplicate=True),
    Output("image-card", "style", allow_duplicate=True),
    Output("scatter", "style", allow_duplicate=True),
    Output("heatmap", "style", allow_duplicate=True),
    Output("go-live", "style", allow_duplicate=True),
    Output("tooltip-go-live", "children", allow_duplicate=True),
    Output("pause-button", "style", allow_duplicate=True),
    Output("live-indices", "data", allow_duplicate=True),
    Output("live-mode-canceled", "data", allow_duplicate=True),
    Output("selected-live-models", "data", allow_duplicate=True),
    Input("go-live", "n_clicks"),
    State("selected-live-models", "data"),
    State("live-mode-canceled", "data"),
    prevent_initial_call=True,
)
def toggle_controls(n_clicks, selected_models, mode_canceled):
    """
    Toggle the visibility of the sidebar, data overview card, image card, and go-live button
    """
    
    # If cancel was clicked, just reset the cancel flag but don't make other changes
    if n_clicks is not None and n_clicks % 2 == 0 and mode_canceled:
        # Return no_update for all outputs except the last one (cancel flag)
        return (
            no_update,  # show-clusters value
            no_update,  # show-feature-vectors value
            no_update,  # data-selection-controls style
            no_update,  # dimension-reduction-controls style
            no_update,  # clustering-controls style
            no_update,  # data-overview-card style
            no_update,  # live-mode-models style
            no_update,  # sidebar active_item
            no_update,  # image-card style
            no_update,  # scatter style
            no_update,  # heatmap style
            no_update,  # go-live style
            no_update,  # tooltip-go-live children
            no_update,  # pause-button style
            no_update,  # live-indices data
            False,       # Reset the cancel flag
            None,
        )
    
    # Check if continue was already clicked (selected_models is not None)
    if n_clicks is not None and n_clicks % 2 == 0:
        # Going back to offline mode - send Redis messages to reset models
        if redis_client is not None and selected_models is not None:
            try:
                # Reset autoencoder model to None
                redis_client.set("selected_mlflow_model", "")
                
                # Reset dimension reduction model to None  
                redis_client.set("selected_dim_reduction_model", "")
                
                # Also publish update notifications
                message = {
                    "model_type": "autoencoder",
                    "model_name": None,
                    "timestamp": time.time()
                }
                redis_client.publish("model_updates", json.dumps(message))
                
                message = {
                    "model_type": "dimred",
                    "model_name": None,
                    "timestamp": time.time()
                }
                redis_client.publish("model_updates", json.dumps(message))
                
                logger.info("Published model reset messages to Redis")
            except Exception as e:
                logger.error(f"Error publishing model reset messages to Redis: {e}")
        
        # Going back to offline mode
        return (
            False,
            False,
            {},  # Show data selection controls
            {},  # Show dimension reduction controls
            {},  # Show clustering controls
            {},  # Show data overview card
            {"display": "none"},  # Hide live mode models section
            no_update,  # Don't change active sidebar item
            {"height": "64vh"},  # Normal image card height
            {"height": "46vh"},  # Normal scatter height
            {"height": "46vh"},  # Normal heatmap height
            {  # Regular go-live button style
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "#00313C",
                "background-color": "white",
                "border": "0px",
            },
            "Go to Live Mode",  # Change tooltip text
            {  # Hide pause button
                "display": "none",
            },
            [],  # Clear live indices
            False,  # Reset canceled flag
            None,  # Reset selected_models to None
        )
    
    # First click or other odd clicks - going to live mode        
    if n_clicks is not None and n_clicks % 2 == 1 and selected_models is not None:
        return (
            False,
            False,
            {"display": "none"},  # Hide data selection controls
            {"display": "none"},  # Hide dimension reduction controls
            {"display": "none"},  # Hide clustering controls
            {"display": "none"},  # Hide data overview card
            {},  # Show live mode models section
            ["item-1"],  # Set first item as active
            {"width": "100%", "height": "85vh"},  # Full width and taller image card
            {"height": "65vh"},  # Taller scatter
            {"height": "65vh"},  # Taller heatmap
            {  # Orange go-live button
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "white",
                "background-color": "#D57800",
                "border": "0px",
            },
            "Go to Offline Mode",  # Change tooltip text
            {  # Show pause button
                "display": "flex",
                "font-size": "1.5rem",
                "padding": "5px",
            },
            [],  # Initialize empty live indices
            False,  # Reset canceled flag
            selected_models,  # Keep selected_models unchanged
        )
    
    raise PreventUpdate


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Output("heatmap", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Output("buffer", "data", allow_duplicate=True),
    Input("go-live", "n_clicks"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
    
)
def reset_panels_on_exit_live_mode(n_clicks,selected_models):
    """
    Reset all the visualization panels when switching from live to offline mode
    """
    # Only reset when exiting live mode (even clicks)
    if n_clicks is not None and n_clicks % 2 == 0 and selected_models is not None:
        return plot_empty_scatter(), plot_empty_heatmap(), "Number of images selected: 0", {}
        
    # Don't reset panels when just opening the dialog
    raise PreventUpdate


@callback(
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Input("go-live", "n_clicks"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def update_data_project_dict(n_clicks, selected_models):
    """
    Update the data project dictionary when toggling modes
    """
    if n_clicks is not None:
        # Case 1: Entering live mode with models selected
        if n_clicks % 2 == 1 and selected_models is not None:
            return {
                "root_uri": "",
                "data_type": "tiled",
                "datasets": [],
                "project_id": None,
                "live_models": selected_models
            }
        # Case 2: After cancel (even clicks but no models selected)
        elif n_clicks % 2 == 0 and n_clicks > 0 and selected_models is None:
            # Stay in offline mode if cancel was clicked (no model selected)
            raise PreventUpdate
        # Case 3: Exiting live mode (even clicks with models selected)
        elif n_clicks % 2 == 0 and n_clicks > 0:
            # Exiting live mode - completely reset to empty project
            return {
                "root_uri": "",
                "data_type": "tiled",
                "datasets": [],  # Empty datasets list is key
                "project_id": None
            }
    raise PreventUpdate


@callback(
    Output("selected-live-models", "data", allow_duplicate=True),
    Output({"base_id": "file-manager", "name": "data-project-dict"}, "data", allow_duplicate=True),
    Output("update-live-models-button", "color"),
    Output("update-live-models-button", "children"),
    Output("scatter", "figure", allow_duplicate=True),
    Output("heatmap", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Output("live-indices", "data", allow_duplicate=True),
    Output("model-loading-spinner", "style", allow_duplicate=True),  # Add spinner output
    Output("in-model-transition", "data", allow_duplicate=True),  # Add transition state output
    Input("update-live-models-button", "n_clicks"),
    State("live-mode-autoencoder-dropdown", "value"),
    State("live-mode-dimred-dropdown", "value"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def update_live_models(n_clicks, autoencoder_model, dimred_model, data_project_dict):
    """
    Update the selected models when the update button is clicked
    """
    if n_clicks is None:
        raise PreventUpdate
        
    if autoencoder_model is None or dimred_model is None:
        # Show error notification
        return no_update, no_update, "danger", "Invalid Selection", no_update, no_update, no_update, no_update, no_update, no_update
    
    # Update the selected models
    selected_models = {"autoencoder": autoencoder_model, "dimred": dimred_model}
    
    # Update data project dict with new models
    data_project_dict["live_models"] = selected_models
    
    # Create empty figure to reset the scatter plot
    empty_figure = plot_empty_scatter()
    
    # Create empty figure to reset the heatmap
    empty_heatmap = plot_empty_heatmap()
    
    # Reset stats text
    stats_text = "Number of images selected: 0"
    
    # Reset live indices
    empty_indices = []
    
    # Show loading spinner
    spinner_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100%",
        "height": "100%",
        "backgroundColor": "rgba(0, 0, 0, 0.7)",
        "zIndex": 9998,
        "display": "block"  # Show the spinner
    }
    
    return selected_models, data_project_dict, "secondary", "Updated", empty_figure, empty_heatmap, stats_text, empty_indices, spinner_style, True

@callback(
    Output("update-live-models-button", "color", allow_duplicate=True),
    Output("update-live-models-button", "children", allow_duplicate=True),
    Input("live-mode-autoencoder-dropdown", "value"),
    Input("live-mode-dimred-dropdown", "value"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def reset_update_button(autoencoder_model, dimred_model, selected_models):
    """
    Reset the update button color when dropdowns change
    """
    # If no models are selected yet or either dropdown is empty
    if selected_models is None or autoencoder_model is None or dimred_model is None:
        return "primary", "Update Models"
        
    # Check if the current selection is different from what's already selected
    if (autoencoder_model != selected_models.get("autoencoder") or 
        dimred_model != selected_models.get("dimred")):
        return "primary", "Update Models"
    
    # If the selection is the same as current, keep the "Updated" state
    return "secondary", "Updated"


@callback(
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Output("live-indices", "data", allow_duplicate=True),
    Input("ws-live", "message"),
    State("selected-live-models", "data"),
    State("go-live", "n_clicks"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("live-indices", "data"),
    State("in-model-transition", "data"),
    prevent_initial_call=True,
)
def live_update_data_project_dict(message, selected_models, n_clicks, data_project_dict, live_indices, in_transition):
    """
    Update data project dict with the data uri from the live experiment
    """
    if n_clicks is not None and n_clicks % 2 == 1 and selected_models is not None:
        try:
            message_data = json.loads(message["data"])
            
            # Check if model names match currently selected models
            autoencoder_model = message_data.get("autoencoder_model")
            dimred_model = message_data.get("dimred_model")
            
            if selected_models is not None:
                current_autoencoder = selected_models.get("autoencoder")
                current_dimred = selected_models.get("dimred")
                
                # Skip messages from different models
                if (autoencoder_model and autoencoder_model != current_autoencoder) or \
                   (dimred_model and dimred_model != current_dimred):
                    logging.info(f"Skipping data project update from different models: got {autoencoder_model}/{dimred_model}, expected {current_autoencoder}/{current_dimred}")
                    # Return a reset data_project_dict with empty datasets
                    updated_dict = data_project_dict.copy()
                    updated_dict["datasets"] = []
                    return updated_dict, []
            
            # use tiled_url to be compatible with LatentSpaceEvent
            tiled_uri = message_data.get("tiled_url", "")
            if not tiled_uri:
                return data_project_dict, live_indices
                
            split_uri = urlsplit(tiled_uri)
            path_parts = split_uri.path.rsplit("/", 1)
            root_uri = urlunsplit(
                (split_uri.scheme, split_uri.netloc, path_parts[0] + "/", "", "")
            )
            uri = path_parts[1]

            index = message_data.get("index", 0)

            # Make sure live_indices is a list
            if live_indices is None:
                live_indices = []
                
            live_indices.append(index)

            # Update cum_size according to the received index
            cum_size = max(live_indices) + 1

            # Update the data project dict
            if data_project_dict.get("root_uri") != root_uri:
                data_project_dict["root_uri"] = root_uri
                data_project_dict["data_type"] = "tiled"
                if "live_models" not in data_project_dict:
                    data_project_dict["live_models"] = selected_models

            if len(data_project_dict.get("datasets", [])) == 0:
                data_project_dict["datasets"] = [
                    {
                        "uri": uri,
                        "cumulative_data_count": cum_size,
                    }
                ]
            else:
                data_project_dict["datasets"][0] = {
                    "uri": uri,
                    "cumulative_data_count": cum_size,
                }
                
            return data_project_dict, live_indices
        except Exception as e:
            logging.error(f"Error in live_update_data_project_dict: {e}")
            return data_project_dict, live_indices
    
    return data_project_dict, live_indices


@callback(
    Output("buffer", "data"),
    Output("scatter", "figure", allow_duplicate=True),
    Output("model-loading-spinner", "style", allow_duplicate=True),  # Add spinner output
    Output("in-model-transition", "data", allow_duplicate=True),  # Add transition state output
    Input("ws-live", "message"),
    State("scatter", "figure"),
    State("pause-button", "n_clicks"),
    State("buffer", "data"),
    State("selected-live-models", "data"),
    State("go-live", "n_clicks"),  # Add to check live mode
    State("in-model-transition", "data"),  # Add transition state
    prevent_initial_call=True,
)
def set_live_latent_vectors(message, current_figure, pause_n_clicks, buffer_data, selected_models, go_live_n_clicks, in_transition):
    # Only process if in live mode
    if go_live_n_clicks is None or go_live_n_clicks % 2 == 0 or selected_models is None:
        raise PreventUpdate
        
    try:
        data = json.loads(message["data"])
        logging.debug(f"Received data: {data}")
        
        # Get model names from message
        autoencoder_model = data.get("autoencoder_model")
        dimred_model = data.get("dimred_model")
        
        # Check if model names match currently selected models
        if selected_models is not None:
            current_autoencoder = selected_models.get("autoencoder")
            current_dimred = selected_models.get("dimred")
            
            # Skip messages from different models
            if (autoencoder_model and autoencoder_model != current_autoencoder) or \
               (dimred_model and dimred_model != current_dimred):
                logging.info(f"Skipping message from different models: got {autoencoder_model}/{dimred_model}, expected {current_autoencoder}/{current_dimred}")
                # Return an empty buffer and empty figure when models don't match
                empty_fig = plot_empty_scatter()
                return {}, empty_fig, no_update, no_update
                
            # If we're in transition and got a matching model message, hide the spinner
            if in_transition:
                spinner_style = {
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100%",
                    "height": "100%",
                    "backgroundColor": "rgba(0, 0, 0, 0.7)",
                    "zIndex": 9998,
                    "display": "none"  # Hide the spinner
                }
                transition_state = False
            else:
                # Keep the current spinner state
                spinner_style = no_update
                transition_state = no_update
        
        # Get feature_vector, handling it consistently
        feature_vector = data.get("feature_vector")
        if feature_vector is None:
            return buffer_data, no_update, spinner_style, transition_state
            
        latent_vectors = np.array(feature_vector, dtype=float)

        latent_vectors = (
            latent_vectors.reshape(1, -1) if latent_vectors.ndim == 1 else latent_vectors
        )
        n_components = latent_vectors.shape[1]

        # If the pause button is clicked, buffer the latent vectors
        if pause_n_clicks is not None and pause_n_clicks % 2 == 1:
            if not buffer_data:
                # First time buffering
                buffer_data["num_components"] = n_components
                buffer_data["latent_vectors"] = latent_vectors
                return buffer_data, no_update, spinner_style, transition_state
            else:
                # Append to existing buffer
                buffer_data["latent_vectors"] = np.vstack(
                    (buffer_data["latent_vectors"], latent_vectors)
                )
                return buffer_data, no_update, spinner_style, transition_state

        # If figure is empty (no customdata yet), return a new figure.
        if not current_figure["data"] or len(current_figure["data"]) == 0 or "customdata" not in current_figure["data"][0]:
            new_fig = generate_scatter_data(latent_vectors, n_components)
            return {}, new_fig, spinner_style, transition_state

        # Otherwise, do a partial update of the existing figure using Patch
        try:
            # Use the Patch approach, but with try-except to handle potential errors
            figure_patch = Patch()
    
            # Build lists from the newly arriving latent vectors
            xs_new = latent_vectors[:, 0].tolist()
            ys_new = latent_vectors[:, 1].tolist()
            customdata_new = [[0]] * len(xs_new)  # or adapt to your custom data usage
    
            # Check if the arrays exist before extending them
            if "x" in current_figure["data"][0] and current_figure["data"][0]["x"] is not None:
                figure_patch["data"][0]["x"] = current_figure["data"][0]["x"] + xs_new
            else:
                figure_patch["data"][0]["x"] = xs_new
                
            if "y" in current_figure["data"][0] and current_figure["data"][0]["y"] is not None:
                figure_patch["data"][0]["y"] = current_figure["data"][0]["y"] + ys_new
            else:
                figure_patch["data"][0]["y"] = ys_new
                
            if "customdata" in current_figure["data"][0] and current_figure["data"][0]["customdata"] is not None:
                figure_patch["data"][0]["customdata"] = current_figure["data"][0]["customdata"] + customdata_new
            else:
                figure_patch["data"][0]["customdata"] = customdata_new
                
            return {}, figure_patch, spinner_style, transition_state
        except Exception as e:
            # If patching fails, create a new figure instead
            logging.warning(f"Error patching figure: {e}, creating new figure")
            new_fig = generate_scatter_data(latent_vectors, n_components)
            return {}, new_fig, spinner_style, transition_state

    except Exception as e:
        logging.error(f"Error in set_live_latent_vectors: {e}")
        return buffer_data, no_update, no_update, no_update

@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("pause-button", "n_clicks"),
    State("buffer", "data"),
    State("scatter", "figure"),
    State("go-live", "n_clicks"),  # Add to check live mode
    prevent_initial_call=True,
)
def set_buffered_latent_vectors(n_clicks, buffer_data, current_figure, go_live_n_clicks):
    # Only process if in live mode
    if go_live_n_clicks is None or go_live_n_clicks % 2 == 0:
        raise PreventUpdate
        
    if n_clicks is not None and n_clicks % 2 == 1 or buffer_data == {}:
        raise PreventUpdate

    try:
        num_components = buffer_data.get("num_components")
        latent_vectors = buffer_data.get("latent_vectors")
        
        if num_components is None or latent_vectors is None:
            raise PreventUpdate

        # If the scatter plot is empty, generate new scatter data
        if "customdata" not in current_figure["data"][0]:
            return generate_scatter_data(latent_vectors, num_components)

        # If the scatter plot is not empty, append the new latent vectors
        for latent_vector in latent_vectors:
            current_figure["data"][0]["customdata"].append([0])
            current_figure["data"][0]["x"].append(float(latent_vector[0]))
            current_figure["data"][0]["y"].append(float(latent_vector[1]))
        return current_figure
    except Exception as e:
        logging.error(f"Error in set_buffered_latent_vectors: {e}")
        return no_update


@callback(
    Output("pause-button", "children", allow_duplicate=True),
    Output("tooltip-pause-button", "children", allow_duplicate=True),
    Input("pause-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_pause_button(n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        icon = "pajamas:play"
        children = "Continue live display"
    else:
        icon = "lucide:circle-pause"
        children = "Pause live display"
    return DashIconify(icon=icon, style={"padding": "0px"}), children


@callback(
    Output("pause-button", "children", allow_duplicate=True),
    Output("tooltip-pause-button", "children", allow_duplicate=True),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_pause_button_go_live(go_live_n_clicks):
    return (
        DashIconify(icon="lucide:circle-pause", style={"padding": "0px"}),
        "Pause live display",
    )
