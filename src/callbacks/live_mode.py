import logging
import os
import time

import plotly.graph_objects as go
import numpy as np
from dash import (
    ClientsideFunction,
    Input,
    Output,
    Patch,
    State,
    callback,
    clientside_callback,
    no_update,
)
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from src.arroyo_reduction.redis_model_store import (
    RedisModelStore,
)  # Import the RedisModelStore class
from src.utils.mlflow_utils import MLflowClient
from src.utils.plot_utils import (
    generate_scatter_data,
    plot_empty_heatmap,
    plot_empty_scatter,
)

# Initialize Redis model store instead of direct Redis client
REDIS_HOST = os.getenv("REDIS_HOST", "kvrocks")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6666))
redis_model_store = RedisModelStore(host=REDIS_HOST, port=REDIS_PORT)

logger = logging.getLogger("lse.live_mode")
mlflow_client = MLflowClient()


@callback(
    Output("live-model-dialog", "is_open"),
    Output("live-autoencoder-dropdown", "options"),
    Output("live-dimred-dropdown", "options"),
    Output("live-autoencoder-dropdown", "value"),  # Add this output
    Output("live-dimred-dropdown", "value"),       # Add this output
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def show_model_selection_dialog(n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        # Show dialog immediately with loading placeholders
        loading_options = [{"label": "Loading models...", "value": None, "disabled": True}]
        
        # Just show the dialog with loading placeholders and set values to None
        # Setting value to None will show the placeholder when options are loaded
        return True, loading_options, loading_options, None, None
    
    return False, [], [], None, None

@callback(
    Output("live-autoencoder-dropdown", "options", allow_duplicate=True),
    Output("live-dimred-dropdown", "options", allow_duplicate=True),
    Input("live-model-dialog", "is_open"),
    prevent_initial_call=True,
)
def update_model_dropdowns(dialog_open):
    """
    Update the dropdowns with actual model options after a slight delay
    """
    if not dialog_open:
        raise PreventUpdate
    
    
    try:
        # Load actual model options
        autoencoder_options = mlflow_client.get_mlflow_models(
            livemode=True, model_type="autoencoder"
        )
        
        dimred_options = mlflow_client.get_mlflow_models(
            livemode=True, model_type="dimension_reduction"
        )
        
        # Return the actual options to replace the loading placeholders
        return autoencoder_options, dimred_options
        
    except Exception as e:
        logger.error(f"Error loading model options: {e}")
        error_option = [{"label": "Error loading models", "value": None}]
        return error_option, error_option




@callback(
    Output("live-model-dialog", "is_open", allow_duplicate=True),
    Output("selected-live-models", "data"),
    Output("data-selection-controls", "style"),
    Output("dimension-reduction-controls", "style"),
    Output("clustering-controls", "style"),
    Output("data-overview-card", "style"),
    Output("live-mode-models", "style"),
    Output("live-mode-autoencoder-dropdown", "options", allow_duplicate=True),
    Output("live-mode-autoencoder-dropdown", "value", allow_duplicate=True),
    Output("live-mode-autoencoder-version-dropdown", "options", allow_duplicate=True),
    Output("live-mode-autoencoder-version-dropdown", "value", allow_duplicate=True),
    Output("live-mode-autoencoder-version-dropdown", "disabled", allow_duplicate=True),
    Output("live-mode-dimred-dropdown", "options", allow_duplicate=True),
    Output("live-mode-dimred-dropdown", "value", allow_duplicate=True),
    Output("live-mode-dimred-version-dropdown", "options", allow_duplicate=True),
    Output("live-mode-dimred-version-dropdown", "value", allow_duplicate=True),
    Output("live-mode-dimred-version-dropdown", "disabled", allow_duplicate=True),
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
    Output("model-loading-spinner", "style", allow_duplicate=True),
    Output("in-model-transition", "data", allow_duplicate=True),
    Output("experiment-replay-controls", "style"),
    Output("live-experiment-name-display", "style", allow_duplicate=True),  # NEW: ADD THIS LINE
    Output("live-experiment-name-text", "children", allow_duplicate=True),  # NEW: ADD THIS LINE
    Input("live-model-continue", "n_clicks"),
    State("live-experiment-name-input", "value"),  # NEW: ADD THIS LINE
    State("live-autoencoder-dropdown", "value"),
    State("live-autoencoder-version-dropdown", "value"),
    State("live-dimred-dropdown", "value"),
    State("live-dimred-version-dropdown", "value"),
    State("live-autoencoder-dropdown", "options"),
    State("live-dimred-dropdown", "options"),
    prevent_initial_call=True,
)
def handle_model_continue(
    continue_clicks,
    experiment_name,  # NEW: ADD THIS LINE
    selected_autoencoder,
    selected_autoencoder_version,
    selected_dimred,
    selected_dimred_version,
    autoencoder_options,
    dimred_options,
):
    """Handle the Continue button click with version selection"""
    if not continue_clicks:
        raise PreventUpdate

    # NEW: ADD THIS VALIDATION BLOCK - START
    if not experiment_name or not experiment_name.strip():
        logger.warning("No experiment name provided")
        raise PreventUpdate
    # NEW: ADD THIS VALIDATION BLOCK - END

    # Check model compatibility before proceeding
    if not mlflow_client.check_model_compatibility(
        selected_autoencoder, selected_dimred
    ):
        logger.warning(
            f"Incompatible models selected: {selected_autoencoder} and {selected_dimred}"
        )
        return no_update

    # Store models with versions in Redis
    try:
        # Create model identifiers with versions
        autoencoder_id = f"{selected_autoencoder}:{selected_autoencoder_version}"
        dimred_id = f"{selected_dimred}:{selected_dimred_version}"
        
        logger.info(f"Storing autoencoder model from dialog: {autoencoder_id}")
        redis_model_store.store_autoencoder_model(autoencoder_id)

        logger.info(f"Storing dimension reduction model from dialog: {dimred_id}")
        redis_model_store.store_dimred_model(dimred_id)
        
        # NEW: ADD THIS BLOCK - START
        logger.info(f"Storing experiment name from dialog: {experiment_name.strip()}")
        redis_model_store.store_experiment_name(experiment_name.strip())
        # NEW: ADD THIS BLOCK - END
    except Exception as e:
        logger.error(f"Error storing models in Redis: {e}")
        return no_update

    selected_models = {
        "autoencoder": selected_autoencoder,
        "autoencoder_version": selected_autoencoder_version,
        "dimred": selected_dimred,
        "dimred_version": selected_dimred_version,
        "experiment_name": experiment_name.strip(),  # NEW: ADD THIS LINE
    }

    # Get version options for sidebar
    autoencoder_versions = mlflow_client.get_model_versions(selected_autoencoder)
    dimred_versions = mlflow_client.get_model_versions(selected_dimred)

    # Show loading spinner
    spinner_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100%",
        "height": "100%",
        "backgroundColor": "rgba(0, 0, 0, 0.7)",
        "zIndex": 9998,
        "display": "block",
    }

    return (
        False,  # Close dialog
        selected_models,  # Save selected models
        {"display": "none"},  # Hide data selection controls
        {"display": "none"},  # Hide dimension reduction controls
        {"display": "none"},  # Hide clustering controls
        {"display": "none"},  # Hide data overview card
        {},  # Show live mode models section
        autoencoder_options,  # Copy options to sidebar
        selected_autoencoder,  # Set selected autoencoder name
        autoencoder_versions,  # Set autoencoder version options
        selected_autoencoder_version,  # Set selected autoencoder version
        False,  # Enable autoencoder version dropdown
        dimred_options,  # Copy options to sidebar
        selected_dimred,  # Set selected dimred name
        dimred_versions,  # Set dimred version options
        selected_dimred_version,  # Set selected dimred version
        False,  # Enable dimred version dropdown
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
        plot_empty_scatter(),  # Clear scatter
        plot_empty_heatmap(),  # Clear heatmap
        "Number of images selected: 0",  # Reset stats
        {},  # Clear buffer
        [],  # Clear indices
        spinner_style,  # Show loading spinner
        True,  # Set transition state
        {"display": "none"},  # Hide experiment replay controls
        {"display": "block"},  # NEW: ADD THIS LINE - Show experiment name display
        experiment_name.strip(),  # NEW: ADD THIS LINE - Set experiment name text
    )

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


# Replace the existing toggle_continue_button callback in live_mode.py

@callback(
    Output("live-model-continue", "disabled"),
    Output("live-model-continue", "color", allow_duplicate=True),
    Output("live-model-continue", "children", allow_duplicate=True),
    Input("live-experiment-name-input", "value"),  # NEW: ADD THIS LINE
    Input("live-autoencoder-dropdown", "value"),
    Input("live-dimred-dropdown", "value"),
    Input("live-autoencoder-version-dropdown", "value"),
    Input("live-dimred-version-dropdown", "value"),
    prevent_initial_call=True,
)
def toggle_continue_button(experiment_name, selected_autoencoder, selected_dimred, auto_version, dimred_version):  # NEW: ADD experiment_name parameter
    """
    Disable the continue button if models or versions are not selected or are incompatible
    Also update the button color and text to indicate state
    """
    # NEW: ADD THIS CHECK BLOCK - START
    if not experiment_name or not experiment_name.strip():
        return True, "secondary", "Continue"
    # NEW: ADD THIS CHECK BLOCK - END
    
    # Check if any required field is not selected
    if (selected_autoencoder is None or selected_dimred is None or 
        auto_version is None or dimred_version is None):
        return True, "secondary", "Continue"  # Disabled with neutral color

    # Check if models are compatible
    is_compatible = mlflow_client.check_model_compatibility(
        selected_autoencoder, selected_dimred
    )

    if is_compatible:
        # Models are compatible - enable with primary color
        return False, "primary", "Continue"
    else:
        # Models are incompatible - disable with danger color
        return True, "danger", "Incompatible Models"

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
    Output("experiment-replay-controls", "style", allow_duplicate=True),  # ADDED
    Output("live-experiment-name-display", "style", allow_duplicate=True),  # NEW: ADD THIS LINE
    Output("live-experiment-name-text", "children", allow_duplicate=True),  # NEW: ADD THIS LINE
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
            False,  # Reset the cancel flag
            None,  # selected-live-models data
            no_update,  # experiment-replay-controls style - ADDED
            no_update,  # NEW: ADD THIS LINE - experiment-name-display style
            no_update,  # NEW: ADD THIS LINE - experiment-name-text
        )

    # Check if continue was already clicked (selected_models is not None)
    if n_clicks is not None and n_clicks % 2 == 0:
        # Going back to offline mode - send Redis messages to reset models
        if selected_models is not None:
            try:
                # Reset autoencoder model to empty string
                redis_model_store.store_autoencoder_model("")

                # Reset dimension reduction model to empty string
                redis_model_store.store_dimred_model("")

                # NEW: ADD THIS BLOCK - START
                redis_model_store.store_experiment_name("")
                # NEW: ADD THIS BLOCK - END

                logger.info("Published model reset messages to Redis")
            except Exception as e:
                logger.error(f"Error publishing model reset messages to Redis: {e}")

        # Going back to offline mode
        return (
            False,  # show-clusters value
            False,  # show-feature-vectors value
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
            {},  # Show experiment replay controls - ADDED
            {"display": "none"},  # NEW: ADD THIS LINE - Hide experiment name display
            "",  # NEW: ADD THIS LINE - Clear experiment name text
        )

    # First click or other odd clicks - going to live mode
    if n_clicks is not None and n_clicks % 2 == 1 and selected_models is not None:
        experiment_name = selected_models.get("experiment_name", "")  # NEW: ADD THIS LINE
        
        return (
            False,  # show-clusters value
            False,  # show-feature-vectors value
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
            {"display": "none"},  # Hide experiment replay controls - ADDED
            {"display": "block" if experiment_name else "none"},  # NEW: ADD THIS LINE - Show/hide experiment display
            experiment_name,  # NEW: ADD THIS LINE - Set experiment name text
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
def reset_panels_on_exit_live_mode(n_clicks, selected_models):
    """
    Reset all the visualization panels when switching from live to offline mode
    """
    # Only reset when exiting live mode (even clicks)
    if n_clicks is not None and n_clicks % 2 == 0 and selected_models is not None:
        return (
            plot_empty_scatter(),
            plot_empty_heatmap(),
            "Number of images selected: 0",
            {},
        )

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
                "live_models": selected_models,
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
                "data_type": "",
                "datasets": [],  # Empty datasets list is key
                "project_id": None,
            }
    raise PreventUpdate

@callback(
    Output("selected-live-models", "data", allow_duplicate=True),
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Output("update-live-models-button", "color"),
    Output("update-live-models-button", "children"),
    Output("scatter", "figure", allow_duplicate=True),
    Output("heatmap", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Output("live-indices", "data", allow_duplicate=True),
    Output("model-loading-spinner", "style", allow_duplicate=True),
    Output("in-model-transition", "data", allow_duplicate=True),
    Output("buffer", "data", allow_duplicate=True),
    Input("update-live-models-button", "n_clicks"),
    State("live-mode-autoencoder-dropdown", "value"),
    State("live-mode-autoencoder-version-dropdown", "value"),  # ADDED
    State("live-mode-dimred-dropdown", "value"),
    State("live-mode-dimred-version-dropdown", "value"),  # ADDED
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def update_live_models(n_clicks, autoencoder_model, autoencoder_version, 
                       dimred_model, dimred_version, data_project_dict):
    """
    Update the selected models when the sidebar update button is clicked.
    This function:
    1. Checks model compatibility
    2. Stores models with versions in Redis if compatible
    3. Updates data project dictionary
    4. Resets visualizations
    5. Shows loading spinner
    """
    if n_clicks is None:
        raise PreventUpdate

    # Check model compatibility (names only, since dimensions don't change between versions)
    if not mlflow_client.check_model_compatibility(autoencoder_model, dimred_model):
        logger.warning(
            f"Incompatible models selected: {autoencoder_model}, {dimred_model}"
        )
        return (
            no_update,  # selected-live-models.data
            no_update,  # data-project-dict.data
            "danger",  # update-live-models-button.color
            "Incompatible Models",  # update-live-models-button.children
            no_update,  # scatter.figure
            no_update,  # heatmap.figure
            no_update,  # stats-div.children
            no_update,  # live-indices.data
            no_update,  # model-loading-spinner.style
            no_update,  # in-model-transition.data
            no_update,  # buffer.data
        )

    # Create model identifiers with versions
    autoencoder_id = f"{autoencoder_model}:{autoencoder_version}"
    dimred_id = f"{dimred_model}:{dimred_version}"

    # Store models in Redis
    try:
        logger.info(f"Storing autoencoder model from sidebar: {autoencoder_id}")
        redis_model_store.store_autoencoder_model(autoencoder_id)

        logger.info(f"Storing dimension reduction model from sidebar: {dimred_id}")
        redis_model_store.store_dimred_model(dimred_id)
    except Exception as e:
        logger.error(f"Error storing models in Redis: {e}")
        return (
            no_update,
            no_update,
            "danger",
            "Update Fail",
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    # Update the selected models (include versions)
    selected_models = {
        "autoencoder": autoencoder_model,
        "autoencoder_version": autoencoder_version,
        "dimred": dimred_model,
        "dimred_version": dimred_version
    }

    # Update data project dict with new models
    data_project_dict = {
        **data_project_dict,  # Preserve all existing fields
        "root_uri": "",  # Reset root_uri
        "datasets": [],  # Reset datasets to empty list
        "live_models": selected_models,  # Update models
    }

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
        "display": "block",  # Show the spinner
    }

    # Return all 11 outputs
    return (
        selected_models,
        data_project_dict,
        "secondary",
        "Updated",
        empty_figure,
        empty_heatmap,
        stats_text,
        empty_indices,
        spinner_style,
        True,  # Set transition state to True
        [],  # Clear buffer data
    )


@callback(
    Output("update-live-models-button", "disabled", allow_duplicate=True),
    Output("update-live-models-button", "color", allow_duplicate=True),
    Output("update-live-models-button", "children", allow_duplicate=True),
    Input("live-mode-autoencoder-dropdown", "value"),
    Input("live-mode-autoencoder-version-dropdown", "value"),  # ADDED
    Input("live-mode-dimred-dropdown", "value"),
    Input("live-mode-dimred-version-dropdown", "value"),  # ADDED
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def reset_update_button(autoencoder_model, autoencoder_version, 
                        dimred_model, dimred_version, selected_models):
    """
    Reset the update button color when dropdowns change and check model compatibility.
    Now also checks if versions have changed.
    """
    # If no models are selected yet or any dropdown is empty
    if (selected_models is None or 
        autoencoder_model is None or dimred_model is None or
        autoencoder_version is None or dimred_version is None):
        return True, "secondary", "Update Models"  # Disabled, secondary color

    # Check if models are compatible (names only)
    is_compatible = mlflow_client.check_model_compatibility(
        autoencoder_model, dimred_model
    )

    if not is_compatible:
        # Models are incompatible - disable button with danger color
        return True, "danger", "Incompatible Models"

    # Check if EITHER name OR version changed
    models_changed = (
        autoencoder_model != selected_models.get("autoencoder") or
        autoencoder_version != selected_models.get("autoencoder_version") or
        dimred_model != selected_models.get("dimred") or
        dimred_version != selected_models.get("dimred_version")
    )

    if models_changed:
        # Models are compatible and changed - enable button with primary color
        return False, "primary", "Update Models"
    else:
        # Models are compatible but unchanged - show "Updated" state
        return True, "secondary", "Updated"


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("buffer-debounce", "n_intervals"),
    State("scatter", "figure"),
    State("pause-button", "n_clicks"),
    State("buffer", "data"),
    prevent_initial_call=True,
)
def set_live_latent_vectors(n_intervals, current_figure, pause_n_clicks, buffer_data):
    # Only process if in live mode
    if pause_n_clicks is not None and pause_n_clicks % 2 == 1 or not buffer_data:
        raise PreventUpdate

    logging.debug(f"Received data: {buffer_data}")

    try:
        # Rebuild the latent vector array from the buffer
        vectors = []
        for entry in buffer_data:
            vector = entry.get("feature_vector")
            if vector is not None:
                vectors.append(vector)

        latent_vectors = np.array(vectors, dtype=float)
        total_n_vectors = len(latent_vectors)

        if total_n_vectors == 0:
            raise PreventUpdate

        n_components = latent_vectors.shape[1]

        # If figure is empty (no customdata yet), return a new figure.
        if (
            not current_figure["data"]
            or len(current_figure["data"]) == 0
            or "customdata" not in current_figure["data"][0]
        ):
            # For time-based coloring
            time_array = np.arange(len(latent_vectors))
            return generate_scatter_data(
                latent_vectors,
                n_components,
                color_by="metadata",
                metadata_array=time_array,
                metadata_label="Frame Index"
            )

        # Incremental update logic
        current_n_points = len(current_figure["data"][0]["x"])

        if total_n_vectors <= current_n_points:
            logging.debug("No new latent vectors to append.")
            raise PreventUpdate

        # Use Patch for incremental update
        try:
            # Get only new vectors
            new_vectors = latent_vectors[current_n_points:]

            # Create patch
            figure_patch = Patch()

            # Update data arrays
            if "x" in current_figure["data"][0]:
                figure_patch["data"][0]["x"] = (
                    current_figure["data"][0]["x"] + new_vectors[:, 0].tolist()
                )
            else:
                figure_patch["data"][0]["x"] = new_vectors[:, 0].tolist()

            if "y" in current_figure["data"][0]:
                figure_patch["data"][0]["y"] = (
                    current_figure["data"][0]["y"] + new_vectors[:, 1].tolist()
                )
            else:
                figure_patch["data"][0]["y"] = new_vectors[:, 1].tolist()

            # Update customdata
            if "customdata" in current_figure["data"][0]:
                figure_patch["data"][0]["customdata"] = current_figure["data"][0][
                    "customdata"
                ] + [[i] for i in range(current_n_points, total_n_vectors)]
            else:
                figure_patch["data"][0]["customdata"] = [[i] for i in range(total_n_vectors)]

            # Update marker colors with full range of indices
            figure_patch["data"][0]["marker"] = dict(
                size=8,
                color=list(range(total_n_vectors)),  # Use full range of indices
                colorscale="jet",
                showscale=True,
                colorbar=dict(
                    title="Frame Index",
                ),
            )

            return figure_patch

        except Exception as e:
            logging.warning(
                f"Error patching scatter plot: {e}, preserving current figure"
            )
            return current_figure
    except PreventUpdate:
        # This is expected behavior, just re-raise
        raise
    except Exception as e:
        logging.error(f"Error updating scatter plot: {e}")
        raise PreventUpdate


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


clientside_callback(
    ClientsideFunction(namespace="liveWS", function_name="updateLiveData"),
    output=[
        Output("buffer", "data"),
        Output(
            {"base_id": "file-manager", "name": "data-project-dict"},
            "data",
            allow_duplicate=True,
        ),
        Output("live-indices", "data", allow_duplicate=True),
        Output(
            "model-loading-spinner", "style", allow_duplicate=True
        ),  # Add spinner output
        Output(
            "in-model-transition", "data", allow_duplicate=True
        ),  # Add transition state output
    ],
    inputs=[Input("ws-live", "message")],
    state=[
        State("buffer", "data"),
        State("go-live", "n_clicks"),
        State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
        State("live-indices", "data"),
        State("selected-live-models", "data"),  # Added selected models state
    ],
    prevent_initial_call=True,
)
@callback(
    Output("model-loading-spinner-text", "children"),  # Remove the style output
    Input("in-model-transition", "data"),
    Input("buffer-debounce", "n_intervals"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def update_loading_spinner_text(in_transition, n_intervals, selected_models):
    """
    Update the spinner text based on a simple rule:
    - If is_loading_model is True: "Loading Model..."
    - Otherwise: "Waiting for data..."
    """
    if not in_transition or selected_models is None:
        return "Loading Model..."  # Default when not in transition
    
    try:
        # Get loading state directly from Redis
        loading_state = redis_model_store.get_model_loading_state()
        
        # Add detailed logging
        logger.info(f"Loading state from Redis: {loading_state}")
        is_loading = loading_state and loading_state.get("is_loading_model", False)
        logger.info(f"is_loading_model value: {is_loading}")
        
        # Exact logic as requested
        if is_loading:
            logger.info("Setting spinner text to 'Loading Model...'")
            return "Loading Model..."
        else:
            logger.info("Setting spinner text to 'Waiting for data...'")
            return "Waiting for data..."
            
    except Exception as e:
        logger.error(f"Error checking Redis model state: {e}")
        return "Loading Model..."  # Default on error

@callback(
    Output("model-loading-spinner-text", "children", allow_duplicate=True),  # Set allow_duplicate=True
    Input("live-model-continue", "n_clicks"),
    State("live-autoencoder-dropdown", "value"),
    State("live-dimred-dropdown", "value"),
    prevent_initial_call=True,
)
def set_initial_loading_state(continue_clicks, autoencoder_model, dimred_model):
    """
    Set the initial loading state in Redis when models are first selected
    """
    if continue_clicks:
        try:
            # Set loading state to True initially
            if redis_model_store and redis_model_store.redis_client:
                redis_model_store.redis_client.set("model_loading_state", "True")
                redis_model_store.redis_client.set("loading_model_type", "initial")
                logger.info("Set initial loading state in Redis: is_loading=True")
                
                return "Loading Model..."  # Return just the text, not the style
        except Exception as e:
            logger.error(f"Error setting initial loading state in Redis: {e}")
    
    raise PreventUpdate

@callback(
    Output("current-page", "data", allow_duplicate=True),
    Output("image-length", "data", allow_duplicate=True),
    Output("user-upload-data-dir", "data", allow_duplicate=True),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def force_data_reset(n_clicks):
    """Reset all data stores when toggling between modes"""
    if n_clicks is not None:
        # Reset for both entering and exiting live mode
        return 0, 0, None
    raise PreventUpdate

# Dialog version dropdowns
@callback(
    Output("live-autoencoder-version-dropdown", "options"),
    Output("live-autoencoder-version-dropdown", "disabled"),
    Output("live-autoencoder-version-dropdown", "value"),
    Input("live-autoencoder-dropdown", "value"),
    prevent_initial_call=True,
)
def update_dialog_autoencoder_versions(selected_model):
    """Load version options when autoencoder model name is selected in dialog"""
    if not selected_model:
        return [], True, None
    
    try:
        versions = mlflow_client.get_model_versions(selected_model)
        if versions:
            # Default to latest version
            return versions, False, versions[0]["value"]
        return [], True, None
    except Exception as e:
        logger.error(f"Error loading autoencoder versions: {e}")
        return [], True, None


@callback(
    Output("live-dimred-version-dropdown", "options"),
    Output("live-dimred-version-dropdown", "disabled"),
    Output("live-dimred-version-dropdown", "value"),
    Input("live-dimred-dropdown", "value"),
    prevent_initial_call=True,
)
def update_dialog_dimred_versions(selected_model):
    """Load version options when dimred model name is selected in dialog"""
    if not selected_model:
        return [], True, None
    
    try:
        versions = mlflow_client.get_model_versions(selected_model)
        if versions:
            # Default to latest version
            return versions, False, versions[0]["value"]
        return [], True, None
    except Exception as e:
        logger.error(f"Error loading dimred versions: {e}")
        return [], True, None


# Sidebar version dropdowns
@callback(
    Output("live-mode-autoencoder-version-dropdown", "options"),
    Output("live-mode-autoencoder-version-dropdown", "disabled"),
    Output("live-mode-autoencoder-version-dropdown", "value"),
    Input("live-mode-autoencoder-dropdown", "value"),
    prevent_initial_call=True,
)
def update_sidebar_autoencoder_versions(selected_model):
    """Load version options when autoencoder model name is selected in sidebar"""
    if not selected_model:
        return [], True, None
    
    try:
        versions = mlflow_client.get_model_versions(selected_model)
        if versions:
            # Default to latest version
            return versions, False, versions[0]["value"]
        return [], True, None
    except Exception as e:
        logger.error(f"Error loading autoencoder versions: {e}")
        return [], True, None


@callback(
    Output("live-mode-dimred-version-dropdown", "options"),
    Output("live-mode-dimred-version-dropdown", "disabled"),
    Output("live-mode-dimred-version-dropdown", "value"),
    Input("live-mode-dimred-dropdown", "value"),
    prevent_initial_call=True,
)
def update_sidebar_dimred_versions(selected_model):
    """Load version options when dimred model name is selected in sidebar"""
    if not selected_model:
        return [], True, None
    
    try:
        versions = mlflow_client.get_model_versions(selected_model)
        if versions:
            # Default to latest version
            return versions, False, versions[0]["value"]
        return [], True, None
    except Exception as e:
        logger.error(f"Error loading dimred versions: {e}")
        return [], True, None