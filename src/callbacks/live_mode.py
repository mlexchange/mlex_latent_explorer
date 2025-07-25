import logging
import os

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
    Output("live-autoencoder-dropdown", "value"),
    Output("live-dimred-dropdown", "value"),
    Input("go-live", "n_clicks"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def show_model_selection_dialog(n_clicks, last_selected_models):
    if n_clicks is not None and n_clicks % 2 == 1:
        # Get model options filtered by type
        autoencoder_options = mlflow_client.get_mlflow_models(
            livemode=True, model_type="autoencoder"
        )
        dimred_options = mlflow_client.get_mlflow_models(
            livemode=True, model_type="dimension_reduction"
        )

        # Set default values from previous selection if available
        autoencoder_default = None
        dimred_default = None

        if last_selected_models is not None:
            autoencoder_default = last_selected_models.get("autoencoder")
            dimred_default = last_selected_models.get("dimred")

        return (
            True,
            autoencoder_options,
            dimred_options,
            autoencoder_default,
            dimred_default,
        )
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
    Output("model-loading-spinner", "style", allow_duplicate=True),
    Output("in-model-transition", "data", allow_duplicate=True),
    Input("live-model-continue", "n_clicks"),
    State("live-autoencoder-dropdown", "value"),
    State("live-dimred-dropdown", "value"),
    State("live-autoencoder-dropdown", "options"),
    State("live-dimred-dropdown", "options"),
    prevent_initial_call=True,
)
def handle_model_continue(
    continue_clicks,
    selected_autoencoder,
    selected_dimred,
    autoencoder_options,
    dimred_options,
):
    """
    Handle the Continue button click in the model selection dialog.
    This function:
    1. Checks model compatibility
    2. Stores models in Redis if compatible
    3. Updates UI elements for Live Mode
    4. Shows loading spinner
    """
    if not continue_clicks:
        raise PreventUpdate

    # Check model compatibility before proceeding
    if not mlflow_client.check_model_compatibility(
        selected_autoencoder, selected_dimred
    ):
        logger.warning(
            f"Incompatible models selected: {selected_autoencoder}, {selected_dimred}"
        )
        # Prevent dialog from closing
        return no_update

    # Store models in Redis
    try:
        # Store autoencoder model
        logger.info(f"Storing autoencoder model from dialog: {selected_autoencoder}")
        redis_model_store.store_autoencoder_model(selected_autoencoder)

        # Store dimension reduction model
        logger.info(f"Storing dimension reduction model from dialog: {selected_dimred}")
        redis_model_store.store_dimred_model(selected_dimred)
    except Exception as e:
        logger.error(f"Error storing models in Redis: {e}")
        return no_update

    # Models are compatible and stored in Redis, proceed with closing dialog and transition
    selected_models = {
        "autoencoder": selected_autoencoder,
        "dimred": selected_dimred,
    }

    # Show loading spinner and set transition state to True
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
    Output("live-model-continue", "color", allow_duplicate=True),
    Output("live-model-continue", "children", allow_duplicate=True),
    Input("live-autoencoder-dropdown", "value"),
    Input("live-dimred-dropdown", "value"),
    prevent_initial_call=True,  # Add this to prevent initial call
)
def toggle_continue_button(selected_autoencoder, selected_dimred):
    """
    Disable the continue button if models are not selected or are incompatible
    Also update the button color and text to indicate state
    """
    # First check if either model is not selected
    if selected_autoencoder is None or selected_dimred is None:
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
            None,
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
                "data_type": "tiled",
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
    State("live-mode-dimred-dropdown", "value"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def update_live_models(n_clicks, autoencoder_model, dimred_model, data_project_dict):
    """
    Update the selected models when the sidebar update button is clicked.
    This function:
    1. Checks model compatibility
    2. Stores models in Redis if compatible
    3. Updates data project dictionary
    4. Resets visualizations
    5. Shows loading spinner
    """
    if n_clicks is None:
        raise PreventUpdate

    # Check model compatibility
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

    # Store models in Redis
    try:
        # Store autoencoder model
        logger.info(f"Storing autoencoder model from sidebar: {autoencoder_model}")
        redis_model_store.store_autoencoder_model(autoencoder_model)

        # Store dimension reduction model
        logger.info(f"Storing dimension reduction model from sidebar: {dimred_model}")
        redis_model_store.store_dimred_model(dimred_model)
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

    # Update the selected models
    selected_models = {"autoencoder": autoencoder_model, "dimred": dimred_model}

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
    Input("live-mode-dimred-dropdown", "value"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def reset_update_button(autoencoder_model, dimred_model, selected_models):
    """
    Reset the update button color when dropdowns change and check model compatibility
    """
    # If no models are selected yet or either dropdown is empty
    if selected_models is None or autoencoder_model is None or dimred_model is None:
        return True, "secondary", "Update Models"  # Disabled, secondary color

    # Check if models are compatible using the shared compatibility check function
    is_compatible = mlflow_client.check_model_compatibility(
        autoencoder_model, dimred_model
    )

    if not is_compatible:
        # Models are incompatible - disable button with danger color
        return True, "danger", "Incompatible Models"

    # Check if the current selection is different from what's already selected
    models_changed = autoencoder_model != selected_models.get(
        "autoencoder"
    ) or dimred_model != selected_models.get("dimred")

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

        n_components = latent_vectors.shape[1]

        # If figure is empty (no customdata yet), return a new figure.
        if (
            not current_figure["data"]
            or len(current_figure["data"]) == 0
            or "customdata" not in current_figure["data"][0]
        ):
            return generate_scatter_data(latent_vectors, n_components)

        # Incremental update logic
        current_n_points = len(current_figure["data"][0]["x"])
        total_n_vectors = len(latent_vectors)

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

            if "customdata" in current_figure["data"][0]:
                figure_patch["data"][0]["customdata"] = current_figure["data"][0][
                    "customdata"
                ] + [[0]] * len(new_vectors)
            else:
                figure_patch["data"][0]["customdata"] = [[0]] * len(new_vectors)

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
