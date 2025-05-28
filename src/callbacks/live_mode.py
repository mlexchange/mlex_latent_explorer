import json
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from dash import Input, Output, Patch, State, callback, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from src.utils.plot_utils import generate_scatter_data
from src.utils.mlflow_utils import get_mlflow_models_live



@callback(
    Output("live-model-dialog", "is_open"),
    Output("live-autoencoder-dropdown", "options"),
    Output("live-dimred-dropdown", "options"),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def show_model_selection_dialog(n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        model_options = get_mlflow_models_live()
       
        return True, model_options, model_options
    return False, [], []


@callback(
    Output("live-model-dialog", "is_open", allow_duplicate=True),
    Output("selected-live-models", "data"),
    Input("live-model-continue", "n_clicks"),
    State("live-autoencoder-dropdown", "value"),
    State("live-dimred-dropdown", "value"),
    prevent_initial_call=True,
)
def handle_model_continue(continue_clicks, selected_autoencoder, selected_dimred):
    if continue_clicks:
        return False, {"autoencoder": selected_autoencoder, "dimred": selected_dimred}
    raise PreventUpdate


@callback(
    Output("live-model-dialog", "is_open", allow_duplicate=True),
    Output("go-live", "n_clicks"),
    Input("live-model-cancel", "n_clicks"),
    State("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def handle_model_cancel(cancel_clicks, go_live_clicks):
    if cancel_clicks and go_live_clicks is not None and go_live_clicks % 2 == 1:
        return False, go_live_clicks - 1
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
    Output("data-selection-controls", "style"),
    Output("dimension-reduction-controls", "style"),
    Output("clustering-controls", "style"),
    Output("data-overview-card", "style"),
    Output("sidebar", "active_item"),
    Output("image-card", "style"),
    Output("scatter", "style"),
    Output("heatmap", "style"),
    Output("go-live", "style"),
    Output("tooltip-go-live", "children"),
    Output("pause-button", "style"),
    Output("live-indices", "data", allow_duplicate=True),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_controls(n_clicks):
    """
    Toggle the visibility of the sidebar, data overview card, image card, and go-live button
    """
    if n_clicks is not None and n_clicks % 2 == 1:
        return (
            False,
            False,
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            ["item-1"],
            {"width": "100%", "height": "85vh"},
            {"height": "65vh"},
            {"height": "65vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "white",
                "background-color": "#D57800",
                "border": "0px",
            },
            "Go to Offline Mode",
            {
                "display": "flex",
                "font-size": "1.5rem",
                "padding": "5px",
            },
            [],
        )
    else:
        return (
            False,
            False,
            {},
            {},
            {},
            {},
            no_update,
            {"height": "64vh"},
            {"height": "46vh"},
            {"height": "46vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "#00313C",
                "background-color": "white",
                "border": "0px",
            },
            "Go to Live Mode",
            {
                "display": "none",
            },
            [],
        )


@callback(
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Input("selected-live-models", "data"),
    State("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def update_data_project_dict(selected_models, n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1 and selected_models is not None:
        return {
            "root_uri": "",
            "data_type": "tiled",
            "datasets": [],
            "project_id": None,
            "live_models": selected_models
        }
    else:
        raise PreventUpdate


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
    prevent_initial_call=True,
)
def live_update_data_project_dict(message, selected_models, n_clicks, data_project_dict, live_indices):
    """
    Update data project dict with the data uri from the live experiment
    """
    if n_clicks is not None and n_clicks % 2 == 1 and selected_models is not None:
        message = json.loads(message["data"])
        tiled_uri = message["tiled_uri"]
        split_uri = urlsplit(tiled_uri)
        path_parts = split_uri.path.rsplit("/", 1)
        root_uri = urlunsplit(
            (split_uri.scheme, split_uri.netloc, path_parts[0] + "/", "", "")
        )
        uri = path_parts[1]

        index = message["index"]

        live_indices.append(index)

        # Update cum_size according to the received index
        cum_size = max(live_indices) + 1

        # Update the data project dict
        if data_project_dict["root_uri"] != root_uri:
            data_project_dict["root_uri"] = root_uri
            data_project_dict["data_type"] = "tiled"
            if "live_models" not in data_project_dict:
                data_project_dict["live_models"] = selected_models

        if len(data_project_dict["datasets"]) == 0:
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


@callback(
    Output("buffer", "data"),
    Output("scatter", "figure", allow_duplicate=True),
    Input("ws-live", "message"),
    State("scatter", "figure"),
    State("pause-button", "n_clicks"),
    State("buffer", "data"),
    State("selected-live-models", "data"),
    prevent_initial_call=True,
)
def set_live_latent_vectors(message, current_figure, pause_n_clicks, buffer_data, selected_models):
    if selected_models is None:
        raise PreventUpdate
        
    data = json.loads(message["data"])

    latent_vectors = np.array(data["feature_vector"], dtype=float)

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
            return buffer_data, no_update
        else:
            # Append to existing buffer
            buffer_data["latent_vectors"] = np.vstack(
                (buffer_data["latent_vectors"], latent_vectors)
            )
            return buffer_data, no_update

    # If figure is empty (no customdata yet), return a new figure.
    if not current_figure["data"] or "customdata" not in current_figure["data"][0]:
        new_fig = generate_scatter_data(latent_vectors, n_components)
        return {}, new_fig

    # Otherwise, do a partial update of the existing figure using Patch
    figure_patch = Patch()

    # Build lists from the newly arriving latent vectors
    xs_new = latent_vectors[:, 0].tolist()
    ys_new = latent_vectors[:, 1].tolist()
    customdata_new = [[0]] * len(xs_new)  # or adapt to your custom data usage

    figure_patch["data"][0]["x"].extend(xs_new)
    figure_patch["data"][0]["y"].extend(ys_new)
    figure_patch["data"][0]["customdata"].extend(customdata_new)

    return {}, figure_patch


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("pause-button", "n_clicks"),
    State("buffer", "data"),
    State("scatter", "figure"),
    prevent_initial_call=True,
)
def set_buffered_latent_vectors(n_clicks, buffer_data, current_figure):
    if n_clicks is not None and n_clicks % 2 == 1 or buffer_data == {}:
        raise PreventUpdate

    num_components = buffer_data["num_components"]
    latent_vectors = buffer_data["latent_vectors"]

    # If the scatter plot is empty, generate new scatter data
    if "customdata" not in current_figure["data"][0]:
        return generate_scatter_data(latent_vectors, num_components)

    # If the scatter plot is not empty, append the new latent vectors
    else:
        for latent_vector in latent_vectors:
            current_figure["data"][0]["customdata"].append([0])
            current_figure["data"][0]["x"].append(int(latent_vector[0]))
            current_figure["data"][0]["y"].append(int(latent_vector[1]))
        return current_figure


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
