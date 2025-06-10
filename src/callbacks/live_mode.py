import logging

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

from src.utils.plot_utils import generate_scatter_data

logging.getLogger("lse.live_mode")


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
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def update_data_project_dict(n_clicks):
    if n_clicks is not None:
        return {
            "root_uri": "",
            "data_type": "tiled",
            "datasets": [],
            "project_id": None,
        }
    else:
        raise PreventUpdate


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("buffer-debounce", "n_intervals"),
    State("buffer", "data"),
    State("scatter", "figure"),
    State("pause-button", "n_clicks"),
    prevent_initial_call=True,
)
def set_live_latent_vectors(n_intervals, buffer_data, current_figure, pause_n_clicks):
    logging.debug(f"Received data: {buffer_data}")
    if buffer_data == {}:
        logging.debug("No data received in buffer.")
        raise PreventUpdate
    latent_vectors = np.zeros((0, 2), dtype=float)
    for buffer_entry in buffer_data:
        partial_latent_vectors = np.array(buffer_entry["feature_vector"], dtype=float)
        latent_vectors = np.vstack((latent_vectors, partial_latent_vectors))

    latent_vectors = (
        latent_vectors.reshape(1, -1) if latent_vectors.ndim == 1 else latent_vectors
    )
    n_components = latent_vectors.shape[1]

    # If figure is empty (no customdata yet), return a new figure.
    if not current_figure["data"] or "customdata" not in current_figure["data"][0]:
        new_fig = generate_scatter_data(latent_vectors, n_components)
        return new_fig

    # Otherwise, do a partial update of the existing figure using Patch
    figure_patch = Patch()

    # Build lists from the newly arriving latent vectors
    xs_new = latent_vectors[:, 0].tolist()
    ys_new = latent_vectors[:, 1].tolist()
    customdata_new = [[0]] * len(xs_new)  # or adapt to your custom data usage

    figure_patch["data"][0]["x"].extend(xs_new)
    figure_patch["data"][0]["y"].extend(ys_new)
    figure_patch["data"][0]["customdata"].extend(customdata_new)

    return figure_patch


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
    latent_vectors = buffer_data["feature_vector"]

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
    ],
    inputs=[Input("ws-live", "message")],
    state=[
        State("buffer", "data"),
        State("go-live", "n_clicks"),
        State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
        State("live-indices", "data"),
    ],
    prevent_initial_call=True,
)
