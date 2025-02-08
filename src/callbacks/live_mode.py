import json

import numpy as np
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from src.utils.data_utils import tiled_results
from src.utils.plot_utils import generate_scatter_data


@callback(
    Output("show-clusters", "value", allow_duplicate=True),
    Output("show-feature-vectors", "value", allow_duplicate=True),
    Output("sidebar", "style"),
    Output("data-overview-card", "style"),
    Output("image-card", "style"),
    Output("go-live", "style"),
    Output("pause-button", "style"),
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
            {"width": "98vw", "height": "88vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "white",
                "background-color": "#00313C",
                "border": "0px",
            },
            {
                "display": "flex",
                "font-size": "1.5rem",
                "padding": "5px",
            },
        )
    else:
        return (
            False,
            False,
            {"overflow-y": "scroll", "height": "90vh"},
            {},
            {"height": "67vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "#00313C",
                "background-color": "white",
                "border": "0px",
            },
            {
                "display": "none",
            },
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
    Output(
        {"base_id": "file-manager", "name": "data-project-dict"},
        "data",
        allow_duplicate=True,
    ),
    Input("ws-live", "message"),
    State("go-live", "n_clicks"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def live_update_data_project_dict(message, n_clicks, data_project_dict):
    """
    Update data project dict with the data uri from the live experiment
    """
    if n_clicks is not None and n_clicks % 2 == 1:
        message = json.loads(message["data"])
        root_uri = message["root_uri"]
        data_uri = message["data_uri"]

        # Update the data project dict
        if data_project_dict["root_uri"] != root_uri:
            data_project_dict["root_uri"] = root_uri

        if len(data_project_dict["datasets"]) == 0:
            cum_size = 1
        else:
            cum_size = data_project_dict["datasets"][-1]["cumulative_data_count"] + 1

        data_project_dict["datasets"] += [
            {
                "uri": data_uri,
                "cumulative_data_count": cum_size,
            }
        ]
    return data_project_dict


@callback(
    Output("buffer", "data"),
    Output("scatter", "figure", allow_duplicate=True),
    Input("ws-live", "message"),
    State("scatter", "figure"),
    State("pause-button", "n_clicks"),
    State("buffer", "data"),
    prevent_initial_call=True,
)
def set_live_latent_vectors(message, current_figure, pause_n_clicks, buffer_data):
    # Parse the incoming message
    message = json.loads(message["data"])
    latent_vectors_uri = message["feature_vector_uri"]

    # Retrieve data from tiled_uri
    latent_vectors = tiled_results.get_data_by_trimmed_uri(latent_vectors_uri)
    n_components = latent_vectors.shape[1]

    # If the pause button is clicked, buffer the latent vectors
    if pause_n_clicks is not None and pause_n_clicks % 2 == 1:
        if buffer_data == {}:
            buffer_data["num_components"] = n_components
            buffer_data["latent_vectors"] = latent_vectors
            return buffer_data, no_update
        else:
            buffer_data["latent_vectors"] = np.vstack(
                (buffer_data["latent_vectors"], latent_vectors)
            )
            return buffer_data, no_update

    # If the scatter plot is empty, generate new scatter data
    if "customdata" not in current_figure["data"][0]:
        return {}, generate_scatter_data(latent_vectors, n_components)

    # If the scatter plot is not empty, append the new latent vectors
    else:
        current_figure["data"][0]["customdata"].append([0])
        current_figure["data"][0]["x"].append(int(latent_vectors[:, 0]))
        current_figure["data"][0]["y"].append(int(latent_vectors[:, 1]))
        return {}, current_figure


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
    Input("pause-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_pause_button(n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        icon = "lucide:circle-play"
    else:
        icon = "lucide:circle-pause"
    return DashIconify(icon=icon, style={"padding": "0px"})


@callback(
    Output("pause-button", "children"),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_pause_button_go_live(go_live_n_clicks):
    return DashIconify(icon="lucide:circle-pause", style={"padding": "0px"})
