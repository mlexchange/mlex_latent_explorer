import json

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate

from src.utils.data_utils import tiled_results


@callback(
    Output("sidebar", "style"),
    Output("data-overview-card", "style"),
    Output("image-card", "style"),
    Output("image-card-body", "style"),
    Output("go-live", "style"),
    Input("go-live", "n_clicks"),
)
def toggle_controls(n_clicks):
    """
    Toggle the visibility of the sidebar, data overview card, image card, and go-live button
    """
    if n_clicks is not None and n_clicks % 2 == 1:
        return (
            {"display": "none"},
            {"display": "none"},
            {"width": "98vw", "height": "88vh"},
            {"height": "100%"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "white",
                "background-color": "#00313C",
                "border": "0px",
            },
        )
    else:
        return (
            {"overflow-y": "scroll", "height": "90vh"},
            {"display": "block"},
            {"display": "block"},
            {"height": "62vh"},
            {
                "display": "flex",
                "font-size": "40px",
                "padding": "5px",
                "color": "#00313C",
                "background-color": "white",
                "border": "0px",
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
        flow_id = message["flow_id"]
        metadata = tiled_results.get_metadata(flow_id)
        root_uri = metadata["io_parameters"]["root_uri"]
        assert len(metadata["io_parameters"]["data_uris"]) == 1
        data_uri = metadata["io_parameters"]["data_uris"][0]
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
    Output("latent_vectors", "data", allow_duplicate=True),
    Input("ws-live", "message"),
    State("latent_vectors", "data"),
    State("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def set_live_latent_vectors(message, latent_vectors, n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        message = message["data"]
        message = json.loads(message)
        flow_id = message["flow_id"]
        new_latent_vectors = tiled_results.get_latent_vectors(flow_id)
        if latent_vectors is None:
            latent_vectors = np.array(new_latent_vectors)
        else:
            latent_vectors = np.concatenate(
                [latent_vectors, new_latent_vectors], axis=0
            )
        return latent_vectors
    else:
        raise PreventUpdate


@callback(
    Output("latent_vectors", "data", allow_duplicate=True),
    Output("heatmap", "figure", allow_duplicate=True),
    Input("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def live_clear_plots(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return None, go.Figure(
        data=go.Heatmap(),
        layout=dict(
            autosize=True,
            margin=go.layout.Margin(l=20, r=20, b=20, t=20, pad=0),
        ),
    )
