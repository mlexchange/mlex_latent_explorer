import json

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from app_layout import app, tiled_results


@app.callback(
    Output("data-selection", "style"),
    Output("dim-red-controls", "style"),
    Output("clustering-controls", "style"),
    Output("heatmap-controls", "style"),
    Input("go-live", "n_clicks"),
)
def toggle_controls(n_clicks):
    # accordionitems share borders, hence we add border-top for heatmap to avoid losing the top
    # border when hiding the top items
    if n_clicks is not None and n_clicks % 2 == 1:
        return [{"display": "none"}] * 3 + [
            {"border-top": "1px solid rgb(223,223,223)"}
        ]
    else:
        return [{"display": "block"}] * 3 + [{"display": "block"}]


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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
