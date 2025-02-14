import os

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_extensions import WebSocket
from dash_iconify import DashIconify

from ..utils.plot_utils import draw_rows, plot_empty_heatmap, plot_empty_scatter

NUM_IMGS_OVERVIEW = 6
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "localhost")
WEBSOCKET_PORT = os.getenv("WEBSOCKET_PORT", 5000)


def main_display():
    main_display = html.Div(
        [
            dbc.Card(
                id="data-overview-card",
                children=[
                    dbc.CardHeader("Data Overview"),
                    dbc.CardBody(
                        id="data-overview",
                        children=dbc.Row(
                            [
                                dbc.Button(
                                    className="fa fa-step-backward",
                                    id="first-page",
                                    color="secondary",
                                    style={"width": "3%", "margin-right": "1%"},
                                    disabled=True,
                                ),
                                dbc.Button(
                                    className="fa fa-chevron-left",
                                    id="prev-page",
                                    style={"width": "3%"},
                                    disabled=True,
                                ),
                                html.Div(
                                    draw_rows(1, NUM_IMGS_OVERVIEW),
                                    style={"width": "84%", "height": "12vh"},
                                ),
                                dbc.Button(
                                    className="fa fa-chevron-right",
                                    id="next-page",
                                    style={"width": "3%", "margin-right": "1%"},
                                    disabled=True,
                                ),
                                dbc.Button(
                                    className="fa fa-step-forward",
                                    id="last-page",
                                    color="secondary",
                                    style={
                                        "width": "3%",
                                    },
                                    disabled=True,
                                ),
                                dcc.Store(id="current-page", data=0),
                            ],
                            justify="center",
                            style={"margin-top": "0%"},
                        ),
                        style={"height": "15vh"},
                    ),
                ],
            ),
            dbc.Card(
                id="image-card",
                style={"height": "64vh"},
                children=[
                    dbc.CardHeader("Latent Space Analysis"),
                    dbc.CardBody(
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label(
                                                            [
                                                                "Select a Group of Points using ",
                                                                html.Span(
                                                                    html.I(
                                                                        DashIconify(
                                                                            icon="lucide:lasso"
                                                                        )
                                                                    ),
                                                                    className="icon",
                                                                ),
                                                                " or ",
                                                                html.Span(
                                                                    html.I(
                                                                        DashIconify(
                                                                            icon="lucide:box-select"
                                                                        )
                                                                    ),
                                                                    className="icon",
                                                                ),
                                                                " tools",
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            DashIconify(
                                                                icon="lucide:circle-pause",
                                                                style={
                                                                    "padding": "0px"
                                                                },
                                                            ),
                                                            id="pause-button",
                                                            color="primary",
                                                            style={"display": "none"},
                                                        ),
                                                    ],
                                                    width=1,
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.Col(
                                        dbc.RadioItems(
                                            id="mean-std-toggle",
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {"label": "Mean", "value": "mean"},
                                                {
                                                    "label": "Standard Deviation",
                                                    "value": "sigma",
                                                },
                                            ],
                                            value="mean",
                                            label_style={"width": "180px"},
                                        ),
                                        style={"text-align": "right"},
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="scatter",
                                            figure=plot_empty_scatter(),
                                            style={"height": "95%"},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="heatmap",
                                            figure=plot_empty_heatmap(),
                                            style={"height": "95%"},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-0",
                                style={"height": "85%"},
                            ),
                            dbc.Row(
                                dbc.Label(
                                    id="stats-div",
                                    children="Number of images selected: 0",
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="buffer", data={}),
            dcc.Store(id="live-indices", data=[]),
            WebSocket(id="ws-live", url=f"ws:{WEBSOCKET_URL}:{WEBSOCKET_PORT}"),
        ],
    )
    return main_display
