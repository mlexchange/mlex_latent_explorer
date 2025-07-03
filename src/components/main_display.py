import logging
import os

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_extensions import WebSocket
from dash_iconify import DashIconify

from ..utils.plot_utils import draw_rows, plot_empty_heatmap, plot_empty_scatter
from .model_selection_dialog import create_model_selection_dialog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_IMGS_OVERVIEW = 6
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8765/lse")
logger.info(f"WebSocket URL: {WEBSOCKET_URL}")


def main_display():
    main_display = html.Div(
        id="main-display",
        style={"padding": "0px 10px 0px 510px"},
        children=[
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
                                                    dbc.ButtonGroup(
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
                                                                style={
                                                                    "display": "none"
                                                                },
                                                                className="me-1",
                                                            ),
                                                            dbc.Tooltip(
                                                                "Pause live display",
                                                                id="tooltip-pause-button",
                                                                target="pause-button",
                                                                placement="top",
                                                            ),
                                                            dbc.Button(
                                                                DashIconify(
                                                                    icon="lucide:eraser",
                                                                    style={
                                                                        "padding": "0px",
                                                                    },
                                                                ),
                                                                id="clear-selection-button",
                                                                color="primary",
                                                                style={
                                                                    "display": "flex",
                                                                    "font-size": "1.3rem",
                                                                    "padding": "6.5px",
                                                                },
                                                            ),
                                                            dbc.Tooltip(
                                                                "Clear lasso selection",
                                                                target="clear-selection-button",
                                                                placement="top",
                                                            ),
                                                        ],
                                                        className="w-auto",
                                                    ),
                                                    width=2,
                                                    style={"text-align": "right"},
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
                                        style={"text-align": "center"},
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="scatter",
                                            figure=plot_empty_scatter(),
                                            style={"height": "46vh"},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="heatmap",
                                            figure=plot_empty_heatmap(),
                                            style={"height": "46vh"},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-0",
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
            # Add model selection dialog for live mode
            create_model_selection_dialog(),
            dcc.Store(id="selected-live-models", data=None),
            dcc.Store(id="buffer", data={}),
            dcc.Interval(id="buffer-debounce", interval=100, n_intervals=0),  # 100ms
            dcc.Store(id="live-indices", data=[]),
            WebSocket(id="ws-live", url=WEBSOCKET_URL),
        ],
    )
    return main_display
