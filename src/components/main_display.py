import dash_bootstrap_components as dbc
from dash import dcc, html

from ..utils.plot_utils import draw_rows, plot_heatmap, plot_scatter

NUM_IMGS_OVERVIEW = 6


def main_display():
    main_display = html.Div(
        [
            dbc.Card(
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
                                    style={"width": "84%"},
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
                        style={"height": "20vh"},
                    ),
                ]
            ),
            dbc.Card(
                id="image-card",
                children=[
                    dbc.CardHeader("Latent Space Analysis"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="scatter",
                                        figure=plot_scatter(),
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="heatmap",
                                        figure=plot_heatmap(),
                                    ),
                                    width=6,
                                ),
                            ],
                            className="g-0",
                        ),
                        style={"height": "55vh"},
                    ),
                ],
            ),
        ],
    )
    return main_display
