import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify
from mlex_utils.dash_utils.components_bootstrap.component_utils import (
    DbcControlItem as ControlItem,
)

from ..utils.mask_utils import get_mask_options


def get_scatter_control_panel():
    """
    Creates the scatter control panel
    """
    scatter_control_panel = [
        ControlItem(
            "Scatter Colors",
            "scatter-color-title",
            dbc.Select(
                id="scatter-color",
                options=[
                    {"label": "Cluster", "value": "cluster"},
                    {"label": "Label", "value": "label"},
                ],
                value="cluster",
            ),
        ),
        html.P(),
        ControlItem(
            "Select cluster",
            "cluster-dropdown-title",
            dbc.Select(
                id="cluster-dropdown",
                value=-1,
            ),
        ),
        dcc.Interval(
            id="interval-component",
            interval=3000,  # in milliseconds
            max_intervals=-1,  # keep triggering indefinitely, None
            n_intervals=0,
        ),
    ]
    return scatter_control_panel


def get_heatmap_control_panel():
    """
    Creates the heatmap control panel
    """
    heatmap_control_panel = [
        dbc.Label(
            [
                "Select a Group of Points using ",
                html.Span(
                    html.I(DashIconify(icon="lucide:lasso")),
                    className="icon",
                ),
                " or ",
                html.Span(
                    html.I(DashIconify(icon="lucide:box-select")),
                    className="icon",
                ),
                " tools",
            ],
            className="mb-3",
        ),
        dbc.Label(
            id="stats-div",
            children=[
                "Number of images selected: 0",
                html.Br(),
                "Clusters represented: N/A",
                html.Br(),
                "Labels represented: N/A",
            ],
        ),
        ControlItem(
            "Display Image Options",
            "display-image-options-title",
            dbc.Select(
                id="mean-std-toggle",
                options=[
                    {"label": "Mean", "value": "mean"},
                    {"label": "Standard Deviation", "value": "sigma"},
                ],
                value="mean",
                className="mb-2",
            ),
        ),
        html.P(),
    ]
    return heatmap_control_panel


def get_clustering_control_panel(model_list):
    cluster_algo_panel = [
        ControlItem(
            "Algorithm",
            "cluster-algo-dropdown-title",
            dbc.Select(
                id="cluster-algo-dropdown",
                options=model_list,
                value=model_list[0],
            ),
        ),
        html.Div(id="additional-cluster-params"),
        html.P(),
        html.Div(
            [
                dbc.Button(
                    "Cluster",
                    color="primary",
                    id="run-cluster-algo",
                    style={"width": "95%"},
                ),
            ],
            className="row",
            style={
                "align-items": "center",
                "justify-content": "center",
            },
        ),
        html.Div(id="invisible-submit-div"),
    ]
    return cluster_algo_panel


def sidebar(file_explorer, job_manager, clustering_models):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:      Dash file explorer
        job_manager:        Job manager object
        clustering_models:  Clustering models
    """
    sidebar = [
        dbc.Accordion(
            id="sidebar",
            children=[
                dbc.AccordionItem(
                    title="Data selection",
                    children=file_explorer,
                ),
                dbc.AccordionItem(
                    title="Data transformation",
                    children=[
                        ControlItem(
                            "",
                            "empty-title-log-transform",
                            dbc.Switch(
                                id="log-transform",
                                value=False,
                                label="Log Transform",
                            ),
                        ),
                        html.P(),
                        ControlItem(
                            "Min-Max Percentile",
                            "min-max-percentile-title",
                            dcc.RangeSlider(
                                id="min-max-percentile",
                                min=0,
                                max=100,
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True,
                                },
                                value=[0, 100],
                            ),
                        ),
                        html.P(),
                        ControlItem(
                            "Mask Selection",
                            "mask-dropdown-title",
                            dbc.Select(
                                id="mask-dropdown",
                                options=get_mask_options(),
                                value="None",
                            ),
                        ),
                    ],
                ),
                dbc.AccordionItem(
                    children=[
                        job_manager,
                        ControlItem(
                            "",
                            "empty-title-feature-vectors",
                            dbc.Switch(
                                id="show-feature-vectors",
                                value=False,
                                label="Show Feature Vectors",
                                disabled=True,
                            ),
                        ),
                    ],
                    title="Dimension Reduction",
                ),
                dbc.AccordionItem(
                    children=get_clustering_control_panel(clustering_models),
                    title="Clustering",
                ),
                dbc.AccordionItem(
                    children=get_heatmap_control_panel() + get_scatter_control_panel(),
                    title="Plot Control Panel",
                ),
            ],
            style={"overflow-y": "scroll", "height": "90vh"},
        )
    ]
    return sidebar
