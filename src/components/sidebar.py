import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify
from mlex_utils.dash_utils.components_bootstrap.component_utils import (
    DbcControlItem as ControlItem,
)

import os

from ..utils.mask_utils import get_mask_options


def sidebar(file_explorer, job_manager, clustering_job_manager):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:          Dash file explorer
        job_manager:            Job manager object
        clustering_job_manager: Job manager object for clustering
    """
    # MLflow models initialization happens in execute.py
    
    sidebar = html.Div(
        [
            dbc.Offcanvas(
                id="sidebar-offcanvas",
                is_open=True,
                backdrop=False,
                scrollable=True,
                style={
                    "padding": "80px 0px 0px 0px",
                    "width": "500px",
                },  # Avoids being covered by the navbar
                title="Controls",
                children=dbc.Accordion(
                    id="sidebar",
                    always_open=True,
                    children=[
                        dbc.AccordionItem(
                            id="data-selection-controls",
                            title="Data selection",
                            children=file_explorer,
                        ),
                        dbc.AccordionItem(
                            id="data-transformation-controls",
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
                            id="dimension-reduction-controls",
                            children=[
                                ControlItem(
                                    "Autoencoder Model",
                                    "mlflow-model-title",
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Select(
                                                    id="mlflow-model-dropdown",
                                                    options=[],  # Empty initially, populated by callback
                                                    value=None,
                                                ),
                                                width=10
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    DashIconify(
                                                        icon="tabler:refresh",
                                                        width=20,
                                                        height=20
                                                    ),
                                                    id="refresh-mlflow-models",
                                                    color="light",
                                                    size="sm",
                                                    style={"margin-left": "-10px"}
                                                ),
                                                width=1,
                                                style={"padding-left": "0"}
                                            )
                                        ])
                                    ])
                                ),
                                html.Div(style={"height": "20px"}),  # Additional spacing
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
                            id="clustering-controls",
                            children=[
                                clustering_job_manager,
                                ControlItem(
                                    "",
                                    "empty-title-clusters",
                                    dbc.Switch(
                                        id="show-clusters",
                                        value=False,
                                        label="Show Clusters",
                                        disabled=True,
                                    ),
                                ),
                            ],
                            title="Clustering",
                        ),
                    ],
                ),
            ),
            create_show_sidebar_affix(),
        ]
    )

    return sidebar


# get_mlflow_models function moved to execute.py
# Functions that retrieve MLflow models moved to execute.py


def create_show_sidebar_affix():
    return html.Div(
        [
            dbc.Button(
                DashIconify(icon="circum:settings", width=30),
                id="sidebar-view",
                size="sm",
                color="secondary",
                className="rounded-circle",
                style={"aspectRatio": "1 / 1"},
            ),
            dbc.Tooltip(
                "Toggle sidebar",
                target="sidebar-view",
                placement="top",
            ),
        ],
        style={
            "position": "fixed",
            "bottom": "70px",
            "right": "10px",
            "zIndex": 9999999,  # Note: zIndex is unitless
            "opacity": "0.8",
        },
    )