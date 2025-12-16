import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify
from mlex_utils.dash_utils.components_bootstrap.component_utils import (
    DbcControlItem as ControlItem,
)

from ..utils.mask_utils import get_mask_options


def sidebar(file_explorer, job_manager, clustering_job_manager):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:          Dash file explorer
        job_manager:            Job manager object
        clustering_job_manager: Job manager object for clustering
    """

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
                            id="live-mode-models",
                            title="Live Mode Models",
                            children=[
                                # NEW: Add experiment name display at the top
                                html.Div(
                                    id="live-experiment-name-display",
                                    children=[
                                        html.Div(
                                            [
                                                html.Strong("Experiment: ", style={"color": "#00313C"}),
                                                html.Span(
                                                    id="live-experiment-name-text",
                                                    children="",
                                                    style={"color": "#D57800", "fontWeight": "500"}
                                                ),
                                            ],
                                            style={
                                                "padding": "10px",
                                                "backgroundColor": "#f8f9fa",
                                                "borderRadius": "5px",
                                                "marginBottom": "15px",
                                                "border": "1px solid #dee2e6"
                                            }
                                        ),
                                    ],
                                    style={"display": "none"}  # Hidden by default, shown in live mode
                                ),
                                ControlItem(
                                    "Autoencoder Model",
                                    "live-mode-autoencoder-title",
                                    dbc.Select(
                                        id="live-mode-autoencoder-dropdown",
                                        options=[],
                                        placeholder="Select model name...",
                                    ),
                                ),
                                html.P(),
                                ControlItem(
                                    "Autoencoder Version",
                                    "live-mode-autoencoder-version-title",
                                    dbc.Select(
                                        id="live-mode-autoencoder-version-dropdown",
                                        options=[],
                                        placeholder="Select version...",
                                        disabled=True,
                                    ),
                                ),
                                html.P(),
                                ControlItem(
                                    "Dimension Reduction Model",
                                    "live-mode-dimred-title",
                                    dbc.Select(
                                        id="live-mode-dimred-dropdown",
                                        options=[],
                                        placeholder="Select model name...",
                                    ),
                                ),
                                html.P(),
                                ControlItem(
                                    "Dimension Reduction Version",
                                    "live-mode-dimred-version-title",
                                    dbc.Select(
                                        id="live-mode-dimred-version-dropdown",
                                        options=[],
                                        placeholder="Select version...",
                                        disabled=True,
                                    ),
                                ),
                                html.P(),
                                # Add warning text before the button with gray color and circle icon
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                DashIconify(
                                                    icon="lucide:alert-circle", 
                                                    width=20, 
                                                    height=20,
                                                    style={"marginRight": "5px"}
                                                ),
                                                html.Span(
                                                    "Updating models will refresh panels and data will be lost.",
                                                    style={"fontSize": "0.9rem"}
                                                ),
                                            ],
                                            style={
                                                "display": "flex", 
                                                "alignItems": "center", 
                                                "color": "#6c757d",  # Gray color (Bootstrap secondary)
                                                "marginBottom": "10px"
                                            }
                                        ),
                                        dbc.Button(
                                            "Update Models",
                                            id="update-live-models-button",
                                            color="primary",
                                            className="w-100",
                                            disabled=False,
                                        ),
                                    ]
                                ),
                            ],
                            style={"display": "none"},  # Hidden by default (offline mode)
                        ),
                        dbc.AccordionItem(
                            id="dimension-reduction-controls",
                            children=[
                                ControlItem(
                                    "Autoencoder Model",
                                    "mlflow-model-title",
                                    html.Div(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Select(
                                                            id="mlflow-model-dropdown",
                                                            options=[],  # Empty initially, populated by callback
                                                            value=None,
                                                            placeholder="Please select a model",
                                                        ),
                                                        width=10,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            DashIconify(
                                                                icon="tabler:refresh",
                                                                width=20,
                                                                height=20,
                                                            ),
                                                            id="refresh-mlflow-models",
                                                            color="light",
                                                            size="sm",
                                                            className="rounded-circle",
                                                            style={
                                                                "aspectRatio": "1 / 1"
                                                            },
                                                        ),
                                                        width=1,
                                                        style={"padding-left": "0"},
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ),
                                html.Div(
                                    style={"height": "20px"}
                                ),  # Additional spacing
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
                        # Moved experiment replay section after clustering
                        dbc.AccordionItem(
                            id="experiment-replay-controls",
                            title="Experiment Replay",
                            children=[
                                # CHANGED: Replace dropdown with DatePickerSingle
                                ControlItem(
                                    "Select Date",
                                    "date-picker-title",
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col(
                                                html.Div([
                                                    dcc.DatePickerSingle(
                                                        id="replay-date-picker",
                                                        placeholder="Select a date",
                                                        display_format="YYYY-MM-DD",
                                                        date=None,
                                                        disabled=False,
                                                        calendar_orientation="horizontal",
                                                        with_portal=False,
                                                        number_of_months_shown=1,
                                                    ),
                                                ], style={"width": "100%"}, id="datepicker-wrapper"),
                                                width=10,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    DashIconify(
                                                        icon="tabler:refresh",
                                                        width=18,
                                                        height=18,
                                                    ),
                                                    id="refresh-available-dates",
                                                    color="light",
                                                    size="sm",
                                                    className="rounded-circle",
                                                    style={"aspectRatio": "1 / 1"},
                                                ),
                                                width=1,
                                                style={"padding-left": "0"},
                                            ),
                                        ], className="align-items-center"),
                                        # CHANGED: Add store for available dates
                                        dcc.Store(id="available-dates-store", data=[]),
                                    ]),
                                ),
                                html.P(),
                                # Experiment Name dropdown
                                ControlItem(
                                    "Experiment Name",
                                    "experiment-name-title",
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Select(
                                                    id="experiment-name-dropdown",
                                                    options=[],
                                                    value=None,
                                                    placeholder="Select an experiment name",
                                                ),
                                                width=10,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    DashIconify(
                                                        icon="tabler:refresh",
                                                        width=18,
                                                        height=18,
                                                    ),
                                                    id="refresh-experiment-names",
                                                    color="light",
                                                    size="sm",
                                                    className="rounded-circle",
                                                    style={"aspectRatio": "1 / 1"},
                                                ),
                                                width=1,
                                                style={"padding-left": "0"},
                                            ),
                                        ], className="align-items-center"),
                                    ]),
                                ),
                                html.P(),
                                # UUID dropdown
                                ControlItem(
                                    "Experiment UUID",
                                    "experiment-uuid-title",
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Select(
                                                    id="experiment-uuid-dropdown",
                                                    options=[],
                                                    value=None,
                                                    placeholder="Select an experiment UUID",
                                                ),
                                                width=10,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    DashIconify(
                                                        icon="tabler:refresh",
                                                        width=18,
                                                        height=18,
                                                    ),
                                                    id="refresh-experiment-uuids",
                                                    color="light",
                                                    size="sm",
                                                    className="rounded-circle",
                                                    style={"aspectRatio": "1 / 1"},
                                                ),
                                                width=1,
                                                style={"padding-left": "0"},
                                            ),
                                        ], className="align-items-center"),
                                    ]),
                                ),
                                html.P(),
                                # Data Range slider
                                ControlItem(
                                    "Data Range",
                                    "replay-data-range-title",
                                    dcc.RangeSlider(
                                        id="replay-data-range",
                                        min=0,
                                        max=100,
                                        step=1,
                                        marks={i: str(i) for i in range(0, 101, 20)},
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                        value=[0, 100],
                                    ),
                                ),
                                html.Div(style={"height": "20px"}),
                                dbc.Button(
                                    "Load Experiment",
                                    id="load-experiment-button",
                                    color="primary",
                                    className="w-100",
                                    disabled=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ),
            create_show_sidebar_affix(),
        ]
    )

    return sidebar


def create_show_sidebar_affix():
    return html.Div(
        [
            dbc.Button(
                DashIconify(icon="circum:settings", width=20),
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
            "bottom": "60px",
            "right": "10px",
            "zIndex": 9999,
            "opacity": "0.8",
        },
    )