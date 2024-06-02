import os

import dash_bootstrap_components as dbc
import diskcache
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.long_callback import DiskcacheLongCallbackManager
from dash_extensions import WebSocket
from dash_iconify import DashIconify
from dotenv import load_dotenv
from file_manager.main import FileManager

import templates
from utils_tiled import TiledResults

load_dotenv(".env", override=True)

ALGORITHM_DATABASE = {
    "PCA": "PCA",
    "UMAP": "UMAP",
}

CLUSTER_ALGORITHM_DATABASE = {
    "KMeans": "KMeans",
    "DBSCAN": "DBSCAN",
    "HDBSCAN": "HDBSCAN",
}

READ_DIR = os.getenv("READ_DIR")
WRITE_DIR = os.getenv("WRITE_DIR")
API_KEY = os.getenv("API_KEY", None)
if API_KEY == "":
    API_KEY = None

if os.path.exists(f"{os.getcwd()}/src/example_dataset"):
    EXAMPLE_DATASETS = [
        {
            "label": "Synthetic Shapes",
            "value": f"{os.getcwd()}/src/example_dataset/Demoshapes.npz",
        }
    ]
else:
    EXAMPLE_DATASETS = []

# Tiled Server to store results
RESULT_TILED_URI = os.getenv("RESULT_TILED_URI", "")
RESULT_TILED_API_KEY = os.getenv("RESULT_TILED_API_KEY", None)
tiled_results = TiledResults(RESULT_TILED_URI, RESULT_TILED_API_KEY)
tiled_results.prep_result_tiled_containers()

# SETUP DASH APP
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    "../assets/segmentation-style.css",
]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    long_callback_manager=long_callback_manager,
)

server = app.server

dash_file_explorer = FileManager(READ_DIR, open_explorer=False, api_key=API_KEY)
dash_file_explorer.init_callbacks(app)

# BEGIN DASH CODE
header = templates.header()

# right panel: file manager, scatter plot, individual image  plot
scatter_control_panel = dbc.AccordionItem(
    title="Scatter Plot Control Panel",
    children=[
        dbc.CardBody(
            [
                dbc.Label("Scatter Colors", className="mr-3"),
                dcc.RadioItems(
                    id="scatter-color",
                    options=[
                        {"label": "cluster", "value": "cluster"},
                        {"label": "label", "value": "label"},
                    ],
                    value="cluster",
                    style={"min-width": "250px"},
                    className="mb-2",
                ),
                dbc.Label("Select cluster", className="mr-3"),
                dcc.Dropdown(
                    id="cluster-dropdown",
                    value=-1,
                    style={"min-width": "250px"},
                    className="mb-2",
                ),
                dbc.Label("Select label", className="mr-3"),
                dcc.Dropdown(
                    id="label-dropdown",
                    value=-2,
                    style={"min-width": "250px"},
                ),
            ]
        ),
        dcc.Interval(
            id="interval-component",
            interval=3000,  # in milliseconds
            max_intervals=-1,  # keep triggering indefinitely, None
            n_intervals=0,
        ),
    ],
)

heatmap_control_panel = dbc.AccordionItem(
    id="heatmap-controls",
    title="Heatmap Control Panel",
    children=[
        dbc.CardBody(
            [
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
                        " Tools :",
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
                dbc.Label("Display Image Options", className="mr-3"),
                dcc.RadioItems(
                    id="mean-std-toggle",
                    options=[
                        {"label": "Mean", "value": "mean"},
                        {"label": "Standard Deviation", "value": "sigma"},
                    ],
                    value="mean",
                    style={"min-width": "250px"},
                    className="mb-2",
                ),
            ]
        ),
    ],
)

image_panel = [
    dbc.Card(
        id="image-card",
        children=[
            dbc.CardHeader(
                id="data-selection",
                children=[
                    dbc.Label("Select a Dataset", className="mr-2"),
                    dash_file_explorer.file_explorer,
                    dbc.Label("Or try Example Dataset", className="mr-2"),
                    dcc.Dropdown(
                        id="example-dataset-selection",
                        options=EXAMPLE_DATASETS,
                        clearable=True,
                        style={"margin-bottom": "1rem"},
                    ),
                ],
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    id="scatter",
                                    figure=go.Figure(
                                        go.Scattergl(mode="markers"),
                                        layout=go.Layout(
                                            autosize=True,
                                            margin=go.layout.Margin(
                                                l=20,
                                                r=20,
                                                b=20,
                                                t=20,
                                                pad=0,
                                            ),
                                        ),
                                    ),
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    id="heatmap",
                                    figure=go.Figure(
                                        go.Heatmap(),
                                        layout=go.Layout(
                                            autosize=True,
                                            margin=go.layout.Margin(
                                                l=20,
                                                r=20,
                                                b=20,
                                                t=20,
                                                pad=0,
                                            ),
                                        ),
                                    ),
                                ),
                                width=6,
                            ),
                        ]
                    ),
                ]
            ),
        ],
    )
]

# left panel: choose algorithms, submit job, choose scatter plot attributes, and statistics...
algo_panel = dbc.AccordionItem(
    id="dim-red-controls",
    title="Select Dimension Reduction Algorithms",
    children=[
        dbc.CardBody(
            [
                dbc.Label("Optional: Select Pre-trained Autoencoder", className="mr-2"),
                dcc.Dropdown(
                    id="feature-vector-model-list",
                    clearable=True,
                    style={"margin-bottom": "1rem"},
                ),
                html.Hr(),
                dbc.Label("Dimension Reduction Algorithm", className="mr-2"),
                dcc.Dropdown(
                    id="algo-dropdown",
                    options=[
                        {"label": entry, "value": entry} for entry in ALGORITHM_DATABASE
                    ],
                    style={"min-width": "250px"},
                    value="PCA",
                ),
                html.Div(id="additional-model-params"),
                html.Hr(),
                html.Div(
                    [
                        dbc.Label("Name your job", className="mr-2"),
                        dcc.Input(
                            id="job-name",
                            placeholder="test0",
                            style={
                                "width": "100%",
                                "margin-bottom": "1rem",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Submit",
                            color="secondary",
                            id="run-algo",
                            outline=True,
                            size="lg",
                            className="m-1",
                            style={"width": "50%"},
                        ),
                    ],
                    className="row",
                    style={
                        "align-items": "center",
                        "justify-content": "center",
                    },
                ),
                html.Hr(),
                dbc.Alert(id="job-alert", is_open=False, dismissable=True),
                html.Hr(),
                html.Div(
                    [
                        dbc.Label("Select a job..."),
                        dcc.Dropdown(id="job-selector"),
                    ]
                ),
                html.Div(id="invisible-apply-div"),
            ]
        ),
    ],
)

cluster_algo_panel = dbc.AccordionItem(
    id="clustering-controls",
    children=[
        dbc.CardBody(
            [
                dbc.Label("Algorithm", className="mr-2"),
                dcc.Dropdown(
                    id="cluster-algo-dropdown",
                    options=[
                        {"label": entry, "value": entry}
                        for entry in CLUSTER_ALGORITHM_DATABASE
                    ],
                    style={"min-width": "250px"},
                    value="DBSCAN",
                ),
                html.Div(id="additional-cluster-params"),
                html.Hr(),
                html.Div(
                    [
                        dbc.Button(
                            "Apply",
                            color="secondary",
                            id="run-cluster-algo",
                            outline=True,
                            size="lg",
                            className="m-1",
                            style={"width": "50%"},
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
        ),
    ],
    title="Select Clustering Algorithms",
)


# add alert pop up window
modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Header")),
                dbc.ModalBody("This is the content of the modal", id="modal-body"),
            ],
            id="modal",
            is_open=False,
        ),
    ]
)

control_panel = dbc.Accordion(
    children=[
        algo_panel,
        cluster_algo_panel,
        heatmap_control_panel,
        scatter_control_panel,
    ],
    style={"position": "sticky", "top": "10%", "width": "100%"},
)


# metadata
meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created contents
            dcc.Store(id="image-length", data=0),
            dcc.Store(id="user-upload-data-dir", data=None),
            dcc.Store(id="dataset-options", data=EXAMPLE_DATASETS),
            dcc.Store(id="run-counter", data=0),
            dcc.Store(id="experiment-id", data=None),
            # data_label_schema, latent vectors, clusters
            dcc.Store(id="input_labels", data=None),
            dcc.Store(id="label_schema", data=None),
            dcc.Store(id="model_id", data=None),
            dcc.Store(id="latent_vectors", data=None),
            dcc.Store(id="clusters", data=None),
        ],
    )
]


# DEFINE LAYOUT
app.layout = html.Div(
    [
        header,
        dbc.Container(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            control_panel,
                            width=4,
                            style={"display": "flex", "margin-top": "1em"},
                        ),
                        dbc.Col(image_panel, width=8),
                    ]
                ),
                dbc.Row(dbc.Col(modal)),
                dbc.Row(dbc.Col(meta)),
            ],
            fluid=True,
        ),
        WebSocket(id="ws-live", url="ws://127.0.0.1:8765"),
    ],
)
