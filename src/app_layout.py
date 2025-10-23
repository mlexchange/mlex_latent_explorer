import json
import os

import dash_bootstrap_components as dbc
import diskcache
from dash import Dash, dcc, html
from dash.long_callback import DiskcacheLongCallbackManager
from dotenv import load_dotenv
from file_manager.main import FileManager
from mlex_utils.dash_utils.mlex_components import MLExComponents

from src.components.header import header
from src.components.infrastructure import create_infra_state_affix
from src.components.main_display import main_display
from src.components.sidebar import sidebar
from src.utils.model_utils import Models

load_dotenv(".env")

READ_DIR = os.getenv("READ_DIR")
WRITE_DIR = os.getenv("WRITE_DIR")
DATA_TILED_KEY = os.getenv("DATA_TILED_KEY", None)
if DATA_TILED_KEY == "":
    DATA_TILED_KEY = None

LIVE_TILED_API_KEY = os.getenv("LIVE_TILED_API_KEY", None)
if LIVE_TILED_API_KEY == "":
    LIVE_TILED_API_KEY = None
    
MODE = os.getenv("MODE", "dev")
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["latent-space-explorer"]'))
USER = os.getenv("USER")
NUM_IMGS_OVERVIEW = 6


# SETUP DASH APP
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    "../assets/lse-style.css",
]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    long_callback_manager=long_callback_manager,
)

app.title = "Latent Space Explorer"
app._favicon = "mlex.ico"

server = app.server

dash_file_explorer = FileManager(
    READ_DIR,
    open_explorer=False,
    api_key=DATA_TILED_KEY,
)
dash_file_explorer.init_callbacks(app)
file_explorer = dash_file_explorer.file_explorer

# GET MODELS
latent_space_models = Models(
    modelfile_path="./src/assets/default_models.json", model_type="latent_space_extraction"
)

dim_reduction_models = Models(
    modelfile_path="./src/assets/default_models.json", model_type="dimension_reduction"
)
clustering_models = Models(
    modelfile_path="./src/assets/default_models.json", model_type="clustering"
)

# SETUP MLEx COMPONENTS
mlex_components = MLExComponents("dbc")

job_manager = mlex_components.get_job_manager_minimal(
    model_list=dim_reduction_models.modelname_list,
    mode=MODE,
    aio_id="latent-space-jobs",
    prefect_tags=PREFECT_TAGS + ["latent-space"],
)

clustering_job_manager = mlex_components.get_job_manager_minimal(
    model_list=clustering_models.modelname_list,
    mode=MODE,
    aio_id="clustering-jobs",
    prefect_tags=PREFECT_TAGS + ["clustering"],
    dependency_id={
        "component": "DbcJobManagerAIO",
        "subcomponent": "run-dropdown",
        "aio_id": "latent-space-jobs",
    },
)

# BEGIN DASH CODE

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

# metadata
meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created contents
            dcc.Store(id="image-length", data=0),
            dcc.Store(id="user-upload-data-dir", data=None),
            dcc.Store(id="run-counter", data=0),
            dcc.Store(id="experiment-id", data=None),
            # data_label_schema, latent vectors, clusters
            dcc.Store(id="input_labels", data=None),
            dcc.Store(id="label_schema", data=None),
            dcc.Store(id="model_id", data=None),
            dcc.Store(id="latent_vectors", data=None),
            dcc.Store(id="clusters", data=None),
            dcc.Store(id="live-mode-canceled", data=False),
            # Add model transition tracker
            dcc.Store(id="in-model-transition", data=False),
        ],
    )
]


# DEFINE LAYOUT
app.layout = html.Div(
    [
        # Add a separate overlay div for the loading indicator
        html.Div(
            [
                html.Div(
                    [
                        dbc.Spinner(
                            size="lg",
                            color="white",
                            spinner_style={"width": "4rem", "height": "4rem"},
                        ),
                        html.H2("Loading Models...", id="model-loading-spinner-text", className="mt-3")
                    ],
                    style={
                        "position": "absolute",
                        "top": "50%",
                        "left": "50%",
                        "transform": "translate(-50%, -50%)",
                        "textAlign": "center",
                        "color": "white",
                        "zIndex": 9999
                    }
                )
            ],
            id="model-loading-spinner",
            style={
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "100%",
                "backgroundColor": "rgba(0, 0, 0, 0.7)",
                "zIndex": 9998,
                "display": "none"
            }
        ),
        header(
            "MLExchange | Latent Space Explorer",
            "https://github.com/mlexchange/mlex_latent_explorer",
        ),
        dbc.Container(
            children=[
                sidebar(
                    file_explorer,
                    job_manager,
                    clustering_job_manager,
                ),
                dbc.Row(main_display()),
                dbc.Row(dbc.Col(modal)),
                dbc.Row(dbc.Col(meta)),
                create_infra_state_affix(),
            ],
            fluid=True,
        ),
    ],
)
