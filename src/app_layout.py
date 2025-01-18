import os

import dash_bootstrap_components as dbc
import diskcache
from dash import Dash, dcc, html
from dash.long_callback import DiskcacheLongCallbackManager

# from dash_extensions import WebSocket
from dotenv import load_dotenv
from file_manager.main import FileManager
from mlex_utils.dash_utils.mlex_components import MLExComponents

from src.components.header import header
from src.components.main_display import main_display
from src.components.sidebar import sidebar
from src.utils.model_utils import Models

# from utils_tiled import TiledResults

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
DATA_TILED_KEY = os.getenv("TILED_KEY", None)
MODE = os.getenv("MODE", "dev")
PREFECT_TAGS = os.getenv("PREFECT_TAGS", ["latent-space-explorer"])

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
# tiled_results = TiledResults(RESULT_TILED_URI, RESULT_TILED_API_KEY)
# tiled_results.prep_result_tiled_containers()

# Websocket server
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "127.0.0.1")
WEBSOCKET_PORT = os.getenv("WEBSOCKET_PORT", 8765)

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

dash_file_explorer = FileManager(READ_DIR, open_explorer=False, api_key=DATA_TILED_KEY)
dash_file_explorer.init_callbacks(app)
file_explorer = dash_file_explorer.file_explorer

# GET MODELS
models = Models(modelfile_path="./src/assets/default_models.json")

# SETUP MLEx COMPONENTS
mlex_components = MLExComponents("dbc")
job_manager = mlex_components.get_job_manager(
    model_list=models.modelname_list,
    mode=MODE,
    aio_id="data-clinic-jobs",
    prefect_tags=PREFECT_TAGS,
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
        header(
            "MLExchange | Latent Space Explorer",
            "https://github.com/mlexchange/mlex_latent_explorer",
        ),
        dbc.Container(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            sidebar(file_explorer, job_manager),
                            style={"flex": "0 0 500px"},
                        ),
                        dbc.Col(main_display()),
                    ]
                ),
                dbc.Row(dbc.Col(modal)),
                dbc.Row(dbc.Col(meta)),
            ],
            fluid=True,
        ),
        # WebSocket(id="ws-live", url=f"ws:{WEBSOCKET_URL}:{WEBSOCKET_PORT}"),
    ],
)
