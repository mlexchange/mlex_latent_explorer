from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash import dcc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np
import json
from sklearn.cluster import DBSCAN
import dash_uploader as du

import templates
import ids
from latentxp_utils import generate_cluster_dropdown_options, generate_label_dropdown_options

### GLOBAL VARIABLES
ALGORITHM_DATABASE = {"PCA": "PCA",
                      "UMAP": "UMAP",
                      "tSNE": "tSNE"} ## TODO: update value to compute api link

DATA_OPTION = [
    {"label": "Synthetic Shapes", "value": "data/Demoshapes.npz"}
]

#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server

UPLOAD_FOLDER_ROOT = "data/upload"
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)

#### BEGIN DASH CODE ####
header = templates.header()
# left panel: uploader, scatter plot, individual image  plot
image_panel = [
    dbc.Card(
        id="image-card",
        children=[
            dbc.CardHeader(
                [
                    du.Upload(
                        id='dash-uploader',
                        max_file_size=1800,
                        cancel_button=True,
                        pause_button=True
                    ),
                    dbc.Label('Choose Dataset', className='mr-2'),
                    dcc.Dropdown(
                        id='dataset-selection',
                        options=DATA_OPTION,
                        value = DATA_OPTION[0]['value'],
                        clearable=False,
                        style={'margin-bottom': '1rem'}
                    ),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="scatter",
                    figure=go.Figure(go.Scattergl(mode='markers')),
                        config={
                                "modeBarButtonsToAdd": [
                                "drawrect",
                                "drawopenpath",
                                "eraseshape",
                                ]
                        },
                )
            ),
            ## TODO: dbc.CardFooter for individual image plot
        ]
    )
]

# right panel: choose algorithm, submit job, choose scatter plot attributes, and statistics...
algo_panel = html.Div(
    [dbc.Card(
        id="algo-card",
        style={"width": "100%"},
        children=[
            dbc.Collapse(children=[
                dbc.CardHeader("Dimension Reduction Algorithms"),
                dbc.CardBody(
                    [
                        dbc.Form(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Algorithm", className='mr-2'),
                                        dcc.Dropdown(id="algo-dropdown",
                                                     options=[{"label": entry, "value": entry} for entry in ALGORITHM_DATABASE],
                                                     style={'min-width': '250px'},
                                                     value='PCA',
                                                     ),
                                    ]
                                ),
                                
                                html.Div(id='additional-algo-params',),
                                html.Hr(),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Submit",
                                            id="run-algo",
                                            outline=True,
                                            size="lg",
                                            className="m-1",
                                            style={'width':'50%'}
                                        ),
                                    ],
                                    className='row',
                                    style={'align-items': 'center', 'justify-content': 'center'}
                                )
                            ]
                        )
                    ]
                )
            ],
            id="model-collapse",
            is_open=True,
            style = {'margin-bottom': '0rem'}
            )
        ]
    )
    ]
)

control_panel = [algo_panel] #TODO: add controls for scatter plot and statistics

# metadata
meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created contents
            dcc.Store(id='image-length', data=0),
            dcc.Store(id='uploader-filename', data=[]),
            dcc.Store(id='dataset-options', data=DATA_OPTION),
            dcc.Store(id='run-counter', data=0),
            # data_label_schema, latent vectors, clusters
            dcc.Store(id='input_data', data=None),
            dcc.Store(id='input_labels', data=None),
            dcc.Store(id='label_schema', data=None),
            dcc.Store(id='latent_vectors', data=None),
            dcc.Store(id='clusters', data=[]),
        ],
    )
]


##### DEFINE LAYOUT ####
app.layout = html.Div(
    [
        header, 
        dbc.Container(
            [
                dbc.Row([dbc.Col(image_panel, width=7), dbc.Col(control_panel, width=4)]),
                dbc.Row(dbc.Col(meta)),
            ]
        )
    ]
)
