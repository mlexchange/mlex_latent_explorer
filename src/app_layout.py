from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash import dcc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import dash_uploader as du
import templates

### GLOBAL VARIABLES
ALGORITHM_DATABASE = {"PCA": "PCA",
                      "UMAP": "UMAP",
                      }

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
                )
            ),
            dbc.CardFooter(
                dcc.Graph(
                    id="heatmap",
                    figure=go.Figure(go.Heatmap())
                )
            )
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
                                dbc.Label("Algorithm", className='mr-2'),
                                dcc.Dropdown(id="algo-dropdown",
                                                options=[{"label": entry, "value": entry} for entry in ALGORITHM_DATABASE],
                                                style={'min-width': '250px'},
                                                value='PCA',
                                                ),
                                html.Div(id='additional-model-params'),
                                html.Hr(),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Submit",
                                            color="secondary",
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
            ],
            id="model-collapse",
            is_open=True,
            style = {'margin-bottom': '0rem'}
            )
        ]
    )
    ]
)

scatter_control_panel =  html.Div(
    [dbc.Card(
        style={"width": "100%"},
        children=[
            dbc.CardHeader("Scatter Plot Control Panel"),
            dbc.CardBody([
                        dbc.Label('Scatter Colors', className='mr-3'),
                        dcc.RadioItems(id='scatter-color',
                                        options=[
                                            {'label': 'cluster', 'value': 'cluster'},
                                            {'label': 'label', 'value': 'label'}
                                            ],
                                        value = 'cluster',
                                        style={'min-width': '250px'},
                                        className='mb-2'),
                        dbc.Label("Select cluster", className='mr-3'),
                        dcc.Dropdown(id='cluster-dropdown',
                                        value=-1,
                                        style={'min-width': '250px'},
                                        className='mb-2'),
                        dbc.Label("Select label", className='mr-3'),
                        dcc.Dropdown(id='label-dropdown',
                                        value=-2,
                                        style={'min-width': '250px'},
                                        )
            ])
        ]
    )]
)

heatmap_control_panel =  html.Div(
    [dbc.Card(
        style={"width": "100%"},
        children=[
            dbc.CardHeader("Heatmap Control Panel"),
            dbc.CardBody([ 
                            dbc.Label([
                                    'Select a Group of Points using ',
                                    html.Span(html.I(DashIconify(icon="lucide:lasso")), className='icon'),
                                    ' or ',
                                    html.Span(html.I(DashIconify(icon="lucide:box-select")), className='icon'),
                                    ' Tools :'
                                    ], 
                                    className='mb-3'),
                            dbc.Label('Display Image Options', className='mr-3'),
                            dcc.RadioItems(id='mean-std-toggle',
                                           options=[
                                               {'label': 'Mean', 'value': 'mean'},
                                                {'label': 'Standard Deviation', 'value': 'sigma'}
                                                ],
                                           value = 'mean',
                                           style={'min-width': '250px'},
                                           className='mb-2'),
                            dbc.Label(id='stats-div', children=[
                                   'Number of images selected: 0',
                                   html.Br(),
                                   'Clusters represented: N/A',
                                   html.Br(),
                                   'Labels represented: N/A',
                                ]),
            ])
        ]
    )]
)

control_panel = [algo_panel, scatter_control_panel, heatmap_control_panel] #TODO: add controls for scatter plot and statistics

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
            dcc.Store(id='model_id', data=None),
            dcc.Store(id='latent_vectors', data=None),
            dcc.Store(id='clusters', data=None),
        ],
    )
]


##### DEFINE LAYOUT ####
app.layout = html.Div(
    [
        header, 
        dbc.Container(
            [
                dbc.Row([ dbc.Col(control_panel, width=4), 
                         dbc.Col(image_panel, width=7)
                        ]),
                dbc.Row(dbc.Col(meta)),
            ]
        )
    ]
)
