from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np
import json
from sklearn.cluster import DBSCAN


import templates
import ids
from latentxp_utils import generate_cluster_dropdown_options, generate_label_dropdown_options

external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server

#--------------------------------- IO ----------------------------------
f = open("/Users/runbojiang/Desktop/mlex_latent_explorer/data/label_schema.json")
LABEL_NAMES = json.load(f)
#latent_vectors = np.load("/app/work/data/pacmacX.npy")
#latent_vectors = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/pacmacX.npy")
pca_latent_vectors = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/pca.npz")['array']
pca_latent_vectors_3d = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/pca_3d.npz")['array']
umap_latent_vectors = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/umap.npz")['array']
latent_vector_options = {'PCA': pca_latent_vectors, 'UMAP': umap_latent_vectors, 'PCA_3d': pca_latent_vectors_3d}

obj = DBSCAN(eps=1.70, min_samples=1, leaf_size=5)
#clusters = obj.fit_predict(latent_vectors)
pca_clusters = obj.fit_predict(pca_latent_vectors)
pca_clusters_3d = obj.fit_predict(pca_latent_vectors_3d)
umap_clusters = obj.fit_predict(umap_latent_vectors)
cluster_options = {'PCA': pca_clusters, 'UMAP': umap_clusters, 'PCA_3d': pca_clusters_3d}

header = templates.header()
body = html.Div([
    html.Div([
        # tabs
        html.Div([
            dcc.Tabs(id=ids.TABS, value='PCA', children=[
                dcc.Tab(label='PCA', value='PCA'),
                dcc.Tab(label='UMAP', value='UMAP'),
            ]),
        ], className='column'),
        # parameters for dimension reduction methods
        html.Div(id=ids.DR_PARAMETERS, children = [
            html.Label('Select parameters for PCA: '),
            html.Label('Select number of principal components to keep:'),
            dcc.RadioItems(
                id=ids.N_COMPONENTS,
                options=[
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'}
                ],
                value='2'
            ),
        ], className='column'),
        # latent plot
        html.Div([
            dcc.Graph(id=ids.SCATTER,
                      figure=go.Figure(go.Scattergl(mode='markers')),
                      style={'padding-bottom': '5%'}),
        ], className='column', style={'flex': '50%', 'padding': '10px'}),

        # individual image
        html.Div([
            dcc.Graph(id=ids.HEATMAP, figure=go.Figure(go.Heatmap()), style={'padding-bottom': '5%'}),
        ], className='column', style={'flex': '50%', 'padding': '10px'}),

    ], className='row', style={'display': 'flex'}),
    html.Div([
        # control panel
        html.Div([
            # Add controls and human interactions here
            # Example: dcc.Slider(), dcc.Dropdown(), etc.

            # Add a radio button for toggling coloring options
            html.Label('Scatter Colors:'),
            dcc.RadioItems(id=ids.SCATTER_COLOR, options=[{'label': 'cluster', 'value': 'cluster'},
                                                        {'label': 'label', 'value': 'label'}],
                           value='cluster'),
            html.Br(),

            html.Label('Select cluster:'),
            dcc.Dropdown(id=ids.CLUSTER_DROPDOWN,
                         options=generate_cluster_dropdown_options(pca_clusters),
                         value=-1),
            html.Br(),

            html.Label('Select label:'),
            dcc.Dropdown(id=ids.LABEL_DROPDOWN,
                         options=generate_label_dropdown_options(LABEL_NAMES),
                         value=-2),
        ], className='column', style={'flex': '50%', 'padding-bottom': '5%'}),

        # Labeler
        # Add a new div for displaying statistics
        html.Div([
            html.Label([
                            'Select a Group of Points using ',
                            html.Span(html.I(DashIconify(icon="lucide:lasso")), className='icon'),
                            ' or ',
                            html.Span(html.I(DashIconify(icon="lucide:box-select")), className='icon'),
                            ' Tools :'
                        ]),
            html.Br(),
            # Add a radio button for toggling mean and standard deviation
            html.Label('Display Image Options:'),
            dcc.RadioItems(id=ids.MEAN_STD_TOGGLE, options=[{'label': 'Mean', 'value': 'mean'},
                                                          {'label': 'Standard Deviation', 'value': 'sigma'}],
                           value='mean'),
            html.Br(),

            html.Div(id=ids.STATS_DIV, children=[
                html.P("Number of images selected: 0"),
                html.P("Clusters represented: N/A"),
                html.P("Labels represented: N/A"),
            ]),

            html.Label('Assign Label:'),
            dcc.Dropdown(id=ids.LABELER,
                         options=generate_label_dropdown_options(LABEL_NAMES, False),
                         value=-1),

            html.Button('Assign Labels', id='assign-labels-button'),

            html.Div(id='label-assign-output'),

        ], className='column', style={'flex': '50%', 'padding': '10px'}),

    ], className='row', style={'display': 'flex'}),

    # hidden components
    html.Div(id="scatter-update-trigger", style={"display": "none"}),
    dcc.Store(id='scatter-axis-range', storage_type='session'),
    dcc.Store(id='selected-points', storage_type='memory'),
    dcc.Store(id='selected-data-store', data=None),
    html.Script("""
                    document.addEventListener('DOMContentLoaded', function() {
                        document.getElementById('assign-labels-button').onclick = function() {
                            setTimeout(function() {
                                document.getElementById(ids.SCATTER).focus();
                            }, 100);
                        };
                    });
                """)

], style={'display': 'grid', 'gridTemplateRows': '1fr 1fr', 'height': '100vh'})


app.layout = html.Div ([header, body])


