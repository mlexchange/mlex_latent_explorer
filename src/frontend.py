from dash import html, dcc , Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import DBSCAN

import os
import io
import pathlib
import re
import json


from new_app_layout import app
from latentxp_utils import hex_to_rgba, generate_scatter_data
import ids
from latentxp_utils import generate_cluster_dropdown_options
from dimension_reduction import computePCA, computeUMAP

#### GLOBAL PARAMS ####
OUTPUT_DIR = pathlib.Path('data/output') # save the latent vectors
USER = 'mlexchange-team'
UPLOAD_FOLDER_ROOT = "data/upload"

@app.callback(        
    Output('additional-algo-params', 'children'),
    Input('algo-dropdown', 'value')
)
def update_algo_parameters(selected_algo):
    # if PCA, do nothig
    if selected_algo == 'PCA':
        return [
                dbc.FormGroup([
                    dbc.Label('Number of Components', className='mr-3'),
                    dcc.Dropdown(id='ncomponents-dropdown',
                                    options=[
                                        {'label': '2 components', 'value': '2'},
                                        {'label': '3 components', 'value': '3'},
                                    ],
                                    value='2',
                                    style={'min-width': '250px'},
                                    ),
                    dbc.Label('Min distance between points', id='invisible1', className='mr-3'),
                    dcc.Dropdown(id='mindist-dropdown',
                                    options=[{'label': str(round(0.1*i, 1)), 'value': str(round(0.1*i, 1))} for i in range(1,10)],
                                    value='0.1',
                                    style={'min-width': '250px', 'display': 'none'},
                                    ),
                    dbc.Label('Number of Nearest Neighbors', id='invisible2', className='mr-3'),
                    dcc.Dropdown(id='nneighbors-dropdown',
                                    options=[{'label': str(i), 'value': str(i)} for i in range(5, 51, 5)],
                                    value='15',
                                    style={'min-width': '250px', 'display': 'none'},
                                    ),
                ])]
    
    if selected_algo == 'UMAP':
        return [dbc.FormGroup(
                    [   
                        dbc.Label('Number of Components', className='mr-3'),
                        dcc.Dropdown(id='ncomponents-dropdown',
                                        options=[
                                            {'label': '2 components', 'value': '2'},
                                            {'label': '3 components', 'value': '3'},
                                        ],
                                        value='2',
                                        style={'min-width': '250px'},
                                        ),
                        dbc.Label('Min distance between points', className='mr-3'),
                        dcc.Dropdown(id='mindist-dropdown',
                                        options=[{'label': str(round(0.1*i, 1)), 'value': str(round(0.1*i, 1))} for i in range(1,10)],
                                        value='0.1',
                                        style={'min-width': '250px'},
                                        ),
                        dbc.Label('Number of Nearest Neighbors', className='mr-3'),
                        dcc.Dropdown(id='nneighbors-dropdown',
                                        options=[{'label': str(i), 'value': str(i)} for i in range(5, 51, 5)],
                                        value='15',
                                        style={'min-width': '250px' },
                                        ),
                    ],
                )]
        
@app.callback(
    Output('input_data', 'data'),
    Output('input_labels', 'data'),
    Output('label_schema', 'data'),
    Input('dataset-selection', 'value')
)
def update_label_schema(selected_dataset):
    data = None
    labels = None
    label_schema = None 

    if selected_dataset == "data/Demoshapes.npz":
        data = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/Demoshapes.npz")['arr_0']
        labels = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/DemoLabels.npy")  
        f = open("/Users/runbojiang/Desktop/mlex_latent_explorer/data/label_schema.json") #"/app/work/data/label_schema.json"
        label_schema = json.load(f)

    return data, labels, label_schema

@app.callback(
    [
        Output('latent_vectors', 'data'),
        Output('clusters', 'data'),
    ],
    Input('run-algo', 'n_clicks'),
    [
        State('input_data', 'data'),
        State('algo-dropdown', 'value'),
        State('ncomponents-dropdown', 'value'),
        State('mindist-dropdown', 'value'),
        State('nneighbors-dropdown', 'value')
    ]
)
def update_latent_vectors_and_clusters(submit_n_clicks, 
                                       input_data, selected_algo, n_components, min_dist, n_neighbors):
    print(selected_algo)
    input_data = np.array(input_data)
    if (submit_n_clicks is None) or (input_data is None):
        raise PreventUpdate
    
    if selected_algo == 'PCA':
        latent_vectors = computePCA(input_data, n_components=int(n_components))
    if selected_algo == 'UMAP':
        latent_vectors = computeUMAP(input_data, n_components=int(n_components), n_neighbors=int(n_neighbors), min_dist=float(min_dist))
    
    clusters = None
    if latent_vectors is not None:
        obj = DBSCAN(eps=1.70, min_samples=1, leaf_size=5)
        clusters = obj.fit_predict(latent_vectors)
        print("len clusters", len(clusters))
    
    return latent_vectors, clusters

@app.callback(
    Output('scatter', 'figure'),
    [
        Input('latent_vectors', 'data'),
    ],
    [
        State('ncomponents-dropdown', 'value'),
        State('clusters', 'data')
    ]
    
)
def update_scatter_plot(latent_vectors, n_components, clusters):
    if latent_vectors is None:
        raise PreventUpdate
    print('update scatter')
    
    latent_vectors = np.array(latent_vectors)    
    if n_components == '2':
        print("lv shape", latent_vectors.shape)
        traces = []
        trace_x = latent_vectors[:, 0].tolist()
        trace_y = latent_vectors[:, 1].tolist()
        traces.append(
            go.Scattergl(
                x = trace_x,
                y = trace_y,
                mode = 'markers'
            )
        )
        fig = go.Figure(data = traces)
        return fig

    else:
        return None
    

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8070, )


