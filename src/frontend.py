import dash
from dash import html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN, HDBSCAN
import pathlib
import json
import uuid
import requests
import os
import requests

from file_manager.data_project import DataProject

from app_layout import app, DOCKER_DATA, UPLOAD_FOLDER_ROOT
from latentxp_utils import kmeans_kwargs, dbscan_kwargs, hdbscan_kwargs, hex_to_rgba, generate_scatter_data, remove_key_from_dict_list, get_content, get_trained_models_list, load_images_by_indices
from dash_component_editor import JSONParameterEditor


#### GLOBAL PARAMS ####
DATA_DIR = str(os.environ['DATA_DIR'])
OUTPUT_DIR = pathlib.Path('data/mlexchange_store/admin')
USER = 'admin' #'mlexchange-team' # move to env file
UPLOAD_FOLDER_ROOT = "data/upload"

@app.callback(
    Output('additional-model-params', 'children'),
    Output('model_id', 'data'),
    Input('algo-dropdown', 'value')
)
def show_dimension_reduction_gui_layouts(selected_algo):
    '''
    This callback display dropdown menu in the frontend for different dimension reduction algos
    Args:
        selected_algo:      Selected dimension reduction algorithm
    Returns:
        item_list:          dropdown menu html code
        model_uid:          selected algo's uid
    '''
    data = requests.get('http://content-api:8000/api/v0/models').json() # all model
   
    if selected_algo == 'PCA':
        conditions = {'name': 'PCA'}
    elif selected_algo == 'UMAP':
        conditions = {'name': 'UMAP'}
    
    model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())] # filter pca or umap
    model_uid = model[0]['content_id']
    new_model = remove_key_from_dict_list(model[0]["gui_parameters"], 'comp_group')

    item_list = JSONParameterEditor(_id={'type': str(uuid.uuid4())},
                                    json_blob=new_model,
    )
    item_list.init_callbacks(app)
        
    return item_list, model_uid

@app.callback(
    Output('additional-cluster-params', 'children'),
    Input('cluster-algo-dropdown', 'value'),
)
def show_clustering_gui_layouts(selected_algo):
    '''
    This callback display drop down menu in the fronend  for different clustering algos
    Args:
        selected_algo:      selected clustering algorithm
    Returns:
        item_list:          dropdown menu html code
    '''
    if selected_algo == 'KMeans':
        kwargs = kmeans_kwargs
    elif selected_algo == 'DBSCAN':
        kwargs = dbscan_kwargs
    elif selected_algo == 'HDBSCAN':
        kwargs = hdbscan_kwargs
    
    item_list = JSONParameterEditor(_id={'type': str(uuid.uuid4())},
                                    json_blob=kwargs["gui_parameters"])
    item_list.init_callbacks(app)
    return item_list

@app.callback(
    [
        Output('input_labels', 'data'),
        Output('label_schema', 'data'),
        Output('label-dropdown', 'options'),
        Output('user-upload-data-dir', 'data'),
    ],
    [
        Input('example-dataset-selection', 'value'),                            # example dataset
        Input({'base_id': 'file-manager', 'name': 'docker-file-paths'},'data'), # FM dataset
        Input('feature-vector-model-list', 'value'),                            # data clinic dataset
    ]
)
def update_data_n_label_schema(selected_example_dataset, upload_file_paths, data_clinic_file_path):
    '''
    This callback updates the selected dataset from the provided example datasets, as well as labels, and label schema
    Args:
        example-dataset-selection:      selected dataset from the provided example datasets, not the one that user uploaded
        upload_file_pahts:      Data project info, the user uploaded zip file using FileManager, list
    Returns:
        input_data:             input image data, numpy.ndarray
        input_labels:           labels of input image data, which is of int values
        label_schema:           the text of each unique label
        label_dropdown:         label dropdown options
        user_upload_data_dir:   dir name for the user uploaded zip file
    '''
    labels = None
    label_schema = {}

    # check if user is using user uploaded zip file or example dataset or data clinic file
    # priority level: FileManage > DataClinic > Example Datasets

    # FileManager - user uploaded zip file of images
    data_project = DataProject()
    data_project.init_from_dict(upload_file_paths)
    data_set = data_project.data # list of len 1920, each element is a local_dataset.LocalDataset object
    options = []
    user_upload_data_dir = None
    if len(data_set) > 0:
        labels = np.full((len(data_set),), -1)
        user_upload_data_dir = os.path.dirname(upload_file_paths[0]['uri'])
    # DataClinic options
    elif data_clinic_file_path is not None:
        df = pd.read_parquet(data_clinic_file_path)
        # data = df.values
        labels = np.full((df.shape[0],), -1)
    # Example dataset option 1
    elif selected_example_dataset == "data/example_shapes/Demoshapes.npz":
        # data = np.load("/app/work/" + selected_example_dataset)['arr_0']
        labels = np.load("/app/work/data/example_shapes/DemoLabels.npy")
        f = open("/app/work/data/example_shapes/label_schema.json")
        label_schema = json.load(f)
    # Example dataset option 2
    elif selected_example_dataset == "data/example_latentrepresentation/f_vectors.parquet":
        df = pd.read_parquet("/app/work/" + selected_example_dataset)
        # data = df.values
        labels = np.full((df.shape[0],), -1)


    if label_schema: 
        options = [{'label': f'Label {label}', 'value': label} for label in label_schema]
    options.insert(0, {'label': 'Unlabeled', 'value': -1})
    options.insert(0, {'label': 'All', 'value': -2})

    return labels, label_schema, options, user_upload_data_dir

def job_content_dict(content):
    job_content = {# 'mlex_app': content['name'],
                   'mlex_app': 'latent_space_explorer',
                   'service_type': content['service_type'],
                   'working_directory': DATA_DIR,
                   'job_kwargs': {'uri': content['uri'], 
                                  'cmd': content['cmd'][0]}
    }
    if 'map' in content:
        job_content['job_kwargs']['map'] = content['map']
    
    return job_content

@app.callback(
    [
        # flag the read variable
        Output('experiment-id', 'data'),
        # reset scatter plot control panel
        Output('scatter-color',  'value'),
        Output('cluster-dropdown', 'value'),
        Output('label-dropdown', 'value'),
        # reset heatmap
        Output('heatmap', 'figure', allow_duplicate=True),
        # reset interval value to
        Output('interval-component', 'max_intervals'),
    ],
    Input('run-algo', 'n_clicks'),
    [
        State('example-dataset-selection', 'value'), # 2 example dataset
        State('user-upload-data-dir', 'data'),       # FM
        State('feature-vector-model-list', 'value'), # DataClinic
        State('model_id', 'data'),
        State('algo-dropdown', 'value'),
        State('additional-model-params', 'children'),
    ],
    prevent_initial_call=True
)
def submit_dimension_reduction_job(submit_n_clicks,
                                   selected_example_dataset, user_upload_data_dir, data_clinic_file_path, 
                                   model_id, selected_algo, children):
    """
    This callback is triggered every time the Submit button is hit:
        - compute latent vectors, which will be saved in data/output/experiment_id
        - reset scatter plot control panel to default
        - reset heatmap to no image
    Args:
        submit_n_clicks:        num of clicks for the submit button
        selected_example_dataset:       selected example dataset
        user_upload_data_dir:   user uploaded dataset
        model_id:               uid of selected dimension reduciton algo
        selected_algo:          selected dimension reduction algo
        children:               div for algo's parameters
    Returns:
        experiment-id:          uuid for current run
        cluster-dropdown:       options for cluster dropdown
        scatter-color:          default scatter-color value
        cluster-dropdown:       default cluster-dropdown value
        heatmap:                empty heatmap figure
        interval:               set interval component to trigger to find the latent_vectors.npy file (-1)
    """
    if not submit_n_clicks:
        raise PreventUpdate
    if not selected_example_dataset and not user_upload_data_dir and not data_clinic_file_path:
        raise PreventUpdate

    input_params = {}
    if children:
        for child in children['props']['children']:
            key   = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value
    print("Dimension reduction algo params: ", input_params)
    model_content = get_content(model_id)
    print(model_content)
    job_content = job_content_dict(model_content)
    job_content['job_kwargs']['kwargs'] = {}
    job_content['job_kwargs']['kwargs']['parameters'] = input_params
    #TODO: other kwargs

    compute_dict = {'user_uid': USER,
                    'host_list': ['mlsandbox.als.lbl.gov', 'local.als.lbl.gov', 'vaughan.als.lbl.gov'],
                    'requirements': {'num_processors': 2,
                                     'num_gpus': 0,
                                     'num_nodes': 2},
                    }
    compute_dict['job_list'] = [job_content]
    compute_dict['dependencies'] = {'0':[]}
    compute_dict['requirements']['num_nodes'] = 1

    # create user directory to store users data/experiments
    experiment_id = str(uuid.uuid4())  # create unique id for experiment
    output_path = OUTPUT_DIR / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)

    # check if user is using user uploaded zip file or example dataset or data clinic file 
    if user_upload_data_dir is not None:
        selected_dataset = user_upload_data_dir
    elif data_clinic_file_path is not None:
        selected_dataset = data_clinic_file_path
    else:
        selected_dataset = selected_example_dataset
    
    # check which dimension reduction algo, then compose command
    if selected_algo == 'PCA':
        cmd_list = ["python pca_run.py", selected_dataset, str(output_path)]
    elif selected_algo == 'UMAP':
        cmd_list = ["python umap_run.py", selected_dataset, str(output_path)]
        
    docker_cmd = " ".join(cmd_list)
    #print(docker_cmd)
    docker_cmd = docker_cmd + ' \'' + json.dumps(input_params) + '\''
    #print(docker_cmd)
    job_content['job_kwargs']['cmd'] = docker_cmd

    response = requests.post('http://job-service:8080/api/v0/workflows', json=compute_dict)
    print("respnse: ", response)
    # job_response = get_job(user=None, mlex_app=job_content['mlex_app'])
    
    return experiment_id, 'cluster', -1, -2, go.Figure(go.Heatmap()), -1

@app.callback(
    [   
        Output('latent_vectors', 'data'),
        Output('interval-component', 'max_intervals', allow_duplicate=True),
    ],
    Input('interval-component', 'n_intervals'),
    State('experiment-id', 'data'),
    State('interval-component', 'max_intervals'),
    prevent_initial_call=True
)
def read_latent_vectors(n_intervals, experiment_id, max_intervals):
    """
    This callback is trigged by the interval:
        - read latent vectors
        - set interval to not trigger (0)
    Args:
        n_intervals:            interval component
        experiment-id:          each run/submit has a unique experiment id
    Returns:
        latent_vectors:         data from dimension reduction algos
        scatter_fig:            scatter plot the latent vectors (no cluster info yet)
        max_intervals:          interval component that controls if trigger the interval indefintely
    """
    if experiment_id is None or n_intervals == 0 or max_intervals == 0:
        raise PreventUpdate

    #read the latent vectors from the output dir
    output_path = OUTPUT_DIR / experiment_id
    npz_files = list(output_path.glob('*.npy'))
    if len(npz_files) > 0 :
        lv_filepath = npz_files[0] # latent vector file path
        latent_vectors = np.load(str(lv_filepath))
        print("latent vector", latent_vectors.shape)
        return latent_vectors, 0
    else:
        return None, -1
    
@app.callback(
    [
        Output('clusters', 'data'),
        Output('cluster-dropdown', 'options'),
    ],
    Input('run-cluster-algo', 'n_clicks'),
    [
        State('latent_vectors', 'data'),
        State('cluster-algo-dropdown', 'value'),
        State('additional-cluster-params', 'children'),
        State('experiment-id', 'data'),
    ]
)
def apply_clustering(apply_n_clicks, 
                     latent_vectors, selected_algo, children, experiment_id):
    """
    This callback is triggered by click the 'Apply' button at the clustering panel:
        - apply cluster
        - save cluster array to npy file
    Args:
        apply_n_clicks:         num of clicks for the apply button
        latent_vectors:         latent vectors from the dimension reduction algo
        selected_algo:          selected clustering algo
        children:               div for clustering algo's parameters
        experiment_id:          current experiment id, keep track to save the clustering.npy
    Returns:
        clusters:               clustering result for each data point
    """
    ## TODO: pop up a widow to ask user to first run diemnsion reduction then apply
    if apply_n_clicks == 0 or experiment_id is None:
        raise PreventUpdate
    latent_vectors = np.array(latent_vectors)

    input_params = {}
    if children:
        for child in children['props']['children']:
            key   = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value
    print("Clustering params:", input_params)
 
    if selected_algo == "KMeans":
        obj = MiniBatchKMeans(n_clusters=input_params['n_clusters'])
    elif selected_algo == "DBSCAN":
        obj = DBSCAN(eps=input_params['eps'], min_samples=input_params['min_samples'])
    elif selected_algo == "HDBSCAN":
        obj = HDBSCAN(min_cluster_size=input_params['min_cluster_size'])

    clusters, options = None, None
    if obj:
        clusters = obj.fit_predict(latent_vectors)
        output_path = OUTPUT_DIR / experiment_id
        np.save(output_path/'clusters.npy', clusters)
        unique_clusters = np.unique(clusters)
        options = [{'label': f'Cluster {cluster}', 'value': cluster} for cluster in unique_clusters if cluster != -1]
        options.insert(0, {'label': 'All', 'value': -1})

    return clusters, options

@app.callback(
    Output('scatter', 'figure'),
    [
        Input('latent_vectors', 'data'),
        Input('cluster-dropdown', 'value'),
        Input('label-dropdown', 'value'),
        Input('scatter-color', 'value'),
        Input('clusters', 'data'), #move clusters to the input
    ],
    [
        State('scatter', 'figure'),
        State('scatter', 'selectedData'),
        State('additional-model-params', 'children'),

        State('input_labels', 'data'),
        State('label_schema', 'data'),
    ]
)
def update_scatter_plot(latent_vectors, selected_cluster, selected_label, scatter_color, clusters,
                        current_figure, selected_data, children, labels, label_names):
    '''
    This callback update the scater plot
    Args:
        latent_vectors:     data from dimension reduction algos
        selected_cluster:   selected cluster option from dropdown
        selected_label:     selected label option from dropdown
        scatter_color:      selected scatter-color option, either cluster or label
        current_figure:     current scatter figure
        selected_data:      lasso or rect selected data points on scatter figure
        children:           div for algo's parameters
        clusters:           clusters for latent vectors
        labels:             labels of input image data, which is of int values
        label_names:        same as label_schema defined earlier
    Returns:
        fig:                updated scatter figure
    '''
    if latent_vectors is None or children is None:
        raise PreventUpdate
    latent_vectors = np.array(latent_vectors)
    print("latent vector shape:", latent_vectors.shape)

    n_components = children['props']['children'][0]["props"]["children"][1]["props"]["value"]

    # if selected_data is not None and len(selected_data.get('points', [])) > 0:
    #     selected_indices = [point['customdata'][0] for point in selected_data['points']]
    if selected_data is not None and len(selected_data.get('points', [])) > 0:
        selected_indices = []
        for point in selected_data['points']:
            if 'customdata' in point and len(point['customdata']):
                selected_indices.append(point['customdata'][0])
        print("selected indices: ", selected_indices)
    else:
        selected_indices = None
    
    if not clusters: # when clusters is None, i.e., after submit dimension reduction but before apply clustering
        clusters = [-1 for i in range(latent_vectors.shape[0])]
    cluster_names = {a: a for a in np.unique(clusters).astype(int)}
    
    scatter_data = generate_scatter_data(latent_vectors,
                                        n_components,
                                        selected_cluster,
                                        clusters,
                                        cluster_names,
                                        selected_label,
                                        labels,
                                        label_names,
                                        scatter_color)

    fig = go.Figure(scatter_data)
    fig.update_layout(legend=dict(tracegroupgap=20))

    if current_figure and 'xaxis' in current_figure['layout'] and 'yaxis' in current_figure[
        'layout'] and 'autorange' in current_figure['layout']['xaxis'] and current_figure['layout']['xaxis'][
        'autorange'] is False:
        # Update the axis range with current figure's values if available and if autorange is False
        fig.update_xaxes(range=current_figure['layout']['xaxis']['range'])
        fig.update_yaxes(range=current_figure['layout']['yaxis']['range'])
    else:
        # If it's the initial figure or autorange is True, set autorange to True to fit all points in view
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

    if selected_indices is not None:
        # Use the selected indices to highlight the selected points in the updated figure
        for trace in fig.data:
            if trace.marker.color is not None:
                trace.marker.color = [hex_to_rgba('grey', 0.3) if i not in selected_indices else 'red' for i in
                                      range(len(trace.marker.color))]
    return fig

@app.callback(
    Output('heatmap', 'figure', allow_duplicate=True),
    [
        Input('scatter', 'clickData'),
        Input('scatter', 'selectedData'),
        Input('mean-std-toggle', 'value'),
    ],
    [
        State('example-dataset-selection', 'value'),                            # example dataset
        State({'base_id': 'file-manager', 'name': 'docker-file-paths'},'data'), # FM dataset
        State('feature-vector-model-list', 'value'),                            # data clinic dataset

    ],
    prevent_initial_call=True

)
def update_heatmap(click_data, selected_data, display_option,
                   selected_example_dataset, upload_file_paths, data_clinic_file_path):
    '''
    This callback update the heatmap
    Args:
        click_data:         clicked data on scatter figure
        selected_data:      lasso or rect selected data points on scatter figure
        display_option:     option to display mean or std
    Returns:
        fig:                updated heatmap
    '''
    if not selected_example_dataset and not upload_file_paths and not data_clinic_file_path:
        raise PreventUpdate
    
    # user select a group of points
    if selected_data is not None and len(selected_data['points']) > 0:
        selected_indices = [point['customdata'][0] for point in selected_data['points']]  # Access customdata for the original indices
        print("selected_indices", selected_indices)
        
        ### FileManager
        # print("upload_file_paths") # if not selected, its an empty list not None
        selected_images = []
        data_project = DataProject()
        data_project.init_from_dict(upload_file_paths)
        data_set = data_project.data
        if len(data_set) > 0:
            print("FM file")
            for i in selected_indices:
                image, uri = data_project.data[i].read_data(export='pillow')
                selected_images.append(np.array(image))
        ### DataClinic
        elif data_clinic_file_path is not None:
            print("data_clinic_file_path")
            print(data_clinic_file_path)
            directory_path = os.path.dirname(data_clinic_file_path)
            selected_images = load_images_by_indices(directory_path, selected_indices)

        ### Example dataset
        elif selected_example_dataset == "data/example_shapes/Demoshapes.npz":
            print("Demoshapes.npz")
            selected_images = np.load("/app/work/" + selected_example_dataset)['arr_0'][selected_indices]
            print(selected_images.shape)
        elif selected_example_dataset == "data/example_latentrepresentation/f_vectors.parquet":
            print("f_vectors.parque")
            df = pd.read_parquet("/app/work/" + selected_example_dataset)
            selected_images = df.iloc[selected_indices].values
        selected_images = np.array(selected_images)

        print("selected_images shape:", selected_images.shape)
        
        # display options
        if display_option == 'mean':
            heatmap_data = go.Heatmap(z=np.mean(selected_images, axis=0))
        elif display_option == 'sigma':
            heatmap_data = go.Heatmap(z=np.std(selected_images, axis=0))

    elif click_data is not None and len(click_data['points']) > 0:
        selected_index = click_data['points'][0]['customdata'][0]  # click_data['points'][0]['pointIndex']
        ### FileManager
        data_project = DataProject()
        data_project.init_from_dict(upload_file_paths)
        data_set = data_project.data
        if len(data_set) > 0:
            clicked_image, uri = data_project.data[selected_index].read_data(export='pillow')
        ### DataClinic
        elif data_clinic_file_path is not None:
            directory_path = os.path.dirname(data_clinic_file_path)
            clicked_image = load_images_by_indices(directory_path, [selected_index])
        ### Example dataset
        elif selected_example_dataset == "data/example_shapes/Demoshapes.npz":
            clicked_image = np.load("/app/work/" + selected_example_dataset)['arr_0'][selected_index]
        elif selected_example_dataset == "data/example_latentrepresentation/f_vectors.parquet":
            df = pd.read_parquet("/app/work/" + selected_example_dataset)
            clicked_image = df.iloc[selected_index].values
        clicked_image = np.array(clicked_image)
        
        heatmap_data = go.Heatmap(z=clicked_image)

    else:
        heatmap_data = go.Heatmap()

    # only update heat map when the input data is 2d images, do not update for input latent vectors
    if heatmap_data['z'] is None or len(np.shape(heatmap_data['z'])) < 2:
        raise PreventUpdate
    
    # Determine the aspect ratio based on the shape of the heatmap_data's z-values
    aspect_x = 1
    aspect_y = 1
    if heatmap_data['z'] is not None:
        if heatmap_data['z'].size > 0:
            print(np.shape(heatmap_data['z']))
            aspect_y, aspect_x = np.shape(heatmap_data['z'])[-2:]

    return go.Figure(
        data=heatmap_data,
        layout=dict(
            autosize=True,
            yaxis=dict(scaleanchor="x", scaleratio=aspect_y / aspect_x),
        )
    )

@app.callback(
    Output('stats-div', 'children'),
    Input('scatter', 'selectedData'),
    [
        State('clusters', 'data'),
        State('input_labels', 'data'),
        State('label_schema', 'data'),
    ]
)
def update_statistics(selected_data, clusters, assigned_labels, label_names):
    '''
    This callback update the statistics panel
    Args:
        selected_data:      lasso or rect selected data points on scatter figure
        clusters:           clusters for latent vectors
        assigned_labels:    labels for each latent vector
        label_names:        same as label schema  
    Returns:
        [num_images, clusters, labels]:     statistics
    '''
    
    assigned_labels = np.array(assigned_labels)

    if selected_data is not None and len(selected_data['points']) > 0:
        selected_indices = [point['customdata'][0] for point in
                            selected_data['points']]  # Access customdata for the original indices
        selected_clusters = []
        if clusters is not None:
            clusters = np.array(clusters)
            selected_clusters = clusters[selected_indices]
        selected_labels = assigned_labels[selected_indices]

        num_images = len(selected_indices)
        unique_clusters = np.unique(selected_clusters)
        unique_labels = np.unique(selected_labels)

        # Format the clusters and labels as comma-separated strings
        clusters_str = ", ".join(str(cluster) for cluster in unique_clusters)
        label_int_to_str_map = {val: key for key, val in label_names.items()}
        labels_str = ", ".join(str(label_int_to_str_map[label]) for label in unique_labels if label >= 0)
    else:
        num_images = 0
        clusters_str = "N/A"
        labels_str = "N/A"

    return [
        f"Number of images selected: {num_images}",
        html.Br(),
        f"Clusters represented: {clusters_str}",
        html.Br(),
        f"Labels represented: {labels_str}",
    ]

@app.callback(
    [Output("modal", "is_open"), Output("modal-body", "children")],
    [
        Input('run-algo', 'n_clicks'), 
        Input('run-cluster-algo', 'n_clicks'),
    ],
    [
        State("modal", "is_open"), 
        State('example-dataset-selection', 'value'),
        State('user-upload-data-dir', 'data'),
        State('feature-vector-model-list', 'value'),
    ]
)
def toggle_modal(n_submit, n_apply,
                 is_open, selected_example_dataset, user_upload_data_dir, data_clinic_file_path):
    '''
    This callback pop up a window to remind user to follow this flow: 
        select dataset -> Submit dimension reduction job -> Apply clustering
    Args:
        n_submit (int):     Number of clicks on the 'Submit' button.
        n_apply (int):      Number of clicks on the 'Apply' button.
        is_open (bool):     Current state of the modal window (open/closed).
        input_data (list):         User selected data
    Returns:
        is_open (bool):     New state of the modal window.
        modal_body_text (str): Text to be displayed in the modal body.
    '''
    at_least_one_dataset_selected = False
    if selected_example_dataset or user_upload_data_dir or data_clinic_file_path:
        at_least_one_dataset_selected = True
    
    if ((n_submit and not at_least_one_dataset_selected) or
        (n_apply and not at_least_one_dataset_selected)):
        return True, "Please select an example dataset or upload your own zipped dataset or choose DataClinic outpu."
    elif n_apply and n_submit is None:
        return True, "Please select a dimension reduction algorithm and click 'Submit' button before clustering."
            
    return False, "No alert."


@app.callback(
    Output('feature-vector-model-list', 'options'),
    Input('interval-component', 'n_intervals'),
)
def update_trained_model_list(interval):
    '''
    This callback updates the list of trained models
    Args:
        tab_value:                      Tab option
        prob_refresh_n_clicks:          Button to refresh the list of probability-based trained models
        similarity_refresh_n_clicks:    Button to refresh the list of similarity-based trained models
    Returns:
        prob_model_list:                List of trained models in mlcoach
        similarity_model_list:          List of trained models in data clinic and mlcoach
    '''
    data_clinic_models = get_trained_models_list(USER, 'data_clinic')
    ml_coach_models = get_trained_models_list(USER, 'mlcoach')
    feature_vector_models = data_clinic_models + ml_coach_models
    #print(feature_vector_models)

    return feature_vector_models


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8070, )
