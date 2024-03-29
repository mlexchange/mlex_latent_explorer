import copy
import json
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
from file_manager.data_project import DataProject
from sklearn.cluster import DBSCAN, HDBSCAN, MiniBatchKMeans

from app_layout import app
from dash_component_editor import JSONParameterEditor
from latentxp_utils import (
    dbscan_kwargs,
    generate_scatter_data,
    hdbscan_kwargs,
    hex_to_rgba,
    kmeans_kwargs,
    remove_key_from_dict_list,
)
from utils_prefect import (
    get_children_flow_run_ids,
    get_flow_runs_by_name,
    schedule_prefect_flow,
)

load_dotenv(".env")

# GLOBAL PARAMS
READ_DIR = os.getenv("READ_DIR", "data")
WRITE_DIR = os.getenv("WRITE_DIR", "mlex_store")

MODEL_DIR = "data/models"
UPLOAD_FOLDER_ROOT = "data/upload"

PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["latent-space-explorer"]'))
TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
FLOW_NAME = os.getenv("FLOW_NAME", "")
FLOW_TYPE = "conda"
CONDA_ENV_NAME = "dimension_reduction_pca"

CONTENT_API_URL = os.getenv("CONTENT_API_URL", "http://localhost:8000/api/v0/models")
DEFAULT_ALGORITHM_DESCRIPTION = os.getenv("DEFAULT_ALGORITHM_DESCRIPTION")


if FLOW_TYPE == "podman":
    TRAIN_PARAMS_EXAMPLE = {
        "flow_type": "podman",
        "params_list": [
            {
                "image_name": "ghcr.io/runboj/mlex_dimension_reduction_pca",
                "image_tag": "main",
                "command": 'python -c \\"import time; time.sleep(30)\\"',
                "params": {
                    "io_parameters": {"uid_save": "uid0001", "uid_retrieve": None}
                },
                "volumes": [
                    f"{READ_DIR}:/app/work/data",
                    f"{WRITE_DIR}:/app/work/mlex_store",
                ],
            }
        ],
    }

    INFERENCE_PARAMS_EXAMPLE = {
        "flow_type": "podman",
        "params_list": [
            {
                "image_name": "ghcr.io/runboj/mlex_dimension_reduction_pca",
                "image_tag": "main",
                "command": 'python -c \\"import time; time.sleep(30)\\"',
                "params": {
                    "io_parameters": {"uid_save": "uid0001", "uid_retrieve": None}
                },
                "volumes": [
                    f"{READ_DIR}:/app/work/data",
                    f"{WRITE_DIR}:/app/work/mlex_store",
                ],
            },
        ],
    }
else:
    TRAIN_PARAMS_EXAMPLE = {
        "flow_type": "conda",
        "params_list": [
            {
                "conda_env_name": f"{CONDA_ENV_NAME}",
                "params": {
                    "io_parameters": {"uid_save": "uid0001", "uid_retrieve": None}
                },
            }
        ],
    }

    INFERENCE_PARAMS_EXAMPLE = {
        "flow_type": "conda",
        "params_list": [
            {
                "conda_env_name": f"{CONDA_ENV_NAME}",
                "params": {
                    "io_parameters": {"uid_save": "uid0001", "uid_retrieve": None}
                },
            },
        ],
    }


@app.callback(
    Output("additional-model-params", "children"),
    Output("model_id", "data"),
    Input("algo-dropdown", "value"),
)
def show_dimension_reduction_gui_layouts(selected_algo):
    """
    This callback display dropdown menu in the frontend for different dimension reduction algos
    Args:
        selected_algo:      Selected dimension reduction algorithm
    Returns:
        item_list:          dropdown menu html code
        model_uid:          selected algo's uid
    """
    try:
        data = requests.get(CONTENT_API_URL).json()  # all model
    except Exception as e:
        print(f"Cannot access content api: {e}", flush=True)
        with open(DEFAULT_ALGORITHM_DESCRIPTION, "r") as f:
            data = [json.load(f)]

    if selected_algo == "PCA":
        conditions = {"name": "PCA"}
    elif selected_algo == "UMAP":
        conditions = {"name": "UMAP"}

    model = [
        d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())
    ]  # filter pca or umap
    model_uid = model[0]["content_id"]
    new_model = remove_key_from_dict_list(model[0]["gui_parameters"], "comp_group")

    item_list = JSONParameterEditor(
        _id={"type": str(uuid.uuid4())},
        json_blob=new_model,
    )
    item_list.init_callbacks(app)

    return item_list, model_uid


@app.callback(
    Output("additional-cluster-params", "children"),
    Input("cluster-algo-dropdown", "value"),
)
def show_clustering_gui_layouts(selected_algo):
    """
    This callback display drop down menu in the fronend  for different clustering algos
    Args:
        selected_algo:      selected clustering algorithm
    Returns:
        item_list:          dropdown menu html code
    """
    if selected_algo == "KMeans":
        kwargs = kmeans_kwargs
    elif selected_algo == "DBSCAN":
        kwargs = dbscan_kwargs
    elif selected_algo == "HDBSCAN":
        kwargs = hdbscan_kwargs

    item_list = JSONParameterEditor(
        _id={"type": str(uuid.uuid4())}, json_blob=kwargs["gui_parameters"]
    )
    item_list.init_callbacks(app)
    return item_list


@app.callback(
    Output("job-selector", "options"),
    Input("interval-component", "n_intervals"),
)
def update_job_selector(n_intervals):
    # TODO: Split train/inference and add data project name
    jobs = get_flow_runs_by_name(tags=PREFECT_TAGS)
    return jobs


@app.callback(
    [
        Output("input_labels", "data"),
        Output("label_schema", "data"),
        Output("label-dropdown", "options"),
    ],
    [
        Input("example-dataset-selection", "value"),  # example dataset
        Input(
            {"base_id": "file-manager", "name": "data-project-dict"}, "data"
        ),  # FM dataset
    ],
)
def update_data_n_label_schema(selected_example_dataset, data_project_dict):
    """
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
    """
    labels = None
    label_schema = {}

    # check if user is using user uploaded zip file or example dataset or data clinic file
    # priority level: FileManage > DataClinic > Example Datasets

    data_project = DataProject.from_dict(data_project_dict)
    options = []
    if len(data_project.datasets) > 0:
        labels = np.full((len(data_project.datasets),), -1)
    # Example dataset option 1
    elif selected_example_dataset == "data/example_shapes/Demoshapes.npz":
        labels = np.load("/app/work/data/example_shapes/DemoLabels.npy")
        f = open("/app/work/data/example_shapes/label_schema.json")
        label_schema = json.load(f)
    # Example dataset option 2
    elif (
        selected_example_dataset
        == "data/example_latentrepresentation/f_vectors.parquet"
    ):
        df = pd.read_parquet("/app/work/" + selected_example_dataset)
        labels = np.full((df.shape[0],), -1)

    if label_schema:
        options = [
            {"label": f"Label {label}", "value": label} for label in label_schema
        ]
    options.insert(0, {"label": "Unlabeled", "value": -1})
    options.insert(0, {"label": "All", "value": -2})

    return labels, label_schema, options


@app.callback(
    [
        # reset scatter plot control panel
        Output("scatter-color", "value"),
        Output("cluster-dropdown", "value"),
        Output("label-dropdown", "value"),
        # reset heatmap
        Output("heatmap", "figure", allow_duplicate=True),
        # reset interval value to
        Output("interval-component", "max_intervals"),
    ],
    Input("run-algo", "n_clicks"),
    [
        State("job-name", "value"),  # job_name
        State("example-dataset-selection", "value"),  # 2 example dataset
        State("feature-vector-model-list", "value"),  # DataClinic
        State("model_id", "data"),
        State("algo-dropdown", "value"),
        State("additional-model-params", "children"),
        State(
            {"base_id": "file-manager", "name": "data-project-dict"}, "data"
        ),  # DataProject for FM
    ],
    prevent_initial_call=True,
)
def submit_dimension_reduction_job(
    submit_n_clicks,
    job_name,
    selected_example_dataset,
    data_clinic_file_path,
    model_id,
    selected_algo,
    children,
    data_project_dict,
):
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
        cluster-dropdown:       options for cluster dropdown
        scatter-color:          default scatter-color value
        cluster-dropdown:       default cluster-dropdown value
        heatmap:                empty heatmap figure
        interval:               set interval component to trigger to find the latent_vectors.npy file (-1)
    """
    if not submit_n_clicks:
        raise PreventUpdate
    if (
        not selected_example_dataset
        and not data_project_dict
        and not data_clinic_file_path
    ):
        raise PreventUpdate

    job_params = job_params = copy.deepcopy(TRAIN_PARAMS_EXAMPLE)
    input_params = {}
    if children:
        for child in children["props"]["children"]:
            key = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value
    print("Dimension reduction algo params: ", input_params, flush=True)

    # check if user is using user uploaded zip file or example dataset or data clinic file
    data_project = DataProject.from_dict(data_project_dict)
    if len(data_project.datasets) > 0:
        print("FM", flush=True)
        data_project = DataProject.from_dict(data_project_dict)
        io_parameters = {
            "data_uris": [dataset.uri for dataset in data_project.datasets],
            "data_tiled_api_key": data_project.api_key,
            "data_type": data_project.data_type,
            "root_uri": data_project.root_uri,
        }

    else:
        print("selected_example_dataset: " + selected_example_dataset, flush=True)
        io_parameters = {
            "data_uris": [selected_example_dataset],
            "data_tiled_api_key": None,
            "data_type": "file",
            "root_uri": None,
        }

    # Autoencoder
    if data_clinic_file_path is not None:
        auto_io_params = io_parameters.copy()
        auto_io_params["model_dir"] = data_clinic_file_path + "/last.ckpt"
        if FLOW_TYPE == "podman":
            autoencoder_params = {
                "image_name": "ghcr.io/mlexchange/mlex_pytorch_autoencoders:main",
                "image_tag": "main",
                "command": "python src/predict_model.py",
                "params": {
                    "io_parameters": auto_io_params,
                    "target_width": 64,
                    "target_height": 64,
                    "batch_size": 32,
                },
                "volumes": [
                    f"{READ_DIR}:/app/work/data",
                    f"{WRITE_DIR}:/app/work/mlex_store",
                ],
            }
        else:
            autoencoder_params = {
                "conda_env_name": "pytorch_autoencoders",
                "params": {
                    "io_parameters": auto_io_params,
                    "target_width": 64,
                    "target_height": 64,
                    "batch_size": 32,
                },
                "python_file_name": "mlex_pytorch_autoencoders/src/predict_model.py",
            }
        job_params["params_list"].insert(0, autoencoder_params)

    # prefect
    current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y/%m/%d %H:%M:%S")
    if not job_name:
        job_name = "test0"
    job_name += " " + str(current_time)
    # TODO: Hash root_uri + data_uris
    project_name = "fake_name"
    print(PREFECT_TAGS, flush=True)

    # check which dimension reduction algo, then compose command
    if selected_algo == "PCA":
        if FLOW_TYPE == "podman":
            job_params["params_list"][-1]["command"] = "python pca_run.py"
        else:
            job_params["params_list"][-1][
                "python_file_name"
            ] = "mlex_dimension_reduction_pca/pca_run.py"
    elif selected_algo == "UMAP":
        if FLOW_TYPE == "podman":
            job_params["params_list"][-1]["command"] = "python umap_run.py"
        else:
            job_params["params_list"][-1][
                "python_file_name"
            ] = "mlex_dimension_reduction_umap/umap_run.py"

    job_params["params_list"][-1]["params"]["io_parameters"] = io_parameters
    job_params["params_list"][-1]["params"]["io_parameters"][
        "output_dir"
    ] = "mlex_store"
    job_params["params_list"][-1]["params"]["io_parameters"]["uid_save"] = ""
    job_params["params_list"][-1]["params"]["io_parameters"]["uid_retrieve"] = None
    job_params["params_list"][-1]["params"]["model_parameters"] = input_params
    print(job_params)
    print(TRAIN_PARAMS_EXAMPLE, flush=True)

    # run prefect job, job_uid is the new experiment id -> uid_save in the pca_example.yaml file
    job_uid = schedule_prefect_flow(
        FLOW_NAME,
        parameters=job_params,
        flow_run_name=f"{job_name} {current_time}",
        tags=PREFECT_TAGS + ["train", project_name],
    )
    job_message = f"Job has been succesfully submitted with uid: {job_uid}."
    print(job_message, flush=True)

    fig = go.Figure(
        go.Heatmap(),
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=20, r=20, b=20, t=20, pad=0),
        ),
    )
    return "cluster", -1, -2, fig, -1


@app.callback(
    [
        Output("latent_vectors", "data"),
        Output("interval-component", "max_intervals", allow_duplicate=True),
    ],
    Input("interval-component", "n_intervals"),
    State("job-selector", "value"),
    State("interval-component", "max_intervals"),
    prevent_initial_call=True,
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

    children_flows = get_children_flow_run_ids(experiment_id)
    if len(children_flows) > 0:
        # read the latent vectors from the output dir
        output_path = f"mlex_store/{children_flows[-1]}/latent_vectors.npy"
        print(output_path, flush=True)
        if os.path.exists(output_path):
            latent_vectors = np.load(output_path)
            print("latent vector", latent_vectors.shape)
            return latent_vectors, 0
    return None, -1


@app.callback(
    [
        Output("clusters", "data"),
        Output("cluster-dropdown", "options"),
    ],
    Input("run-cluster-algo", "n_clicks"),
    [
        State("latent_vectors", "data"),
        State("cluster-algo-dropdown", "value"),
        State("additional-cluster-params", "children"),
        State("job-selector", "value"),
    ],
)
def apply_clustering(
    apply_n_clicks, latent_vectors, selected_algo, children, experiment_id
):
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
    # TODO: pop up a widow to ask user to first run diemnsion reduction then apply
    if apply_n_clicks == 0 or experiment_id is None:
        raise PreventUpdate
    latent_vectors = np.array(latent_vectors)

    input_params = {}
    if children:
        for child in children["props"]["children"]:
            key = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value
    print("Clustering params:", input_params, flush=True)

    if selected_algo == "KMeans":
        obj = MiniBatchKMeans(n_clusters=input_params["n_clusters"])
    elif selected_algo == "DBSCAN":
        obj = DBSCAN(eps=input_params["eps"], min_samples=input_params["min_samples"])
    elif selected_algo == "HDBSCAN":
        obj = HDBSCAN(min_cluster_size=input_params["min_cluster_size"])

    clusters, options = None, None
    if obj:
        children_flows = get_children_flow_run_ids(experiment_id)
        if len(children_flows) > 0:
            clusters = obj.fit_predict(latent_vectors)
            output_path = f"mlex_store/{children_flows[0]}"
            np.save(f"{output_path}/clusters.npy", clusters)
            unique_clusters = np.unique(clusters)
            options = [
                {"label": f"Cluster {cluster}", "value": cluster}
                for cluster in unique_clusters
                if cluster != -1
            ]
            options.insert(0, {"label": "All", "value": -1})

    print("clusters", clusters, flush=True)

    return clusters, options


@app.callback(
    Output("scatter", "figure"),
    [
        Input("latent_vectors", "data"),
        Input("cluster-dropdown", "value"),
        Input("label-dropdown", "value"),
        Input("scatter-color", "value"),
        Input("clusters", "data"),  # move clusters to the input
    ],
    [
        State("scatter", "figure"),
        State("scatter", "selectedData"),
        State("additional-model-params", "children"),
        State("input_labels", "data"),
        State("label_schema", "data"),
    ],
)
def update_scatter_plot(
    latent_vectors,
    selected_cluster,
    selected_label,
    scatter_color,
    clusters,
    current_figure,
    selected_data,
    children,
    labels,
    label_names,
):
    """
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
    """
    if latent_vectors is None or children is None:
        raise PreventUpdate
    latent_vectors = np.array(latent_vectors)
    print("latent vector shape:", latent_vectors.shape, flush=True)

    n_components = children["props"]["children"][0]["props"]["children"][1]["props"][
        "value"
    ]

    if selected_data is not None and len(selected_data.get("points", [])) > 0:
        selected_indices = []
        for point in selected_data["points"]:
            if "customdata" in point and len(point["customdata"]):
                selected_indices.append(point["customdata"][0])
        print("selected indices: ", selected_indices)
    else:
        selected_indices = None

    if (
        not clusters
    ):  # when clusters is None, i.e., after submit dimension reduction but before apply clustering
        clusters = [-1 for i in range(latent_vectors.shape[0])]
    cluster_names = {a: a for a in np.unique(clusters).astype(int)}

    scatter_data = generate_scatter_data(
        latent_vectors,
        n_components,
        selected_cluster,
        clusters,
        cluster_names,
        selected_label,
        labels,
        label_names,
        scatter_color,
    )

    fig = go.Figure(scatter_data)
    fig.update_layout(
        margin=go.layout.Margin(l=20, r=20, b=20, t=20, pad=0),
        legend=dict(tracegroupgap=20),
    )

    if (
        current_figure
        and "xaxis" in current_figure["layout"]
        and "yaxis" in current_figure["layout"]
        and "autorange" in current_figure["layout"]["xaxis"]
        and current_figure["layout"]["xaxis"]["autorange"] is False
    ):
        # Update the axis range with current figure's values if available and if autorange is False
        fig.update_xaxes(range=current_figure["layout"]["xaxis"]["range"])
        fig.update_yaxes(range=current_figure["layout"]["yaxis"]["range"])
    else:
        # If it's the initial figure or autorange is True, set autorange to True to fit all points in view
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

    if selected_indices is not None:
        # Use the selected indices to highlight the selected points in the updated figure
        for trace in fig.data:
            if trace.marker.color is not None:
                trace.marker.color = [
                    hex_to_rgba("grey", 0.3) if i not in selected_indices else "red"
                    for i in range(len(trace.marker.color))
                ]
    return fig


@app.callback(
    Output("heatmap", "figure", allow_duplicate=True),
    [
        Input("scatter", "clickData"),
        Input("scatter", "selectedData"),
        Input("mean-std-toggle", "value"),
    ],
    [
        State("example-dataset-selection", "value"),  # example dataset
        State(
            {"base_id": "file-manager", "name": "data-project-dict"}, "data"
        ),  # DataProject for FM
    ],
    prevent_initial_call=True,
)
def update_heatmap(
    click_data,
    selected_data,
    display_option,
    selected_example_dataset,
    data_project_dict,
):
    """
    This callback update the heatmap
    Args:
        click_data:         clicked data on scatter figure
        selected_data:      lasso or rect selected data points on scatter figure
        display_option:     option to display mean or std
    Returns:
        fig:                updated heatmap
    """
    if not selected_example_dataset and not data_project_dict:
        raise PreventUpdate

    # user select a group of points
    if selected_data is not None and len(selected_data["points"]) > 0:
        selected_indices = [
            point["customdata"][0] for point in selected_data["points"]
        ]  # Access customdata for the original indices
        print("selected_indices", selected_indices)

        # FileManager
        # print("upload_file_paths") # if not selected, its an empty list not None
        selected_images = []

        data_project = DataProject.from_dict(data_project_dict)
        if len(data_project.datasets) > 0:
            print("FM file")
            selected_images, _ = data_project.read_datasets(
                selected_indices, export="pillow"
            )
        # Example dataset
        elif selected_example_dataset == "data/example_shapes/Demoshapes.npz":
            print("Demoshapes.npz")
            selected_images = np.load("/app/work/" + selected_example_dataset)["arr_0"][
                selected_indices
            ]
            print(selected_images.shape)
        elif (
            selected_example_dataset
            == "data/example_latentrepresentation/f_vectors.parquet"
        ):
            print("f_vectors.parque")
            df = pd.read_parquet("/app/work/" + selected_example_dataset)
            selected_images = df.iloc[selected_indices].values
        selected_images = np.array(selected_images)

        print("selected_images shape:", selected_images.shape)

        # display options
        if display_option == "mean":
            heatmap_data = go.Heatmap(z=np.mean(selected_images, axis=0))
        elif display_option == "sigma":
            heatmap_data = go.Heatmap(z=np.std(selected_images, axis=0))

    elif click_data is not None and len(click_data["points"]) > 0:
        selected_index = click_data["points"][0]["customdata"][0]
        # FileManager
        data_project = DataProject.from_dict(data_project_dict)
        if len(data_project.datasets) > 0:
            selected_images, _ = data_project.read_datasets(
                [selected_index], export="pillow"
            )
        # Example dataset
        elif selected_example_dataset == "data/example_shapes/Demoshapes.npz":
            clicked_image = np.load("/app/work/" + selected_example_dataset)["arr_0"][
                selected_index
            ]
        elif (
            selected_example_dataset
            == "data/example_latentrepresentation/f_vectors.parquet"
        ):
            df = pd.read_parquet("/app/work/" + selected_example_dataset)
            clicked_image = df.iloc[selected_index].values
        clicked_image = np.array(clicked_image)

        heatmap_data = go.Heatmap(z=clicked_image)

    else:
        heatmap_data = go.Heatmap()

    # only update heat map when the input data is 2d images, do not update for input latent vectors
    if heatmap_data["z"] is None or len(np.shape(heatmap_data["z"])) < 2:
        raise PreventUpdate

    # Determine the aspect ratio based on the shape of the heatmap_data's z-values
    aspect_x = 1
    aspect_y = 1
    if heatmap_data["z"] is not None:
        if heatmap_data["z"].size > 0:
            print(np.shape(heatmap_data["z"]))
            aspect_y, aspect_x = np.shape(heatmap_data["z"])[-2:]

    return go.Figure(
        data=heatmap_data,
        layout=dict(
            autosize=True,
            margin=go.layout.Margin(l=20, r=20, b=20, t=20, pad=0),
            yaxis=dict(
                scaleanchor="x", scaleratio=aspect_y / aspect_x, autorange="reversed"
            ),
        ),
    )


@app.callback(
    Output("stats-div", "children"),
    Input("scatter", "selectedData"),
    [
        State("clusters", "data"),
        State("input_labels", "data"),
        State("label_schema", "data"),
    ],
)
def update_statistics(selected_data, clusters, assigned_labels, label_names):
    """
    This callback update the statistics panel
    Args:
        selected_data:      lasso or rect selected data points on scatter figure
        clusters:           clusters for latent vectors
        assigned_labels:    labels for each latent vector
        label_names:        same as label schema
    Returns:
        [num_images, clusters, labels]:     statistics
    """
    assigned_labels = np.array(assigned_labels)
    print("assigned_labels", assigned_labels, flush=True)

    if (
        selected_data is not None
        and len(selected_data["points"]) > 0
        and (assigned_labels != [-1]).all()
    ):
        selected_indices = [
            point["customdata"][0] for point in selected_data["points"]
        ]  # Access customdata for the original indices
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
        labels_str = ", ".join(
            str(label_int_to_str_map[label]) for label in unique_labels if label >= 0
        )
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
    Output("feature-vector-model-list", "options"),
    Input("interval-component", "n_intervals"),
)
def update_feature_vector_model_list(n_intervals):
    """
    This callback update the feature vector model list
    Args:
        n_intervals:         interval component
    Returns:
        options:            feature vector model list
    """
    # TODO: Connect to data clinic
    # TODO: Check if inference has already taken place in this dataset
    folder_names = [
        os.path.join(dirpath, dir)
        for dirpath, dirs, _ in os.walk(MODEL_DIR)
        for dir in dirs
    ]
    return folder_names


if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=8070,
    )
