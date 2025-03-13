import json
import os
import traceback
import uuid
from datetime import datetime

import pytz
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import (
    get_children_flow_run_ids,
    get_flow_run_name,
    get_flow_run_state,
    schedule_prefect_flow,
)

from src.app_layout import USER, clustering_models, dim_reduction_models,latent_space_models
from src.utils.data_utils import tiled_results
from src.utils.job_utils import parse_job_params, parse_model_params, parse_clustering_job_params
from src.utils.plot_utils import generate_notification

import os
import mlflow
from mlflow.tracking import MlflowClient
import logging


MODE = os.getenv("MODE", "")
TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
FLOW_NAME = os.getenv("FLOW_NAME", "")
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["latent-space-explorer"]'))
RESULTS_DIR = os.getenv("RESULTS_DIR", "")
FLOW_TYPE = os.getenv("FLOW_TYPE", "conda")


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "notifications-container",
            "aio_id": "latent-space-jobs",
        },
        "children",
        allow_duplicate=True,
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-button",
            "aio_id": "latent-space-jobs",
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": "latent-space-jobs",
        },
        "children",
    ),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": "latent-space-jobs",
        },
        "value",
    ),
    State("log-transform", "value"),
    State("min-max-percentile", "value"),
    State("mask-dropdown", "value"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "job-name",
            "aio_id": "latent-space-jobs",
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "latent-space-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def run_latent_space(
    n_clicks,
    model_parameter_container,
    data_project_dict,
    model_name,
    log,
    percentiles,
    mask,
    job_name,
    project_name,
):
    """
    This callback submits a job request to the compute service
    Args:
        n_clicks:                   Number of clicks
        model_parameter_container:  App parameters
        data_project_dict:          Data project dictionary
        model_name:                 Selected model name
        log:                        Log transform
        percentiles:                Min-max percentiles
        mask:                       Mask selection
        job_name:                   Job name
        project_name:               Project name
    Returns:
        open the alert indicating that the job was submitted
    """
    if n_clicks is not None and n_clicks > 0:
        model_parameters, parameter_errors = parse_model_params(
            model_parameter_container, log, percentiles, mask
        )
        # Check if the model parameters are valid
        if parameter_errors:
            notification = generate_notification(
                "Model Parameters",
                "red",
                "fluent-mdl2:machine-learning",
                "Model parameters are not valid!",
            )
            return notification

        data_project = DataProject.from_dict(data_project_dict)

        latent_space_params = latent_space_models[latent_space_models.modelname_list[0]]
        dim_reduction_params = dim_reduction_models[model_name]
        train_params = parse_job_params(
            data_project,
            model_parameters,
            USER,
            project_name,
            FLOW_TYPE,
            latent_space_params,
            dim_reduction_params,
        )
       
        if MODE == "dev":
            job_uid = str(uuid.uuid4())
            job_message = (
                f"Dev Mode: Job has been succesfully submitted with uid: {job_uid}"
            )
            notification_color = "primary"
        else:
            try:
                # Prepare tiled
                tiled_results.prepare_project_container(USER, project_name)
                # Schedule job
                current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
                    "%Y/%m/%d %H:%M:%S"
                )
                job_uid = schedule_prefect_flow(
                    FLOW_NAME,
                    parameters=train_params,
                    flow_run_name=f"{job_name} {current_time}",
                    tags=PREFECT_TAGS + [project_name, "latent-space"],
                )
                job_message = f"Job has been succesfully submitted with uid: {job_uid}"
                notification_color = "indigo"
            except Exception as e:
                # Print the traceback to the console
                traceback.print_exc()
                job_uid = None
                job_message = f"Job presented error: {e}"
                notification_color = "danger"

        notification = generate_notification(
            "Job Submission", notification_color, "formkit:submit", job_message
        )

        return notification
    raise PreventUpdate


@callback(
    Output("show-feature-vectors", "disabled"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-dropdown",
            "aio_id": "latent-space-jobs",
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "latent-space-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def allow_show_feature_vectors(job_id, project_name):
    """
    Determine whether to show feature vectors for the selected job. This callback checks whether a
    given job has completed and whether its feature vectors are available.
    Args:
        job_id:                 Selected job
        project_name:           Data project name
    Returns:
        show-feature-vectors:   Whether to show feature vectors
    """
    children_job_ids = get_children_flow_run_ids(job_id)

    if get_flow_run_state(children_job_ids[1]) != "COMPLETED":
        return True

    child_job_id = children_job_ids[1]
    expected_result_uri = f"{USER}/{project_name}/{child_job_id}"
    try:
        tiled_results.get_data_by_trimmed_uri(expected_result_uri)
        return False
    except Exception:
        traceback.print_exc()
        return True


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-button",
            "aio_id": "clustering-jobs",
        },
        "disabled",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-dropdown",
            "aio_id": "latent-space-jobs",
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "latent-space-jobs",
        },
        "data",
    ),
)
def allow_run_clustering(job_id, project_name):
    """
    Determine whether to run clustering for the selected job. This callback checks whether a given
    job has completed and whether its feature vectors are available.
    Args:
        job_id:                 Selected job
        project_name:           Data project name
    Returns:
        run-clustering:         Whether to run clustering
    """
    if job_id is None:
        raise PreventUpdate

    children_job_ids = get_children_flow_run_ids(job_id)

    if get_flow_run_state(children_job_ids[1]) != "COMPLETED":
        return True

    child_job_id = children_job_ids[1]
    expected_result_uri = f"{USER}/{project_name}/{child_job_id}"
    try:
        tiled_results.get_data_by_trimmed_uri(expected_result_uri)
        return False
    except Exception:
        traceback.print_exc()
        return True


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "notifications-container",
            "aio_id": "clustering-jobs",
        },
        "children",
        allow_duplicate=True,
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-button",
            "aio_id": "clustering-jobs",
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": "clustering-jobs",
        },
        "children",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": "clustering-jobs",
        },
        "value",
    ),
    State("log-transform", "value"),
    State("min-max-percentile", "value"),
    State("mask-dropdown", "value"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "job-name",
            "aio_id": "clustering-jobs",
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "latent-space-jobs",
        },
        "data",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-dropdown",
            "aio_id": "latent-space-jobs",
        },
        "value",
    ),
    prevent_initial_call=True,
)
def run_clustering(
    n_clicks,
    model_parameter_container,
    model_name,
    log,
    percentiles,
    mask,
    job_name,
    project_name,
    dimension_reduction_job_id,
):
    """
    This callback submits a clustering job request to the compute service
    Args:
        n_clicks:                   Number of clicks
        model_parameter_container:  App parameters
        model_name:                 Selected model name
        log:                        Log transform
        percentiles:                Min-max percentiles
        mask:                       Mask selection
        job_name:                   Job name
        project_name:               Project name
        dimension_reduction_job_id: Job ID
    Returns:
        open the alert indicating that the job was submitted
    """
    if n_clicks is not None and n_clicks > 0:
        model_parameters, parameter_errors = parse_model_params(
            model_parameter_container, log, percentiles, mask
        )
        # Check if the model parameters are valid
        if parameter_errors:
            notification = generate_notification(
                "Model Parameters",
                "red",
                "fluent-mdl2:machine-learning",
                "Model parameters are not valid!",
            )
            return notification

        # Prepare data project with feature vectors
        children_job_ids = get_children_flow_run_ids(dimension_reduction_job_id)
        child_job_id = children_job_ids[1]

        expected_result_uri = f"/{USER}/{project_name}/{child_job_id}"
        data_project_fvec = DataProject.from_dict(
            {
                "root_uri": tiled_results.data_tiled_uri,
                "data_type": "tiled",
                "datasets": [{"uri": expected_result_uri, "cumulative_data_count": 0}],
                "project_id": None,
            },
            api_key=tiled_results.data_tiled_api_key,
        )

        model_exec_params = clustering_models[model_name]
        job_params = parse_clustering_job_params(
            data_project_fvec,
            model_parameters,
            USER,
            project_name,
            FLOW_TYPE,
            model_exec_params["image_name"],
            model_exec_params["image_tag"],
            model_exec_params["python_file_name"],
            model_exec_params["conda_env"],
        )

        if MODE == "dev":
            job_uid = str(uuid.uuid4())
            job_message = (
                f"Dev Mode: Job has been succesfully submitted with uid: {job_uid}"
            )
            notification_color = "primary"
        else:
            try:
                # Prepare tiled
                tiled_results.prepare_project_container(USER, project_name)
                # Schedule job
                current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
                    "%Y/%m/%d %H:%M:%S"
                )
                dimension_reduction_name = get_flow_run_name(dimension_reduction_job_id)
                job_uid = schedule_prefect_flow(
                    FLOW_NAME,
                    parameters=job_params,
                    flow_run_name=f"{dimension_reduction_name} {job_name} {current_time}",
                    tags=PREFECT_TAGS + [project_name, "clustering"],
                )
                job_message = f"Job has been succesfully submitted with uid: {job_uid}"
                notification_color = "indigo"
            except Exception as e:
                # Print the traceback to the console
                traceback.print_exc()
                job_uid = None
                job_message = f"Job presented error: {e}"
                notification_color = "danger"

        notification = generate_notification(
            "Job Submission", notification_color, "formkit:submit", job_message
        )

        return notification
    raise PreventUpdate


@callback(
    Output("show-clusters", "disabled"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "run-dropdown",
            "aio_id": "clustering-jobs",
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "latent-space-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def allow_show_clusters(job_id, project_name):
    """
    Determine whether to show clusters for the selected job. This callback checks whether a given job
    has completed and whether its clusters are available.
    Args:
        job_id:                 Selected job
        project_name:           Data project name
    Returns:
        show-clusters:          Whether to show the clusters
    """
    if job_id is None:
        raise PreventUpdate

    children_job_ids = get_children_flow_run_ids(job_id)

    if get_flow_run_state(children_job_ids[0]) != "COMPLETED":
        return True

    child_job_id = children_job_ids[0]
    expected_result_uri = f"{USER}/{project_name}/{child_job_id}"
    try:
        tiled_results.get_data_by_trimmed_uri(expected_result_uri)
        return False
    except Exception:
        traceback.print_exc()
        return True


