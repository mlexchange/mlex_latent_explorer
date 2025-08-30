import json
import logging
import os
from urllib.parse import urljoin

from src.utils.mlflow_utils import MLflowClient

# I/O parameters for job execution
READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", None)
WRITE_DIR_MOUNT = os.getenv("WRITE_DIR_MOUNT", None)
WRITE_DIR = os.getenv("WRITE_DIR", "")
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

logger = logging.getLogger(__name__)
mlflow_client = MLflowClient()

def parse_tiled_url(url, user, project_name, tiled_base_path="/api/v1/metadata"):
    """
    Given any URL (e.g. http://localhost:8000/results),
    return the same scheme/netloc but with path='/api/v1/metadata'.
    """
    if tiled_base_path not in url:
        url = urljoin(url, os.path.join(tiled_base_path, user, project_name))
    else:
        url = urljoin(url, f"/{user}/{project_name}")
    return url


def parse_job_params(
    data_project,
    model_parameters,
    user,
    project_name,
    latent_space_params,
    dim_reduction_params,
    mlflow_model_id=None,
):
    """
    Parse training job parameters
    """
    data_uris = [dataset.uri for dataset in data_project.datasets]

    results_dir = f"{WRITE_DIR}/{user}"

    io_parameters = {
        "uid_retrieve": "",
        "data_uris": data_uris,
        "data_tiled_api_key": data_project.api_key,
        "data_type": data_project.data_type,
        "root_uri": data_project.root_uri,
        "models_dir": f"{results_dir}/models",
        "results_tiled_uri": parse_tiled_url(RESULTS_TILED_URI, user, project_name),
        "results_tiled_api_key": RESULTS_TILED_API_KEY,
        "results_dir": f"{results_dir}",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "mlflow_tracking_username": MLFLOW_TRACKING_USERNAME,
        "mlflow_tracking_password": MLFLOW_TRACKING_PASSWORD,
        "mlflow_model": mlflow_model_id,
    }

    auto_params = mlflow_client.get_mlflow_params(mlflow_model_id)
    logger.info(f"Autoencoder parameters: {auto_params}")

    # Create a simpler params_list structure with model_name and task_name
    params_list = [
        {
            "model_name": latent_space_params["model_name"],
            "task_name": "inference",
            "params": {
                "io_parameters": io_parameters,
                "model_parameters": auto_params,
            },
        },
        {
            "model_name": dim_reduction_params["model_name"],
            "task_name": "excute",
            "params": {
                "io_parameters": io_parameters,
                "model_parameters": model_parameters,
            },
        },
    ]

    # Keep the job params simplified
    job_params = {
        "params_list": params_list,
    }

    return job_params


def parse_clustering_job_params(
    data_project,
    model_parameters,
    user,
    project_name,
    clustering_params
):
    """
    Parse job parameters for clustering
    """
    data_uris = [dataset.uri for dataset in data_project.datasets]

    results_dir = f"{WRITE_DIR}/{user}"

    io_parameters = {
        "uid_retrieve": "",
        "data_uris": data_uris,
        "data_tiled_api_key": data_project.api_key,
        "data_type": data_project.data_type,
        "root_uri": data_project.root_uri,
        "save_model_path": f"{results_dir}/models",
        "results_tiled_uri": parse_tiled_url(RESULTS_TILED_URI, user, project_name),
        "results_tiled_api_key": RESULTS_TILED_API_KEY,
        "results_dir": f"{results_dir}",
    }

    # Create a simpler params_list structure with model_name and task_name
    params_list = [
        {
            "model_name": clustering_params["model_name"],
            "task_name": "excute",
            "params": {
                "io_parameters": io_parameters,
                "model_parameters": model_parameters,
            },
        }
    ]

    # Keep the job params simplified
    job_params = {
        "params_list": params_list,
    }

    return job_params


def parse_model_params(model_parameters_html, log, percentiles, mask):
    """
    Extracts parameters from the children component of a ParameterItems component,
    if there are any errors in the input, it will return an error status
    """
    errors = False
    input_params = {}
    for param in model_parameters_html["props"]["children"]:
        # param["props"]["children"][0] is the label
        # param["props"]["children"][1] is the input
        parameter_container = param["props"]["children"][1]
        # The actual parameter item is the first and only child of the parameter container
        parameter_item = parameter_container["props"]["children"]["props"]
        key = parameter_item["id"]["param_key"]
        if "value" in parameter_item:
            value = parameter_item["value"]
        elif "checked" in parameter_item:
            value = parameter_item["checked"]
        if "error" in parameter_item:
            if parameter_item["error"] is not False:
                errors = True
        input_params[key] = value

    # Manually add data transformation parameters
    input_params["log"] = log
    input_params["percentiles"] = percentiles
    input_params["mask"] = mask if mask != "None" else None
    return input_params, errors