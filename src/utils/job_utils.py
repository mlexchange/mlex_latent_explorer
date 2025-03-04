import json
import os
from urllib.parse import urljoin

# I/O parameters for job execution
READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", None)
WRITE_DIR_MOUNT = os.getenv("WRITE_DIR_MOUNT", None)
WRITE_DIR = os.getenv("WRITE_DIR", "")
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

# Flow parameters
PARTITIONS_CPU = json.loads(os.getenv("PARTITIONS_CPU", "[]"))
RESERVATIONS_CPU = json.loads(os.getenv("RESERVATIONS_CPU", "[]"))
MAX_TIME_CPU = os.getenv("MAX_TIME_CPU", "1:00:00")
PARTITIONS_GPU = json.loads(os.getenv("PARTITIONS_CPU", "[]"))
RESERVATIONS_GPU = json.loads(os.getenv("RESERVATIONS_CPU", "[]"))
MAX_TIME_GPU = os.getenv("MAX_TIME_CPU", "1:00:00")
SUBMISSION_SSH_KEY = os.getenv("SUBMISSION_SSH_KEY", "")
FORWARD_PORTS = json.loads(os.getenv("FORWARD_PORTS", "[]"))
DOCKER_NETWORK=os.getenv("DOCKER_NETWORK", "")
FLOW_TYPE = os.getenv("FLOW_TYPE", "conda")


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
    flow_type,
    image_name=None,
    image_tag=None,
    python_file_name=None,
    conda_env=None,
):
    """
    Parse job parameters
    """
    # TODO: Use model_name to define the conda_env/algorithm to be executed
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

    if flow_type == "podman":
        job_params = {
            "flow_type": "podman",
            "params_list": [
                {
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "command": f'python {python_file_name}',
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/app/work/data",
                        f"{WRITE_DIR_MOUNT}:/app/work/mlex_store",
                    ],
                }
            ],
        }

    elif flow_type == "docker":
        job_params = {
            "flow_type": "docker",
            "params_list": [
                {
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "command": f'python {python_file_name}',
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                    "volumes": [
                        f"{READ_DIR_MOUNT}:/tiled_storage",
                       
                    ],
                    "network":DOCKER_NETWORK
                }
            ],
        }

    elif flow_type == "conda":
        job_params = {
            "flow_type": "conda",
            "params_list": [
                {
                    "conda_env_name": conda_env,
                    "python_file_name": python_file_name,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                },
            ],
        }

    else:
        job_params = {
            "flow_type": "slurm",
            "params_list": [
                {
                    "job_name": "latent_space_explorer",
                    "num_nodes": 1,
                    "partitions": PARTITIONS_CPU,
                    "reservations": RESERVATIONS_CPU,
                    "max_time": MAX_TIME_CPU,
                    "conda_env_name": "mlex_dimension_reduction_pca",
                    "submission_ssh_key": SUBMISSION_SSH_KEY,
                    "forward_ports": FORWARD_PORTS,
                    "params": {
                        "io_parameters": io_parameters,
                        "model_parameters": model_parameters,
                    },
                }
            ],
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
        # The achtual parameter item is the first and only child of the parameter container
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
