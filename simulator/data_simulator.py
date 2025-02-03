import logging
import os
import time
from datetime import datetime

import pytz
from dotenv import load_dotenv
from mlex_utils.prefect_utils.core import schedule_prefect_flow
from tiled.client import from_uri

load_dotenv(".env")

DATA_TILED_URI = os.getenv("DEFAULT_TILED_URI")
DATA_TILED_API_KEY = os.getenv("DATA_TILED_KEY")
FLOW_NAME = os.getenv("FLOW_NAME", "")
PREFECT_TAGS = ["latent-space-explorer-live"]
WRITE_DIR = os.getenv("WRITE_DIR")
USER = os.getenv("USER")

RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
# Only append "/api/v1/metadata/" if it's not already in the string
if "/api/v1/metadata/" not in RESULTS_TILED_URI:
    RESULTS_TILED_URI = RESULTS_TILED_URI.rstrip("/") + "/api/v1/metadata/"

RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", None)
TRAINED_MODEL_URI = os.getenv("TRAINED_MODEL_URI")
TIMEZONE = os.getenv("TIMEZONE", "UTC")
PUBLISHER_PYTHON_FILE = os.getenv("PUBLISHER_PYTHON_FILE")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

flow = {
    "flow_type": "conda",
    "params_list": [
        {
            "conda_env_name": "mlex_dimension_reduction_umap",
            "python_file_name": "mlex_dimension_reduction_umap/src/umap_run.py",
            "params": {
                "io_parameters": {
                    "uid_retrieve": "",
                    "data_uris": [],
                    "data_tiled_api_key": DATA_TILED_API_KEY,
                    "data_type": "tiled",
                    "root_uri": DATA_TILED_URI,
                    "results_dir": f"{WRITE_DIR}/feature_vectors",
                    "results_tiled_uri": f"{RESULTS_TILED_URI}/{USER}/live_experiment",
                    "results_tiled_api_key": RESULTS_TILED_API_KEY,
                    "load_model_path": TRAINED_MODEL_URI,
                    "project_name": "live_experiment",
                },
                "model_parameters": {
                    "min_dist": 0.1,
                    "n_neighbors": 15,
                    "n_components": 2,
                },
            },
        },
        {
            "conda_env_name": "mlex_rabbitmq_streamer",
            "python_file_name": PUBLISHER_PYTHON_FILE,
            "params": {
                "io_parameters": {
                    "uid_retrieve": "",
                    "project_name": "live_experiment",
                }
            },
        },
    ],
}


def get_data_list(tiled_uri, tiled_api_key=None):
    client = from_uri(tiled_uri, api_key=tiled_api_key)
    data_list = client.keys()[0:10]
    return data_list


def prepare_tiled_results_container(tiled_uri, tiled_api_key):
    write_client = from_uri(tiled_uri, api_key=tiled_api_key)
    for sub_uri in [USER, "live_experiment"]:
        try:
            write_client[sub_uri]
        except KeyError:
            write_client.new(
                structure_family="container",
                key=sub_uri,
                data_sources=[],
            )
        write_client = write_client[sub_uri]
    return write_client


if __name__ == "__main__":
    data_list = get_data_list(DATA_TILED_URI, DATA_TILED_API_KEY)
    write_client = prepare_tiled_results_container(
        RESULTS_TILED_URI, RESULTS_TILED_API_KEY
    )

    for data_uri in data_list:
        logger.info(f"Sending URI {data_uri} for processing.")

        new_flow = flow.copy()
        new_flow["params_list"][0]["params"]["io_parameters"]["data_uris"] = [data_uri]
        current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
            "%Y/%m/%d %H:%M:%S"
        )
        job_name = f"Live model training for {data_uri}"

        schedule_prefect_flow(
            FLOW_NAME,
            new_flow,
            flow_run_name=f"{job_name} {current_time}",
            tags=PREFECT_TAGS + ["train"],
        )

        time.sleep(10)
