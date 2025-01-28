import json
import os
from uuid import uuid4

from dash import MATCH, Input, Output, html
from dotenv import load_dotenv

from src.app_layout import app, clustering_models, dim_reduction_models, mlex_components
from src.callbacks.display import update_data_overview  # noqa: F401
from src.callbacks.execute import run_train  # noqa: F401

load_dotenv(".env")

# Define directories
READ_DIR = os.getenv("READ_DIR", "data")
WRITE_DIR = os.getenv("WRITE_DIR", "mlex_store")
MODEL_DIR = f"{WRITE_DIR}/models"
READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", None)
WRITE_DIR_MOUNT = os.getenv("WRITE_DIR_MOUNT", None)

# Prefect
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["latent-space-explorer"]'))
TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
FLOW_NAME = os.getenv("FLOW_NAME", "")

HOST = os.getenv("HOST", "127.0.0.1")


@app.callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": MATCH,
        },
        "children",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": MATCH,
        },
        "value",
    ),
)
def update_dim_reduction_model_parameters(model_name):
    model = dim_reduction_models[model_name]
    if model["gui_parameters"]:
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        return item_list
    else:
        return html.Div("Model has no parameters")


@app.callback(
    Output("additional-cluster-params", "children"),
    Input("cluster-algo-dropdown", "value"),
)
def update_clustering_model_parameters(model_name):
    model = clustering_models[model_name]
    if model["gui_parameters"]:
        item_list = mlex_components.get_parameter_items(
            _id={"type": str(uuid4())}, json_blob=model["gui_parameters"]
        )
        return item_list
    else:
        return html.Div("Model has no parameters")


if __name__ == "__main__":
    app.run_server(
        debug=True,
        host=HOST,
        port=8070,
    )
