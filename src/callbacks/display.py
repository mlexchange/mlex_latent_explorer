import io
import math
from base64 import b64encode

import numpy as np
from dash import ALL, Input, Output, State, callback
from file_manager.data_project import DataProject
from PIL import Image

from ..app_layout import DATA_TILED_KEY, NUM_IMGS_OVERVIEW


def get_empty_image():
    img = Image.fromarray(255 * (np.ones((32, 32)).astype(np.uint8)))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    contents = buffered.getvalue()
    return "data:image/jpeg;base64," + b64encode(contents).decode("utf-8")


@callback(
    Output({"type": "thumbnail-src", "index": ALL}, "src"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    Input("log-transform", "value"),
    Input("min-max-percentile", "value"),
    Input("current-page", "data"),
    State({"base_id": "file-manager", "name": "total-num-data-points"}, "data"),
)
def update_data_overview(
    data_project_dict,
    log_transform,
    percentiles,
    current_page,
    num_imgs,
):
    imgs = []
    if data_project_dict != {}:
        data_project = DataProject.from_dict(data_project_dict, api_key=DATA_TILED_KEY)
        if (
            len(data_project.datasets) > 0
            and data_project.datasets[-1].cumulative_data_count > 0
        ):
            if percentiles is None:
                percentiles = [0, 100]
            max_index = min((current_page + 1) * NUM_IMGS_OVERVIEW, num_imgs)
            imgs, _ = data_project.read_datasets(
                indices=list(range(current_page * NUM_IMGS_OVERVIEW, max_index)),
                export="base64",
                resize=True,
                log=log_transform,
                percentiles=percentiles,
            )

    if len(imgs) < NUM_IMGS_OVERVIEW:
        for _ in range(NUM_IMGS_OVERVIEW - len(imgs)):
            imgs.append(get_empty_image())

    return imgs


@callback(
    Output("current-page", "data", allow_duplicate=True),
    Input({"base_id": "file-manager", "name": "total-num-data-points"}, "data"),
    Input("first-page", "n_clicks"),
    prevent_initial_call=True,
)
def go_to_first_page(
    num_imgs,
    button_first_page,
):
    """
    Update the current page to the first page
    """
    return 0


@callback(
    Output("current-page", "data", allow_duplicate=True),
    Input("prev-page", "n_clicks"),
    State("current-page", "data"),
    prevent_initial_call=True,
)
def go_to_prev_page(
    button_prev_page,
    current_page,
):
    """
    Update the current page to the previous page
    """
    current_page = current_page - 1
    return current_page


@callback(
    Output("current-page", "data", allow_duplicate=True),
    Input("next-page", "n_clicks"),
    State("current-page", "data"),
    prevent_initial_call=True,
)
def go_to_next_page(
    button_next_page,
    current_page,
):
    """
    Update the current page to the next page
    """
    current_page = current_page + 1
    return current_page


@callback(
    Output("current-page", "data", allow_duplicate=True),
    Input("last-page", "n_clicks"),
    State({"base_id": "file-manager", "name": "total-num-data-points"}, "data"),
    prevent_initial_call=True,
)
def go_to_last_page(
    button_last_page,
    num_imgs,
):
    """
    Update the current page to the last page
    """
    return math.ceil(num_imgs / NUM_IMGS_OVERVIEW) - 1


@callback(
    [
        [Output("first-page", "disabled"), Output("prev-page", "disabled")],
        [Output("next-page", "disabled"), Output("last-page", "disabled")],
        Input("current-page", "data"),
        State({"base_id": "file-manager", "name": "total-num-data-points"}, "data"),
    ],
    prevent_initial_call=True,
)
def disable_buttons(
    current_page,
    num_imgs,
):
    """
    Disable first and last page buttons based on the current page
    """
    max_num_pages = math.ceil(num_imgs / NUM_IMGS_OVERVIEW)
    return (
        2 * [current_page == 0],
        2 * [max_num_pages <= current_page + 1],
    )
