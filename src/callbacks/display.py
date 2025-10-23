import io
import logging
import math
from base64 import b64encode

import numpy as np
import numba as nb
from dash import ALL, Input, Output, Patch, State, callback, no_update
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import get_children_flow_run_ids
from PIL import Image

from src.app_layout import DATA_TILED_KEY, LIVE_TILED_API_KEY, NUM_IMGS_OVERVIEW, USER, long_callback_manager
from src.utils.data_utils import hash_list_of_strings, tiled_results
from src.utils.plot_utils import (
    generate_heatmap_plot,
    generate_scatter_data,
    plot_empty_heatmap,
    plot_empty_scatter,
)

MAX_HEATMAP_SELECTION = 30

# Add logger for display.py
logger = logging.getLogger("lse.display")

@nb.njit(parallel=True)
def fast_mean(arr):
    """
    Custom Numba-accelerated mean calculation along axis 0
    Works with 3D arrays like our image data
    """
    result = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float32)
    for i in nb.prange(arr.shape[1]):
        for j in nb.prange(arr.shape[2]):
            sum_val = 0.0
            for k in range(arr.shape[0]):
                sum_val += arr[k, i, j]
            result[i, j] = sum_val / arr.shape[0]
    return result


@nb.njit(parallel=True)
def fast_std(arr):
    """
    Custom Numba-accelerated standard deviation calculation along axis 0
    Works with 3D arrays like our image data
    """
    # First calculate the mean
    mean = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float32)
    for i in nb.prange(arr.shape[1]):
        for j in nb.prange(arr.shape[2]):
            sum_val = 0.0
            for k in range(arr.shape[0]):
                sum_val += arr[k, i, j]
            mean[i, j] = sum_val / arr.shape[0]
    
    # Then calculate the variance
    var = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float32)
    for i in nb.prange(arr.shape[1]):
        for j in nb.prange(arr.shape[2]):
            sum_squared_diff = 0.0
            for k in range(arr.shape[0]):
                diff = arr[k, i, j] - mean[i, j]
                sum_squared_diff += diff * diff
            var[i, j] = sum_squared_diff / arr.shape[0]
    
    # Return the square root of the variance (standard deviation)
    return np.sqrt(var)

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
    State("go-live", "n_clicks"),
)
def update_data_overview(
    data_project_dict,
    log_transform,
    percentiles,
    current_page,
    num_imgs,
    go_live,
):
    if go_live is not None and go_live % 2 == 1:
        raise PreventUpdate
    
    # Skip if in replay mode - check for the replay_mode flag
    if data_project_dict and data_project_dict.get("replay_mode", False):
        raise PreventUpdate
    
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


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "latent-space-jobs",
        },
        "data",
    ),
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "clustering-jobs",
        },
        "data",
    ),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("go-live", "n_clicks"),
    prevent_initial_call=True,
)
def update_project_name(data_project_dict, go_live):
    if go_live is not None and go_live % 2 == 1:
        raise PreventUpdate
    data_project = DataProject.from_dict(data_project_dict)
    data_uris = [dataset.uri for dataset in data_project.datasets]
    project_name = hash_list_of_strings(data_uris)
    return project_name, project_name


@callback(
    Output("scatter", "figure"),
    Input("show-feature-vectors", "value"),
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
    State("show-clusters", "value"),
)
def show_feature_vectors(
    show_feature_vectors,
    job_id,
    project_name,
    show_clusters,
):
    if show_clusters:
        return no_update
    if show_feature_vectors is False:
        return plot_empty_scatter()

    child_job_id = get_children_flow_run_ids(job_id)[1]
    expected_result_uri = f"{USER}/{project_name}/{child_job_id}"
    latent_vectors = (
        tiled_results.get_data_by_trimmed_uri(expected_result_uri).read().to_numpy()
    )
    metadata = tiled_results.get_data_by_trimmed_uri(expected_result_uri).metadata

    scatter_data = generate_scatter_data(
        latent_vectors, metadata["model_parameters"]["n_components"]
    )
    return scatter_data


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("clear-selection-button", "n_clicks"),
    State("scatter", "figure"),
    prevent_initial_call=True,
)
def clear_selections(n_clicks, current_fig):
    """
    Clears any 'selectedpoints' in the figure's traces using a Dash Patch,
    so we don't have to re-plot from scratch.
    """
    if not n_clicks:
        return no_update

    # Create a Patch object to mutate the figure incrementally
    fig_patch = Patch()

    # Remove layout.selections if present
    if "selections" in current_fig["layout"]:
        fig_patch["layout"]["selections"] = []

    # Clear selectedpoints for each trace
    for i, trace in enumerate(current_fig["data"]):
        if "selectedpoints" in trace and trace["selectedpoints"] is not None:
            fig_patch["data"][i]["selectedpoints"] = None
        if "selected" in trace and trace["selected"] is not None:
            fig_patch["data"][i]["selected"] = {}

    return fig_patch


@callback(
    output=[
        Output("heatmap", "figure"),
        Output("stats-div", "children", allow_duplicate=True),
    ],
    inputs=[
        Input("scatter", "clickData"),
        Input("scatter", "selectedData"),
        Input("mean-std-toggle", "value"),
        Input("log-transform", "value"),
        Input("min-max-percentile", "value"),
    ],
    state=[
        State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
        State("live-indices", "data"),
        State("go-live", "n_clicks"),
    ],
    background=True,  # This makes it a long callback
    manager=long_callback_manager,
    running=[
        (Output("stats-div", "children"), "Loading heatmap...", ""),
    ],
    prevent_initial_call=True,
)
def update_heatmap(
    click_data,
    selected_data,
    display_option,
    log_transform,
    percentiles,
    data_project_dict,
    live_indices,
    go_live,
):
    """
    This callback update the heatmap
    Args:
        click_data:             clicked data on scatter figure
        selected_data:          lasso or rect selected data points on scatter figure
        display_option:         option to display mean or std
        log_transform:          log transform option
        percentiles:            percentiles for min-max scaling
        data_project_dict:      data project dictionary
        live_indices:           indices for live mode
        go_live:                n_clicks for live mode button
    Returns:
        fig:                    updated heatmap
        stats:                  statistics text
    """
    # user select a group of points
    if selected_data is not None and len(selected_data["points"]) > 0:
        selected_indices = [point["pointIndex"] for point in selected_data["points"]]
        
        # Check if selection exceeds limit
        if len(selected_indices) > MAX_HEATMAP_SELECTION:
            return (
                plot_empty_heatmap(),
                f"⚠️ Selection too large ({len(selected_indices)} points). "
                f"Please select {MAX_HEATMAP_SELECTION} or fewer points for heatmap display.",
            )

    # user click on a single point
    elif click_data is not None and len(click_data["points"]) > 0:
        selected_indices = [click_data["points"][0]["pointIndex"]]

    else:
        return (
            plot_empty_heatmap(),
            "Number of images selected: 0",
        )

    if percentiles is None:
        percentiles = [0, 100]

    if len(live_indices) > 0:
        selected_indices = [live_indices[i] for i in selected_indices]

    # Determine which API key to use based on live mode
    is_live_mode = go_live is not None and go_live % 2 == 1
    is_replay_mode = data_project_dict.get("replay_mode", False)
    
    if is_live_mode:
        # Use remote API key for live mode
        api_key = LIVE_TILED_API_KEY
        logger.info("Using LIVE_TILED_API_KEY for live mode heatmap")
    elif is_replay_mode:
        # Use remote API key for replay mode
        api_key = LIVE_TILED_API_KEY
        logger.info("Using LIVE_TILED_API_KEY for replay mode heatmap")
    else:
        # Use regular DATA_TILED_KEY for offline mode
        api_key = DATA_TILED_KEY
        logger.info("Using DATA_TILED_KEY for offline mode heatmap")

    data_project = DataProject.from_dict(data_project_dict, api_key=api_key)
    selected_images, _ = data_project.read_datasets(
        selected_indices,
        resize=True,
        export="pillow",
        log=log_transform,
        percentiles=percentiles,
    )

    selected_images = np.array(selected_images)

    if display_option == "mean":
        plot_data = fast_mean(selected_images)
    else:
        plot_data = fast_std(selected_images)

    return (
        generate_heatmap_plot(plot_data),
        f"Number of images selected: {selected_images.shape[0]}",
    )


@callback(
    Output("scatter", "figure", allow_duplicate=True),
    Output("show-feature-vectors", "disabled", allow_duplicate=True),
    Output("show-feature-vectors", "value", allow_duplicate=True),
    Input("show-clusters", "value"),
    State(
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
def show_clusters(
    show_clusters, clustering_job_id, dimension_reduction_job_id, project_name
):
    """
    Update the 'scatter' figure. If the data dimension or job IDs have changed,
    we must rebuild the figure. Otherwise, we can just do a partial update (patch).
    """
    if clustering_job_id is None:
        raise PreventUpdate

    if not show_clusters:
        return plot_empty_scatter(), False, False

    # Retrieve latent vectors
    dim_red_child_id = get_children_flow_run_ids(dimension_reduction_job_id)[1]
    dim_red_uri = f"{USER}/{project_name}/{dim_red_child_id}"
    dim_red_data = tiled_results.get_data_by_trimmed_uri(dim_red_uri)
    latent_vectors = dim_red_data.read().to_numpy()
    metadata = dim_red_data.metadata

    # Retrieve clustering results
    cluster_child_id = get_children_flow_run_ids(clustering_job_id)[0]
    cluster_uri = f"{USER}/{project_name}/{cluster_child_id}"
    cluster_df = tiled_results.get_data_by_trimmed_uri(cluster_uri).read()
    clusters = cluster_df["cluster_label"].tolist()

    cluster_names = {label: label for label in np.unique(clusters).astype(int)}

    # Build a brand-new scatter figure
    scatter_figure = generate_scatter_data(
        latent_vectors,
        metadata["model_parameters"]["n_components"],
        clusters=clusters,
        color_by="cluster",
        cluster_names=cluster_names,
    )

    return scatter_figure, True, True


@callback(
    Output("sidebar-offcanvas", "is_open", allow_duplicate=True),
    Output("main-display", "style"),
    Input("sidebar-view", "n_clicks"),
    State("sidebar-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, is_open):
    if is_open:
        style = {}
    else:
        style = {"padding": "0px 10px 0px 510px", "width": "100%"}
    return not is_open, style