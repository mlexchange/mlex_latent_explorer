import io
import math
from base64 import b64encode

import numpy as np
from dash import ALL, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import get_children_flow_run_ids
from PIL import Image

from src.app_layout import DATA_TILED_KEY, NUM_IMGS_OVERVIEW, USER
from src.utils.data_utils import hash_list_of_strings, tiled_results
from src.utils.plot_utils import (
    generate_heatmap_plot,
    generate_scatter_data,
    plot_empty_heatmap,
    plot_empty_scatter,
)


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
    prevent_initial_call=True,
)
def update_project_name(data_project_dict):
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

    child_job_id = get_children_flow_run_ids(job_id)[0]
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
    Output("scatter", "clickData"),
    Input("show-feature-vectors", "value"),
    prevent_initial_call=True,
)
def clear_click_data(show_feature_vectors):
    if show_feature_vectors is False:
        return None


@callback(
    Output("heatmap", "figure", allow_duplicate=True),
    Output("stats-div", "children", allow_duplicate=True),
    Input("scatter", "clickData"),
    Input("scatter", "selectedData"),
    Input("mean-std-toggle", "value"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def update_heatmap(
    click_data,
    selected_data,
    display_option,
    data_project_dict,
):
    """
    This callback update the heatmap
    Args:
        click_data:             clicked data on scatter figure
        selected_data:          lasso or rect selected data points on scatter figure
        display_option:         option to display mean or std
        data_project_dict:      data project dictionary
    Returns:
        fig:                    updated heatmap
    """
    # user select a group of points
    if selected_data is not None and len(selected_data["points"]) > 0:
        selected_indices = [point["customdata"][0] for point in selected_data["points"]]

    # user click on a single point
    elif click_data is not None and len(click_data["points"]) > 0:
        selected_indices = [click_data["points"][0]["customdata"][0]]

    else:
        return (
            plot_empty_heatmap(),
            "Number of images selected: 0",
        )

    data_project = DataProject.from_dict(data_project_dict, api_key=DATA_TILED_KEY)
    selected_images, _ = data_project.read_datasets(selected_indices, export="pillow")

    selected_images = np.array(selected_images)

    if display_option == "mean":
        plot_data = np.mean(selected_images, axis=0)
    else:
        plot_data = np.std(selected_images, axis=0)

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
    dim_red_child_id = get_children_flow_run_ids(dimension_reduction_job_id)[0]
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
