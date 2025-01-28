from base64 import b64encode

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from dash_iconify import DashIconify
from plotly.io import to_image


def plot_empty_scatter():
    return go.Figure(
        go.Scattergl(mode="markers"),
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(
                l=20,
                r=20,
                b=20,
                t=20,
                pad=0,
            ),
        ),
    )


def generate_scattergl_plot(
    x_coords,
    y_coords,
    labels,
    label_to_string_map,
    show_legend=False,
    custom_indices=None,
):
    """
    Generates a two dimensional Scattergl plot.

    Parameters:
    x_coords (list): The x-coordinates of the points.
    y_coords (list): The y-coordinates of the points.
    labels (list): The labels of the points.
    label_to_string_map (dict): A mapping from labels to strings.
    show_legend (bool, optional): Whether to show a legend. Default is False.
    custom_indices (list, optional): Custom indices for the points. Default is None.

    Returns:
    go.Figure: The generated Scattergl plot.
    """
    # Create a set of unique labels
    unique_labels = set(labels)

    # Create a trace for each unique label
    traces = []
    for label in unique_labels:
        # Find the indices of the points with the current label
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]

        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        traces.append(
            go.Scattergl(
                x=trace_x,
                y=trace_y,
                customdata=np.array(trace_custom_indices).reshape(-1, 1),
                mode="markers",
                name=str(label_to_string_map[label]),
            )
        )

    # Create the plot with the scatter plot traces
    fig = go.Figure(data=traces)
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(255, 255, 255, 0.9)",
                orientation="h",
            )
        )
    return fig


def generate_scatter3d_plot(
    x_coords,
    y_coords,
    z_coords,
    labels,
    label_to_string_map,
    show_legend=False,
    custom_indices=None,
):
    """
    Generates a three-dimensional Scatter3d plot.

    Parameters:
    x_coords (list): The x-coordinates of the points.
    y_coords (list): The y-coordinates of the points.
    z_coords (list): The z-coordinates of the points.
    labels (list): The labels of the points.
    label_to_string_map (dict): A mapping from labels to strings.
    show_legend (bool, optional): Whether to show a legend. Default is False.
    custom_indices (list, optional): Custom indices for the points. Default is None.

    Returns:
    go.Figure: The generated Scatter3d plot.
    """
    # Create a set of unique labels
    unique_labels = set(labels)

    # Create a trace for each unique label
    traces = []
    for label in unique_labels:
        # Find the indices of the points with the current label
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]
        trace_z = [z_coords[i] for i in trace_indices]

        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        traces.append(
            go.Scatter3d(
                x=trace_x,
                y=trace_y,
                z=trace_z,
                customdata=np.array(trace_custom_indices).reshape(-1, 1),
                mode="markers",
                name=str(label_to_string_map[label]),
                marker=dict(size=3),
            )
        )

    # Create the plot with the Scatter3d traces
    fig = go.Figure(data=traces)
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(255, 255, 255, 0.9)",
                orientation="h",
            )
        )
    return fig


def generate_scatter_data(
    latent_vectors,
    n_components,
    cluster_selection=-1,  # All
    clusters=None,
    cluster_names=None,
    label_selection=-2,  # All
    labels=None,
    label_names=None,
    color_by=None,
):
    """
    Generate data for a plot according to the provided selection options:
    1. all clusters & all labels
    2. all clusters and selected labels
    3. all labels and selected clusters
    4. selected clusters and selected labels

    Parameters:
    latent_vectors (numpy.ndarray, Nx2, floats): [Description]
    n_components: number principal components
    cluster_selection (int): The cluster w want to select. Defaults to -1: all clusters
    clusters (numpy.ndarray, N, ints optional): The cluster number for each data point
    cluster_names (dict, optional): [Description]. A dictionary with cluster names
    label_selection (str, optional): Which label to select. Defaults to -2: all labels. -1 mean
    Unlabeled labels (numpy.ndarray, N, int, optional): The current labels Defaults to None.
    label_names (dict, optional): A dictionary that relates label number to name.
    color_by (str, optional): Determines if we color by label or cluster. Defaults to None.

    Returns:
    plotly.scattergl: A plot as specified.
    """
    vals_names = {}
    if color_by == "cluster":
        vals = clusters
        vals_names = cluster_names
    elif color_by == "label":
        vals = labels
        if label_names is not None:
            vals_names = {value: key for key, value in label_names.items()}
        vals_names[-1] = "Unlabeled"
    else:
        vals = [-1 for i in range(latent_vectors.shape[0])]
        vals_names = {a: a for a in np.unique(vals).astype(int)}

    # all clusters & all labels
    if cluster_selection == -1 and label_selection == -2:
        if n_components == 2:
            scatter_data = generate_scattergl_plot(
                latent_vectors[:, 0], latent_vectors[:, 1], vals, vals_names
            )
        else:
            scatter_data = generate_scatter3d_plot(
                latent_vectors[:, 0],
                latent_vectors[:, 1],
                latent_vectors[:, 2],
                vals,
                vals_names,
            )

    else:

        selected_indices = None
        clusters = np.array(clusters)
        labels = np.array(labels)

        # all labels and selected clusters
        if cluster_selection == -1 and label_selection != -2:
            if label_selection != -1:
                label_selection = label_names[label_selection]
            selected_indices = np.where(labels == label_selection)[0]

        # all clusters and selected labels
        elif label_selection == -2 and cluster_selection > -1:
            selected_indices = np.where(clusters == cluster_selection)[0]

        # selected clusters and selected labels
        elif label_selection != -2 and cluster_selection > -1:
            if label_selection != -1:
                selected_labels = label_names[label_selection]
                selected_indices = np.where(
                    clusters == cluster_selection and labels == selected_labels
                )[0]
            else:
                selected_indices = np.where((clusters == cluster_selection))[0]

        vals = np.array(vals)
        if n_components == 2:
            scatter_data = generate_scattergl_plot(
                latent_vectors[selected_indices, 0],
                latent_vectors[selected_indices, 1],
                vals[selected_indices],
                vals_names,
                custom_indices=selected_indices,
            )
        elif n_components == 3:
            scatter_data = generate_scatter3d_plot(
                latent_vectors[selected_indices, 0],
                latent_vectors[selected_indices, 1],
                latent_vectors[selected_indices, 2],
                vals[selected_indices],
                vals_names,
                custom_indices=selected_indices,
            )

    fig = go.Figure(scatter_data)
    fig.update_layout(
        dragmode="lasso",
        margin=go.layout.Margin(l=20, r=20, b=20, t=20, pad=0),
        legend=dict(tracegroupgap=20),
    )

    return fig


def plot_empty_heatmap():
    return go.Figure(
        go.Heatmap(),
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(
                l=20,
                r=20,
                b=20,
                t=20,
                pad=0,
            ),
        ),
    )


def generate_heatmap_plot(data):
    """
    Generates a heatmap plot.
    Args:
        data: The data to plot.
    Returns:
        Figure: The generated heatmap plot.
    """
    heatmap_data = go.Heatmap(z=data)

    # Determine the aspect ratio based on the shape of the heatmap_data's z-values
    aspect_x = 1
    aspect_y = 1
    if heatmap_data["z"] is not None:
        if heatmap_data["z"].size > 0:
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


def plot_figure(image, height=200, width=200):
    """
    Plots images in frontend
    Args:
        image:  Image to plot
    Returns:
        plot in base64 format
    """
    try:
        h, w = image.size
    except Exception:
        h, w, c = image.size
    fig = px.imshow(image, height=height, width=width * w / h)
    fig.update_xaxes(
        showgrid=False, showticklabels=False, zeroline=False, fixedrange=True
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, zeroline=False, fixedrange=True
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    try:
        fig.update_traces(dict(showscale=False, coloraxis=None))
    except Exception as e:
        print(f"plot error {e}")
    png = to_image(fig, format="jpg")
    png_base64 = b64encode(png).decode("ascii")
    return "data:image/jpg;base64,{}".format(png_base64)


def parse_contents(index):
    """
    This function creates the dash components to display thumbnail images
    Args:
        index:          Index of the dash component
    Returns:
        dash component
    """
    img_card = html.Img(
        id={"type": "thumbnail-src", "index": index},
        style={
            "height": "16vh",
            "margin": "auto",
            "display": "block",
        },
    )
    return img_card


def draw_rows(n_rows, n_cols):
    """
    This function displays the images per page.

    Args:
        n_rows: Number of rows.
        n_cols: Number of columns.

    Returns:
        A list of dbc.Row components, each containing dbc.Col components.
    """
    return [
        dbc.Row(
            [
                dbc.Col(
                    parse_contents(j * n_cols + i),
                )
                for i in range(n_cols)
            ],
            className="g-0",
        )
        for j in range(n_rows)
    ]


def generate_notification(title, color, icon, message=""):
    iconify_icon = DashIconify(
        icon=icon,
        width=24,
        height=24,
        style={"verticalAlign": "middle"},
    )
    return [
        dbc.Toast(
            id="auto-toast",
            children=[
                html.Div(
                    [
                        iconify_icon,
                        html.Span(title, style={"margin-left": "10px"}),
                    ],
                    className="d-flex align-items-center",
                ),
                html.P(message, className="mb-0"),
            ],
            duration=4000,
            is_open=True,
            color=color,
            style={
                "position": "fixed",
                "top": 66,
                "right": 10,
                "width": 350,
                "zIndex": 9999,
            },
        )
    ]
