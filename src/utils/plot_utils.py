from base64 import b64encode

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from plotly.io import to_image


def plot_scatter():
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


def plot_heatmap():
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
