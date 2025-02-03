from dash import html

from src.components.header import header
from src.components.main_display import main_display
from src.components.sidebar import sidebar
from src.utils.model_utils import Models


def test_get_models():
    dim_reduction_models = Models(
        modelfile_path="src/assets/default_models.json",
        model_type="dimension_reduction",
    )
    assert dim_reduction_models is not None
    clustering_models = Models(
        modelfile_path="src/assets/default_models.json", model_type="clustering"
    )
    assert clustering_models is not None


def test_app_layout():
    main_display_dash = main_display()
    assert main_display_dash is not None
    sidebar_dash = (
        sidebar(
            html.Div(),
            html.Div(),
            html.Div(),
        ),
    )
    assert sidebar_dash is not None
    header_dash = (
        header(
            "MLExchange | Latent Space Explorer",
            "https://github.com/mlexchange/mlex_latent_explorer",
        ),
    )
    assert header_dash is not None
