import logging

import pandas as pd
import numpy as np
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

from src.utils.data_utils import get_available_experiment_uuids, tiled_results
from src.utils.plot_utils import generate_scatter_data, plot_empty_scatter

logger = logging.getLogger("lse.replay")


@callback(
    Output("experiment-uuid-dropdown", "options"),
    Output("experiment-uuid-dropdown", "value"),
    Input("refresh-experiment-uuids", "n_clicks"),
    Input("sidebar", "active_item"),  # Also load when sidebar tab is selected
    prevent_initial_call=True,
)
def load_experiment_uuids(n_clicks, active_item):
    """Load available experiment UUIDs from Tiled"""
    uuids = get_available_experiment_uuids()
    
    if uuids:
        return uuids, uuids[0]["value"]
    else:
        return [], None


@callback(
    Output("load-experiment-button", "disabled"),
    Input("experiment-uuid-dropdown", "value"),
)
def toggle_load_button(selected_uuid):
    """Enable/disable load button based on UUID selection"""
    return selected_uuid is None

