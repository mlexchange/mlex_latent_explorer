import dash_bootstrap_components as dbc
from dash import html

def create_model_selection_dialog():
    """
    Creates a modal dialog for selecting models when entering live mode
    """
    return html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader("Select Models for Live Mode", close_button=False),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Please select models to use for processing live data stream:"
                            ),
                            html.H6("Autoencoder Model:"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Select(
                                            id="live-autoencoder-dropdown",
                                            options=[],
                                            value=None,
                                            placeholder="Select model name...",
                                        ),
                                        width=7,
                                    ),
                                    dbc.Col(
                                        dbc.Select(
                                            id="live-autoencoder-version-dropdown",
                                            options=[],
                                            value=None,
                                            placeholder="Select version...",
                                            disabled=True,
                                        ),
                                        width=5,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            html.H6("Dimension Reduction Model:"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Select(
                                            id="live-dimred-dropdown",
                                            options=[],
                                            value=None,
                                            placeholder="Select model name...",
                                        ),
                                        width=7,
                                    ),
                                    dbc.Col(
                                        dbc.Select(
                                            id="live-dimred-version-dropdown",
                                            options=[],
                                            value=None,
                                            placeholder="Select version...",
                                            disabled=True,
                                        ),
                                        width=5,
                                    ),
                                ],
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel", 
                                id="live-model-cancel", 
                                className="me-2", 
                                color="secondary"
                            ),
                            dbc.Button(
                                "Continue", 
                                id="live-model-continue", 
                                color="primary",
                            ),
                        ]
                    ),
                ],
                id="live-model-dialog",
                is_open=False,
                backdrop="static",
                keyboard=False,
                centered=True,
            ),
        ]
    )