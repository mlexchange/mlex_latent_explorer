import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify


def create_infra_state_status(title, icon, id, color):
    return dbc.Row(
        [
            dbc.Col(DashIconify(icon=icon, width=20, color=color, id=id), width="auto"),
            dbc.Col(html.Span(title, className="small")),
        ],
        className="align-items-center",
    )


def create_infra_state_details(
    tiled_results_ready=False,
    prefect_ready=False,
    prefect_worker_ready=False,
    timestamp=None,
):
    not_ready_icon = "pajamas:warning-solid"
    not_ready_color = "red"
    ready_icon = "pajamas:check-circle-filled"
    ready_color = "green"

    children = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Infrastructure", className="card-title"),
                html.P(
                    "----/--/-- --:--:--" if timestamp is None else timestamp,
                    id="infra-state-last-checked",
                    className="small text-muted",
                ),
                html.Hr(),
                create_infra_state_status(
                    "Tiled (Results)",
                    icon=ready_icon if tiled_results_ready else not_ready_icon,
                    color=ready_color if tiled_results_ready else not_ready_color,
                    id="tiled-results-ready",
                ),
                html.Hr(),
                create_infra_state_status(
                    "Prefect (Server)",
                    icon=ready_icon if prefect_ready else not_ready_icon,
                    color=ready_color if prefect_ready else not_ready_color,
                    id="prefect-ready",
                ),
                create_infra_state_status(
                    "Prefect (Worker)",
                    icon=ready_icon if prefect_worker_ready else not_ready_icon,
                    color=ready_color if prefect_worker_ready else not_ready_color,
                    id="prefect-worker-ready",
                ),
            ],
            style={"margin": "0px"},
        ),
        style={"border": "none", "width": "200px", "padding": "0px", "margin": "0px"},
    )
    return children


def create_infra_state_affix():
    return html.Div(
        style={"position": "fixed", "bottom": "20px", "right": "10px", "zIndex": 9999},
        children=[
            dbc.DropdownMenu(
                id="infra-state-summary",
                color="secondary",
                label=DashIconify(
                    icon="ph:network-fill", id="infra-state-icon", width=30
                ),
                children=[
                    dbc.DropdownMenuItem(
                        create_infra_state_details(),
                        header=False,
                        id="infra-state-details",
                        style={"border": "none", "padding": "0px"},
                    ),
                ],
            ),
            dcc.Interval(id="infra-check", interval=60000),
            dcc.Store(id="infra-state"),
        ],
    )
