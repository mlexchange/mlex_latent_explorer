import dash_bootstrap_components as dbc
from dash import html
from dash_iconify import DashIconify


def header(app_title, github_url):
    """
    This header will exist at the top of the webpage rather than browser tab
    Args:
        app_title:      Title of dash app
        github_url:     URL to github repo
    """
    header = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                id="logo",
                                src="assets/mlex.png",
                                height="60px",
                            ),
                            md="auto",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H3(app_title),
                                    ],
                                    id="app-title",
                                )
                            ],
                            md=True,
                            align="center",
                        ),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.NavbarToggler(id="navbar-toggler"),
                                dbc.Collapse(
                                    dbc.Nav(
                                        [
                                            dbc.NavItem(
                                                [
                                                    dbc.Button(
                                                        DashIconify(
                                                            icon="lucide:github"
                                                        ),
                                                        id="github-button",
                                                        style={
                                                            "display": "flex",
                                                            "font-size": "40px",
                                                            "padding": "5px",
                                                            "margin-right": "1rem",
                                                            "color": "#00313C",
                                                            "background-color": "white",
                                                            "border": "0px",
                                                        },
                                                        href=github_url,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Go to GitHub Repository",
                                                        target="github-button",
                                                        placement="bottom",
                                                    ),
                                                ],
                                            ),
                                            dbc.NavItem(
                                                [
                                                    dbc.Button(
                                                        DashIconify(
                                                            icon="lucide:circle-help"
                                                        ),
                                                        id="help-button",
                                                        style={
                                                            "display": "flex",
                                                            "font-size": "40px",
                                                            "padding": "5px",
                                                            "margin-right": "1rem",
                                                            "color": "#00313C",
                                                            "background-color": "white",
                                                            "border": "0px",
                                                        },
                                                        href="https://mlexchange.als.lbl.gov",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Go to Docs",
                                                        target="help-button",
                                                        placement="bottom",
                                                    ),
                                                ]
                                            ),
                                            dbc.NavItem(
                                                [
                                                    dbc.Button(
                                                        DashIconify(
                                                            icon="fluent:video-recording-20-regular",
                                                            style={"padding": "0px"},
                                                        ),
                                                        id="go-live",
                                                        style={
                                                            "display": "flex",
                                                            "font-size": "40px",
                                                            "padding": "5px",
                                                            "color": "#00313C",
                                                            "background-color": "white",
                                                            "border": "0px",
                                                        },
                                                    ),
                                                    dbc.Tooltip(
                                                        "Go to Live Mode",
                                                        id="tooltip-go-live",
                                                        target="go-live",
                                                        placement="bottom",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        navbar=True,
                                    ),
                                    id="navbar-collapse",
                                    navbar=True,
                                ),
                            ],
                            md=2,
                        ),
                    ],
                    align="center",
                ),
            ],
            fluid=True,
        ),
        dark=True,
        color="dark",
        sticky="top",
    )
    return header
