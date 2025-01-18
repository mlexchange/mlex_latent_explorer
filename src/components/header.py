import dash_bootstrap_components as dbc
from dash import html


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
                                                dbc.Button(
                                                    className="fa fa-github",
                                                    style={
                                                        "font-size": "30px",
                                                        "margin-right": "1rem",
                                                        "color": "#00313C",
                                                        "border": "0px",
                                                        "background-color": "white",
                                                    },
                                                    href=github_url,
                                                )
                                            ),
                                            dbc.NavItem(
                                                dbc.Button(
                                                    className="fa fa-question-circle-o",
                                                    style={
                                                        "font-size": "30px",
                                                        "color": "#00313C",
                                                        "background-color": "white",
                                                        "border": "0px",
                                                    },
                                                    href="https://mlexchange.als.lbl.gov",
                                                )
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
