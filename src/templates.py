import dash_bootstrap_components as dbc
from dash import html

button_howto = dbc.Button(
    className="fa fa-question-circle-o",
    style={
        "font-size": "30px",
        "margin-right": "1rem",
        "color": "#00313C",
        "background-color": "white",
        "border": "0px",
    },
    href="https://mlexchange.als.lbl.gov",
)

button_github = dbc.Button(
    className="fa fa-github",
    href="https://github.com/mlexchange/mlex_latent_explorer",
    style={
        "font-size": "30px",
        "margin-right": "1rem",
        "color": "#00313C",
        "border": "0px",
        "background-color": "white",
    },
)

button_live = dbc.Button(
    id="go-live",
    className="fa fa-play",
    style={
        "font-size": "30px",
        "color": "#00313C",
        "border": "0px",
        "background-color": "white",
    },
)


def header():
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
                                        html.H3("MLExchange | Latent Space Explorer"),
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
                                            dbc.NavItem(button_github),
                                            dbc.NavItem(button_howto),
                                            dbc.NavItem(button_live),
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
