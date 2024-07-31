import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = html.Div(
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Tabs(
                        id="health-tabs",
                        active_tab="health-tab-time",
                        children=[
                            dbc.Tab(label="Over Time", tab_id="health-tab-time"),
                            dbc.Tab(label="Across Space", tab_id="health-tab-space"),
                        ],
                        style={"marginBottom": "2rem"},
                    ),
                    html.Div(
                        id="health-spinner",
                        children=[
                            dbc.Spinner(
                                color="primary",
                                spinner_style={"width": "3rem", "height": "3rem"},
                            ),
                        ],
                    ),
                    html.Div(id="health-content"),
                ]
            )
        )
    ]
)
