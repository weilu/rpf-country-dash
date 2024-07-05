import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = html.Div(children=[
    dbc.Card(
        dbc.CardBody([
            dbc.Tabs(id='educationn-tabs', active_tab='edu-tab-time', children=[
                dbc.Tab(label='Over Time', tab_id='edu-tab-time'),
                dbc.Tab(label='Across Space', tab_id='edu-tab-space'),
            ], style={"marginBottom": "2rem"}),
            html.Div(id='education-spinner', children=[
                dbc.Spinner(color="primary", spinner_style={
                    "width": "3rem", "height": "3rem"
                }),
            ]),
            html.Div(id='education-content'),
        ])
    )
])

