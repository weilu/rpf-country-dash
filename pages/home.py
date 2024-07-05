import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go


dash.register_page(__name__)

layout = html.Div(children=[
    dbc.Card(
        dbc.CardBody([
            dbc.Tabs(id='overview-tabs', active_tab='overview-tab-time', children=[
                dbc.Tab(label='Over Time', tab_id='overview-tab-time'),
                dbc.Tab(label='Across Space', tab_id='overview-tab-space'),
            ], style={"marginBottom": "2rem"}),
            html.Div(id='overview-content'),
        ])
    )
])


@callback(
    Output('overview-content', 'children'),
    Input('overview-tabs', 'active_tab'),
)
def render_overview_content(tab):
    if tab == 'overview-tab-time':
        return html.Div([
            dbc.Row(
                dbc.Col(
                    html.H3(
                        children="Total Expenditure",
                    )
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.P(
                        style={"font-size": "16px", "opacity": "70%"},
                        children="TODO: narrative about total & per capita exp",
                    )
                )
            ),
            dbc.Row(
                [
                    # How has total expenditure changed over time?
                    dbc.Col(
                        dcc.Graph(id="overview-total", config={"displayModeBar": False}),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 6, "offset": 0},
                    ),
                    # How has per capita expenditure changed over time?
                    dbc.Col(
                        dcc.Graph(id="overview-per-capita", config={"displayModeBar": False}),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 6, "offset": 0},
                    ),
                ],
            ),
        ])
    elif tab == 'overview-tab-space':
        return html.Div([
            'Geospatial viz'
            # dcc.Graph(id='overview-plot', figure=make_overview_plot(gdp, country))
        ])


def total_figure(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Inflation Adjusted",
            x=df.year,
            y=df.real_expenditure,
            mode="lines+markers",
            marker_color="darkblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Central",
            x=df.year,
            y=df.expenditure - df.decentralized_expenditure,
            marker_color="rgb(17, 141, 255)",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Regional",
            x=df.year,
            y=df.decentralized_expenditure,
            marker_color="rgb(160, 209, 255)",
        )
    )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        barmode="stack",
        hovermode="x",
        title="How has total expenditure changed over time?",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1),
    )

    return fig


def per_capita_figure(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Inflation Adjusted",
            x=df.year,
            y=df.per_capita_real_expenditure,
            mode="lines+markers",
            marker_color="darkblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Per Capita Expenditure",
            x=df.year,
            y=df.per_capita_expenditure,
            marker_color="rgb(17, 141, 255)",
        )
    )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        barmode="stack",
        hovermode="x",
        title="How has per capita expenditure changed over time?",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1),
    )

    return fig

@callback(
    Output('overview-total', 'figure'),
    Output('overview-per-capita', 'figure'),
    Input('stored-data', 'data'),
    Input('country-select', 'value'),
)
def render_overview_total_figure(data, country):
    all_countries = pd.DataFrame(data['expenditure_by_country_year'])
    df = all_countries[all_countries.country_name == country]

    return total_figure(df), per_capita_figure(df)

