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
                        id="overview-narrative",
                        children="loading...",
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
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=-0.14,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank",
                showarrow=False,
                font=dict(size=12)
            )
        ]
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
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=-0.14,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank; Population: UN, Eurostat",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    return fig

def overview_narrative(df):
    country = df.country_name.iloc[0]
    earliest = df[df.year == df.earliest_year].iloc[0].to_dict()
    latest = df[df.year == df.latest_year].iloc[0].to_dict()
    start_year = earliest['year']
    end_year = latest['year']

    total_percent_diff = 100 * (latest['real_expenditure'] - earliest['real_expenditure']) / earliest['real_expenditure']
    total_trend = 'increased' if total_percent_diff > 0 else 'decreased'

    per_capita_percent_diff = 100 * (latest['per_capita_real_expenditure'] - earliest['per_capita_real_expenditure']) / earliest['per_capita_real_expenditure']
    per_capita_trend = 'increased' if per_capita_percent_diff > 0 else 'decreased'

    text = f'After accounting for inflation, total public spending has {total_trend} by {total_percent_diff:.1f}% and per capita spending has {per_capita_trend} by {per_capita_percent_diff:.1f}% between {start_year} and {end_year}. '

    decentral_mean = df.expenditure_decentralization.mean() * 100
    decentral_latest = latest['expenditure_decentralization'] * 100
    decentral_text = f'On average, {decentral_mean:.1f}% of total public spending is executed by local/regional government. '
    if decentral_latest > 0:
        decentral_text += f'In {end_year}, which is the latest year with data available, expenditure decentralization is {decentral_latest:.1f}%. ' 
    text += decentral_text if decentral_mean > 0 else f'BOOST does not have any local/regional spending data for {country}. '

    return text

@callback(
    Output('overview-total', 'figure'),
    Output('overview-per-capita', 'figure'),
    Output('overview-narrative', 'children'),
    Input('stored-data', 'data'),
    Input('country-select', 'value'),
)
def render_overview_total_figure(data, country):
    all_countries = pd.DataFrame(data['expenditure_by_country_year'])
    df = all_countries[all_countries.country_name == country]

    return total_figure(df), per_capita_figure(df), overview_narrative(df)

