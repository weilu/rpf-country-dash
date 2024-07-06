import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


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
            dbc.Row(
                dbc.Col(
                    html.Hr(),
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.H3(
                        children="Functional Spending",
                    )
                )
            ),
            dbc.Row(
                [
                    # How has sector prioritization changed over time?
                    dbc.Col(
                        dcc.Graph(id="functional-breakdown", config={"displayModeBar": False}),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 8, "offset": 0},
                    ),
                    dbc.Col(
                        html.P(
                            id="functional-narrative",
                            children="loading...",
                        ),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 4, "offset": 0},
                    )
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


COFOG_CATS = ['Social protection', 'Recreation, culture and religion', 'Public order and safety', 'Housing and community amenities', 'Health', 'General public services', 'Environmental protection', 'Education', 'Economic affairs', 'Defence']
FUNC_PALETTE = px.colors.qualitative.T10
FUNC_COLORS = {cat: FUNC_PALETTE[i % len(FUNC_PALETTE)] for i, cat in enumerate(COFOG_CATS)}

def functional_figure(df):
    categories = sorted(df.func.unique(), reverse=True)


    fig = go.Figure()

    for cat in categories:
        cat_df = df[df.func == cat]
        fig.add_trace(
            go.Bar(
                name=cat,
                x=cat_df.year,
                y=cat_df.percentage,
                marker_color=FUNC_COLORS[cat],
                customdata=cat_df['expenditure'],
                hovertemplate=(
                    '<b>Year</b>: %{x}<br>'
                    '<b>Expenditure</b>: %{customdata:,} (%{y:.1f}%)'
                ),
            )
        )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        barmode="stack",
        title="How has sector prioritization changed over time?",
        plot_bgcolor="white",
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=-0.1,
                y=-0.2,
                text="Expenditure % by COFOG categories. Source: BOOST",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    return fig

def functional_narrative(df):
    country = df.country_name.iloc[0]
    categories = df.func.unique().tolist()
    latest = df[df.year == df.latest_year].iloc[0].to_dict()
    latest_year = latest['year']
    text = f'For {country}, BOOST provides functional spending data on {len(categories)} categories, based on Classification of the Functions of Government (COFOG). '

    if len(categories) < len(COFOG_CATS):
        missing_cats = set(COFOG_CATS) - set(categories)
        if len(missing_cats) == 1:
            text += f'The cartegory we do not have data on is {list(missing_cats)[0]}. '
        else:
            text += f'The cartegories we do not have data on include {", ".join(missing_cats)}. '

    mean_percentage = df.groupby('func')['percentage'].mean().reset_index()
    n = 3
    top_funcs = mean_percentage.sort_values(by='percentage', ascending=False).head(n)
    text += f'On average, the top {n} spending functional categories are '
    text += ', '.join([f"{row['func']} ({row['percentage']:.1f}%)" for _, row in top_funcs.iterrows()]) + "; "

    bottom_funcs = mean_percentage.sort_values(by='percentage', ascending=True).head(n)
    text += f'while the bottom {n} spenders are '
    text += ', '.join([f"{row['func']} ({row['percentage']:.1f}%)" for _, row in bottom_funcs.iterrows()]) + ". "

    std_percentage = df.groupby('func')['percentage'].std().reset_index()
    m = 2
    stable_funcs = std_percentage.sort_values(by='percentage', ascending=True).head(m)
    text += f'Relatively, public expenditure remain the most stable in '
    text += ' and '.join([f"{row['func']} (std={row['percentage']:.1f})" for _, row in stable_funcs.iterrows()]) + "; "

    flux_funcs = std_percentage.sort_values(by='percentage', ascending=False).head(m)
    text += f'while spending in '
    text += ' and '.join([f"{row['func']} (std={row['percentage']:.1f})" for _, row in flux_funcs.iterrows()])
    text += f' fluctuate the most over time. '

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


@callback(
    Output('functional-breakdown', 'figure'),
    Output('functional-narrative', 'children'),
    Input('stored-data', 'data'),
    Input('country-select', 'value'),
)
def render_overview_total_figure(data, country):
    all_countries = pd.DataFrame(data['expenditure_by_country_func_year'])
    df = all_countries[all_countries.country_name == country]

    total_per_year = df.groupby('year')['expenditure'].sum().reset_index()
    df = df.merge(total_per_year, on='year', suffixes=('', '_total'))
    df['percentage'] = (df['expenditure'] / df['expenditure_total']) * 100

    return functional_figure(df), functional_narrative(df)
