import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import queries

dash.register_page(__name__)

layout = html.Div(children=[
    dbc.Card(
        dbc.CardBody([
            dbc.Tabs(id='education-tabs', active_tab='edu-tab-time', children=[
                dbc.Tab(label='Over Time', tab_id='edu-tab-time'),
                dbc.Tab(label='Across Space', tab_id='edu-tab-space'),
            ], style={"marginBottom": "2rem"}),
            html.Div(id='education-content'),
        ]),
    ),
    dcc.Store(id='stored-data-education')
])


@callback(
    Output('stored-data-education', 'data'),
    Input('stored-data-education', 'data')
)
def fetch_edu_data_once(data):
    if data is None:
        df = queries.get_expenditure_by_func_country_year()
        private_df = queries.get_edu_private_expenditure()
        indicator = queries.get_education_indicator()
        poverty_rate = queries.get_learning_poverty_rate()
        countries = sorted(df['country_name'].unique())
        return ({
            'countries': countries,
            'expenditure_by_func_country_year': df.to_dict('records'),
            "private_expenditure_by_func_country_year": private_df.to_dict('records'),
            "edu_indicator": indicator.to_dict('records'),
            "poverty_rate": poverty_rate.to_dict('records')
        })
    return dash.no_update

@callback(
    Output('education-content', 'children'),
    Input('education-tabs', 'active_tab'),
)
def render_education_content(tab):
    if tab == 'edu-tab-time':
        return html.Div([
            dbc.Row(
                dbc.Col(
                    html.H3(
                        children="How much has the government spent on Education?",
                    )
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.P(
                        id="education-narrative",
                        children="loading...",
                    )
                )
            ),
            dbc.Row(
                [
                    # How has total expenditure changed over time?
                    dbc.Col(
                        dcc.Graph(id="education-total", config={"displayModeBar": False},  ),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 6, "offset": 0},
                        style={"marginBottom": "3rem"}

                    ),
                    ## The graph does not look right. Hiding for now
                    # How has private expenditure vs public expenditure changed over time?
                    # dbc.Col(
                    #     dcc.Graph(id="education-public-private", config={"displayModeBar": False},  ),
                    #     xs={"size": 12, "offset": 0},
                    #     sm={"size": 12, "offset": 0},
                    #     md={"size": 12, "offset": 0},
                    #     lg={"size": 6, "offset": 0},
                    #     style={"marginBottom": "3rem"}

                    # ),
                ],
            ),
                dbc.Row(
            [
                # How has total expenditure changed over time?
                dbc.Col(
                    dcc.Graph(id="education-index", config={"displayModeBar": False},  ),
                    xs={"size": 12, "offset": 0},
                    sm={"size": 12, "offset": 0},
                    md={"size": 12, "offset": 0},
                    lg={"size": 6, "offset": 0},
                ),
                # How has private expenditure vs public expenditure changed over time?
                dbc.Col(
                    dcc.Graph(id="education-outcome", config={"displayModeBar": False}, ),
                    xs={"size": 12, "offset": 0},
                    sm={"size": 12, "offset": 0},
                    md={"size": 12, "offset": 0},
                    lg={"size": 6, "offset": 0},
                ),
            ],
        ),
        ])
    elif tab == 'edu-tab-space':
        return html.Div([
            'Geospatial viz'
            # dcc.Graph(id='overview-plot', figure=make_overview_plot(gdp, country))
        ])
    



def total_edu_figure(df):
    fig = go.Figure()

    if df is None:
        return fig
    fig.add_trace(
        go.Scatter(
            name="Inflation Adjusted",
            x=df.year,
            y=df.real_expenditure_centralized + df.real_expenditure_decentralized,
            mode="lines+markers",
            marker_color="darkblue",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Central",
            x=df.year,
            y=df.real_expenditure_centralized,
            marker_color="rgb(17, 141, 255)",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Regional",
            x=df.year,
            y=df.real_expenditure_decentralized,
            marker_color="rgb(160, 209, 255)",
        ),
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

def education_narrative(data, country):
    spending = pd.DataFrame(data['expenditure_by_func_country_year'])
    spending = spending[(spending.country_name == country) & (spending.func == "Education")]
    total_spending = spending.groupby('year').sum(numeric_only=True).reset_index()
    spending_growth_rate = 0
    start_year = total_spending.year.min()
    end_year = total_spending.year.max()
    start_value = total_spending[total_spending.year == start_year].real_expenditure.values[0]
    end_value = total_spending[total_spending.year == end_year].real_expenditure.values[0]
    spending_growth_rate = ((end_value - start_value)/ start_value) * 100
    text = f'After accounting for inflation, real public spending in education has increased by {spending_growth_rate:.2f}% '\
    f'between {start_year} and {end_year}. \n'\
    f'Generally, while education outcomes related to access can be conceptually linked to the availability of public finance, results related to quality have a more complex chain of causality.'\
    '\n'
    return text
    

def private_public_fig(public, private):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Inflation Adjusted",
            x=df.year,
            y=df.real_expenditure_centralized + df.real_expenditure_decentralized,
            mode="lines+markers",
            marker_color="darkblue",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Central",
            x=df.year,
            y=df.real_expenditure_centralized,
            marker_color="rgb(17, 141, 255)",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Regional",
            x=df.year,
            y=df.real_expenditure_decentralized,
            marker_color="rgb(160, 209, 255)",
        ),
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

@callback(
    Output('education-total', 'figure'),
    Output('education-narrative', 'children'),
    Input('stored-data-education', 'data'),
    Input('country-select', 'value'),
)
def render_overview_total_figure(data, country):
    if data is None:
        return None
    all_countries = pd.DataFrame(data['expenditure_by_func_country_year'])
    df = all_countries[all_countries.country_name == country]
    df = df[df.func == "Education"]
    central = df[df.admin0 == "Central"].groupby('year').sum(numeric_only=True)
    regional = df[df.admin0 == "Regional"].groupby('year').sum(numeric_only=True)
    merged = central.merge(regional, on = ["year"], suffixes=("_centralized", "_decentralized"))
    fig = total_edu_figure(merged.reset_index())
    return fig, education_narrative(data, country)


@callback(
    Output('education-public-private', 'figure'),
    Input('stored-data-education', 'data'),
    Input('country-select', 'value'),
)
def render_public_private_figure(data, country):
    if not data:
        return None
    private = pd.DataFrame(data['private_expenditure_by_func_country_year'])
    private = private[private.country_name == country]
    private['private_percentage'] = private['real_expenditure']/(private['real_expenditure'] + private['real_expenditure'])
    private['public_percentage'] = private['real_expenditure']/(private['real_expenditure'] + private['real_expenditure'])
    fig = go.Figure()
    fig.add_trace(go.Bar(
    name="Private Expenditure",
    y=private["year"].astype(str),
    x=private.private_percentage,
    orientation='h',
    customdata=private.real_expenditure,
    hovertemplate = '%{customdata:$}',
    marker=dict(
        color='rgb(160, 209, 255)',
    )
))
    
    fig.add_trace(go.Bar(
        name="Public Expenditure",
        y=private['year'].astype(str),
        x=private.public_percentage,
        orientation='h',
        customdata=private.real_expenditure,
        hovertemplate = '%{customdata:$}',
        marker=dict(
            color='rgb(17, 141, 255)',
        ),
    ))
    fig.update_layout(barmode='stack',
                      plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=1),
                      title="What % was spent by the govt vs household?",
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
                        ])
    fig.update_xaxes(tickformat=',.0%')

    return fig

@callback(
    Output('education-outcome', 'figure'),
    Input('stored-data-education', 'data'),
    Input('country-select', 'value'),
)
def render_education_outcome(data, country):
    if not data:
        return None
    poverty_rate = pd.DataFrame(data['poverty_rate'])
    poverty_rate = poverty_rate[poverty_rate.country_name == country].sort_values('year')


    real_expenditure = pd.DataFrame(data['expenditure_by_func_country_year'])
    real_expenditure = real_expenditure[(real_expenditure.func == "Education") & (real_expenditure.country_name == country)]
    real_expenditure = real_expenditure.groupby('year').sum(numeric_only=True).reset_index()
    fig = make_subplots( specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            name="learning poverty rate",
            x=poverty_rate.year,
            y=poverty_rate.learning_poverty_rate,
            mode="lines+markers",
            marker_color="darkblue",
        ),
        secondary_y = False
    )
    fig.add_trace(
    go.Scatter(
        name="real expenditure",
        x=real_expenditure.year,
        y=real_expenditure.real_expenditure,
        mode="lines+markers",
        marker_color="MediumPurple",
    ),
    secondary_y=True
    )
    fig.update_layout(
        plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1),
            title="What is the education outcome measured by quality?",
            annotations=[
                dict(
                    xref='paper',
                    yref='paper',
                    x=-0.14,
                    y=-0.2,
                    text="Source: UNESCO Institute of Statistics (UIS)",
                    showarrow=False,
                    font=dict(size=12)
                )]
    )
    return fig

@callback(
    Output('education-index', 'figure'),
    Input('stored-data-education', 'data'),
    Input('country-select', 'value'),
)
def render_education_index(data, country):
    if not data:
        return None
    indicator = pd.DataFrame(data['edu_indicator'])
    indicator = indicator[(indicator.country_name == country) & (indicator.adm1_name == "Total")].sort_values('year')

    real_expenditure = pd.DataFrame(data['expenditure_by_func_country_year'])
    real_expenditure = real_expenditure[(real_expenditure.func == "Education") & (real_expenditure.country_name == country)]
    real_expenditure = real_expenditure.groupby('year').sum(numeric_only=True).reset_index()
    fig = make_subplots( specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            name="education index",
            x=indicator.year,
            y=indicator.education_index,
            mode="lines+markers",
            marker_color="darkblue",
        ),
        secondary_y = False
    )
    fig.add_trace(
    go.Scatter(
        name="real expenditure",
        x=real_expenditure.year,
        y=real_expenditure.real_expenditure,
        mode="lines+markers",
        marker_color="MediumPurple",
    ),
    secondary_y=True
    )
    fig.update_layout(
        plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1),
            title="What is the education outcome measured by access?",
            annotations=[
                dict(
                    xref='paper',
                    yref='paper',
                    x=-0.14,
                    y=-0.2,
                    text="Source: UNDP through GDL. Measured by years of education",
                    showarrow=False,
                    font=dict(size=12)
                )]
    )
    return fig
