import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import queries
from utils import filter_country_sort_year

dash.register_page(__name__)

layout = html.Div(
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Tabs(
                        id="education-tabs",
                        active_tab="edu-tab-time",
                        children=[
                            dbc.Tab(label="Over Time", tab_id="edu-tab-time"),
                            dbc.Tab(label="Across Space", tab_id="edu-tab-space"),
                        ],
                        style={"marginBottom": "2rem"},
                    ),
                    html.Div(id="education-content"),
                ]
            ),
        ),
        dcc.Store(id="stored-data-education-total"),
        dcc.Store(id="stored-data-education-outcome"),
        dcc.Store(id='stored-data-education-private')
    ]
)


@callback(
    Output("stored-data-education-total", "data"),
    Input("stored-data-education-total", "data"),
    Input("stored-data-func", "data"),
)
def fetch_edu_total_data_once(edu_data, shared_data):
    if edu_data is None:

        # filter shared data down to education specific
        exp_by_func = pd.DataFrame(shared_data["expenditure_by_country_func_year"])
        pub_exp = exp_by_func[exp_by_func.func == "Education"]

        return {
            "edu_public_expenditure": pub_exp.to_dict("records"),
        }
    return dash.no_update


@callback(
    Output("stored-data-education-outcome", "data"),
    Input("stored-data-education-outcome", "data"),
    Input("stored-data", "data"),
)
def fetch_edu_outcome_data_once(edu_data, shared_data):
    if edu_data is None:
        learning_poverty = queries.get_learning_poverty_rate()

        hd_index = queries.get_hd_index(shared_data["countries"])

        return {
            "learning_poverty": learning_poverty.to_dict("records"),
            "hd_index": hd_index.to_dict("records"),
        }
    return dash.no_update

@callback(
    Output("stored-data-education-private", "data"),
    Input("stored-data-education-private", "data"),
)
def fetch_edu_private_data_once(edu_data):
    if edu_data is None:

        # filter shared data down to education specific
        priv_exp = queries.get_edu_private_expenditure()
        return {
            "edu_private_expenditure": priv_exp.to_dict("records"),
        }
    return dash.no_update

@callback(
    Output("education-content", "children"),
    Input("education-tabs", "active_tab"),
)
def render_education_content(tab):
    if tab == "edu-tab-time":
        return html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        html.H3(
                            children="Public Spending & Education Outcome",
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
                            dcc.Graph(
                                id="education-total",
                                config={"displayModeBar": False},
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 6, "offset": 0},
                            style={"marginBottom": "3rem"},
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="education-outcome",
                                config={"displayModeBar": False},
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 6, "offset": 0},
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        # How has private expenditure vs public expenditure changed over time?
                        dbc.Col(
                            dcc.Graph(id="education-public-private", config={"displayModeBar": False},  ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 6, "offset": 0},
                            style={"marginBottom": "3rem"}
                        ),
                    ],
                ),
            ]
        )
    elif tab == "edu-tab-space":
        return html.Div(
            [
                "Geospatial viz"
                # dcc.Graph(id='overview-plot', figure=make_overview_plot(gdp, country))
            ]
        )


def total_edu_figure(df):
    fig = go.Figure()

    if df is None:
        print()
        return fig
    fig.add_trace(
        go.Scatter(
            name="Inflation Adjusted",
            x=df.year,
            y=df.real_expenditure,
            mode="lines+markers",
            marker_color="darkblue",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Central",
            x=df.year,
            y=df.central_expenditure,
            marker_color="rgb(17, 141, 255)",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Regional",
            x=df.year,
            y=df.decentralized_expenditure,
            marker_color="rgb(160, 209, 255)",
        ),
    )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        barmode="stack",
        hovermode="x",
        title="How has education expenditure changed over time?",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


def education_narrative(data, country):
    spending = pd.DataFrame(data["edu_public_expenditure"])
    spending = filter_country_sort_year(spending, country)

    start_year = spending.year.min()
    end_year = spending.year.max()
    start_value = spending[spending.year == start_year].real_expenditure.values[0]
    end_value = spending[spending.year == end_year].real_expenditure.values[0]
    spending_growth_rate = ((end_value - start_value) / start_value) * 100
    text = (
        f"After accounting for inflation, real public spending in education has increased by {spending_growth_rate:.2f}% "
        f"between {start_year} and {end_year}. \n"
        f"Generally, while education outcomes related to access can be conceptually linked to the availability of public finance, results related to quality have a more complex chain of causality."
        "\n"
    )
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
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


@callback(
    Output("education-total", "figure"),
    Output("education-narrative", "children"),
    Input("stored-data-education-total", "data"),
    Input("country-select", "value"),
)
def render_overview_total_figure(data, country):
    if data is None:
        return None
    all_countries = pd.DataFrame(data["edu_public_expenditure"])
    df = filter_country_sort_year(all_countries, country)
    fig = total_edu_figure(df)
    return fig, education_narrative(data, country)


@callback(
    Output("education-public-private", "figure"),
    Input("stored-data-education-private", "data"),
    Input("stored-data-education-total", "data"),
    Input("country-select", "value"),
)
def render_public_private_figure(private_data, public_data, country):
    if not private_data or not public_data:
        return
    private = pd.DataFrame(private_data["edu_private_expenditure"])
    private = filter_country_sort_year(private, country)

    public_data = pd.DataFrame(public_data["edu_public_expenditure"])
    public = filter_country_sort_year(public_data, country)

    merged = pd.merge(private, public, on=["year", "country_name"], how="inner", suffixes=["_private", "_public"])
    merged["private_percentage"] = merged["real_expenditure_private"] / (
        merged["real_expenditure_private"] + merged["real_expenditure_public"]
    )
    merged["public_percentage"] = merged["real_expenditure_public"] / (
        merged["real_expenditure_private"] + merged["real_expenditure_public"]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Private Expenditure",
            y=merged["year"].astype(str),
            x=merged.private_percentage,
            orientation="h",
            customdata=merged.real_expenditure_private,
            hovertemplate="%{customdata:$}",
            marker=dict(
                color="rgb(160, 209, 255)",
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Public Expenditure",
            y=merged["year"].astype(str),
            x=merged.public_percentage,
            orientation="h",
            customdata=merged.real_expenditure_public,
            hovertemplate="%{customdata:$}",
            marker=dict(
                color="rgb(17, 141, 255)",
            ),
        )
    )
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1),
        title="What % was spent by the govt vs household?",
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    fig.update_xaxes(tickformat=",.0%")

    return fig


@callback(
    Output("education-outcome", "figure"),
    Input("stored-data-education-outcome", "data"),
    Input("stored-data-education-total", "data"),
    Input("country-select", "value"),
)
def render_education_outcome(outcome_data, total_data, country):
    if not total_data or not outcome_data:
        return
    indicator = pd.DataFrame(outcome_data["hd_index"])
    indicator = filter_country_sort_year(indicator, country)
    indicator = indicator[indicator.adm1_name == "Total"]

    learning_poverty = pd.DataFrame(outcome_data["learning_poverty"])
    learning_poverty = filter_country_sort_year(learning_poverty, country)

    pub_exp = pd.DataFrame(total_data["edu_public_expenditure"])
    pub_exp = filter_country_sort_year(pub_exp, country)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            name="education index",
            x=indicator.year,
            y=indicator.education_index,
            mode="lines+markers",
            line=dict(color="MediumPurple", shape="spline", dash="dot"),
            connectgaps=True,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            name="learning poverty rate",
            x=learning_poverty.year,
            y=learning_poverty.learning_poverty_rate,
            mode="lines+markers",
            line=dict(color="deeppink", shape="spline", dash="dot"),
            connectgaps=True,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            name="inflation adjusted per capita public spending",
            x=pub_exp.year,
            y=pub_exp.per_capita_real_expenditure,
            mode="lines",
            marker_color="darkblue",
            opacity=0.6,
        ),
        secondary_y=False,
    )

    fig.update_layout(
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        title="How has education outcome changed?",
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.25,
                text="Source: Education index measured by years of education: UNDP through GDL. <br>"
                "BOOST, CPI, Learning Poverty: World Bank; Population: UN, Eurostat",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    fig.update_yaxes(
        range=[0, max(pub_exp.per_capita_real_expenditure) * 1.2], secondary_y=False
    )
    fig.update_yaxes(range=[0, 1], secondary_y=True)

    return fig
