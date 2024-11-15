import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from flask_login import current_user
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from queries import QueryService
from utils import (
    empty_plot,
    filter_country_sort_year,
    generate_error_prompt,
    get_correlation_text,
    get_percentage_change_text,
    millify,
    add_opacity,
    require_login,
)
import numpy as np
import traceback
from components.year_slider import slider, get_slider_config


db = QueryService.get_instance()

dash.register_page(__name__)

@require_login
def layout():
    return html.Div(
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
            dcc.Store(id="stored-data-education-private"),
            dcc.Store(id="stored-data-education-sub-func"),
            dcc.Store(id="stored-data-education-subnational"),
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
        learning_poverty = db.get_learning_poverty_rate()

        hd_index = db.get_hd_index(shared_data["countries"])

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
        priv_exp = db.get_edu_private_expenditure()
        return {
            "edu_private_expenditure": priv_exp.to_dict("records"),
        }
    return dash.no_update


@callback(
    Output("stored-data-education-sub-func", "data"),
    Input("stored-data-education-sub-func", "data"),
)
def fetch_edu_sub_func_data_once(edu_data):
    if edu_data is None:
        exp_by_sub_func = db.get_expenditure_by_country_sub_func_year()
        return {
            "expenditure_by_country_sub_func_year": exp_by_sub_func.to_dict("records"),
        }
    return dash.no_update


@callback(
    Output("stored-data-education-subnational", "data"),
    Input("stored-data-education-subnational", "data"),
)
def fetch_edu_subnational_data_once(edu_data):
    if edu_data is None:
        subnational_data = db.expenditure_and_outcome_by_country_geo1_func_year()

        return {
            "edu_subnational_expenditure": subnational_data.to_dict("records"),
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
                            children="Who Pays for Education?",
                        )
                    )
                ),
                dbc.Row(
                    dbc.Col([
                        html.P(
                            id="education-public-private-narrative",
                            children="loading...",
                        ),
                        html.P(
                            id="education-narrative",
                        ),
                    ])
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="education-public-private",
                            config={"displayModeBar": False},
                        ),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 6, "offset": 0},
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="education-total",
                            config={"displayModeBar": False},
                        ),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 6, "offset": 0},
                    ),
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Hr(),
                    )
                ),
                dbc.Row(
                    dbc.Col([
                        html.H3(
                            children="Public Spending & Education Outcome",
                        ),
                    ])
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="education-outcome",
                            config={"displayModeBar": False},
                        ),
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 7, "offset": 0},
                    ),
                    dbc.Col(
                        [
                            html.P(
                                children="Generally, while education outcomes related to access can be conceptually linked to the availability of public finance, results related to quality have a more complex chain of causality.",
                            ),
                            html.P(
                                id="education-outcome-measure",
                                children="",
                            ),
                            html.P(
                                id="education-outcome-narrative",
                                children="loading...",
                            ),
                        ],
                        xs={"size": 12, "offset": 0},
                        sm={"size": 12, "offset": 0},
                        md={"size": 12, "offset": 0},
                        lg={"size": 5, "offset": 0},
                    ),
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Hr(),
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        html.H3(
                            children="Public Spending Breakdown by Education Sector",
                        )
                    )
                ),
                dbc.Row(
                    [
                        # How much did the government spend on different levels of education
                        dbc.Col(
                            dcc.Graph(
                                id="education-sub-func",
                                config={"displayModeBar": False},
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 8, "offset": 0},
                        ),
                        dbc.Col(
                            html.P(
                                id="education-sub-func-narrative",
                                children="loading...",
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 4, "offset": 0},
                        ),
                    ],
                ),
            ]
        )
    elif tab == "edu-tab-space":
        return html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        html.H3(
                            id="education-subnational-title",
                            children="How does spending on education vary across regions?",
                        )
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            html.P(
                                id="education-subnational-narrative",
                                children="loading...",
                            ),
                        ]
                    )
                ),
                dbc.Row(style={"height": "20px"}),
                dbc.Row(
                    [
                        dbc.Col(width=1),
                        html.Div(
                            id="year_slider_edu_container",
                            children=[
                                dcc.Slider(
                                    id="year_slider_edu",
                                    min=0,
                                    max=0,
                                    value=None,
                                    step=None,
                                    included=False,
                                ),
                            ],
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="education-subnational",
                                config={"displayModeBar": False},
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 12, "offset": 0},
                        )
                    ]
                ),
            ],
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
        title="How has govt spending on education changed over time?",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0,
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
    spending.dropna(
        subset=["real_expenditure", "central_expenditure"], inplace=True
    )

    start_year = spending.year.min()
    end_year = spending.year.max()
    start_value = spending[spending.year == start_year].real_expenditure.values[0]
    end_value = spending[spending.year == end_year].real_expenditure.values[0]
    spending_growth_rate = ((end_value - start_value) / start_value)
    trend = 'increased' if end_value > start_value else 'decreased'
    text = f"Between {start_year} and {end_year} after adjusting for inflation, total public spending on education in {country} has {trend} from ${millify(start_value)} to ${millify(end_value)}, reflecting a growth rate of {spending_growth_rate:.0%}. "

    spending['real_central_expenditure'] = spending.real_expenditure / spending.expenditure * spending.central_expenditure
    start_value_central = spending[
        spending.year == start_year
    ].real_central_expenditure.values[0]
    end_value_central = spending[
        spending.year == end_year
    ].real_central_expenditure.values[0]

    spending_growth_rate_central = (
        (end_value_central - start_value_central) / start_value_central
    )

    text += f"In this time period, the central government's inflation-adjusted spending has {get_percentage_change_text(spending_growth_rate_central)} "

    if not np.isnan(
        spending[spending.year == start_year].decentralized_expenditure.values[0]
    ):
        spending['real_decentralized_expenditure'] = spending.real_expenditure / spending.expenditure * spending.decentralized_expenditure
        start_value_decentralized = spending[
            spending.year == start_year
        ].real_decentralized_expenditure.values[0]
        end_value_decentralized = spending[
            spending.year == end_year
        ].real_decentralized_expenditure.values[0]

        spending_growth_rate_decentralized = (
            (end_value_decentralized - start_value_decentralized)
            / start_value_decentralized
        )
        spending_change_regional = f"while the subnational government's inflation-adjusted spending has {get_percentage_change_text(spending_growth_rate_decentralized)}. "
    else:
        spending_change_regional = (
            ". The subnational government's data is not available for this period. "
        )

    text += spending_change_regional

    decentralization = spending[spending.year == end_year].expenditure_decentralization.values[0]
    if pd.isna(decentralization) or decentralization == 0:
        spending_decentralization = "The extent of education spending decentralization is unknown due to a lack of subnational public expenditure data."
    else:
        spending_decentralization = f'By {end_year}, {decentralization:.1%} of education spending has been decentralized.'
    text += spending_decentralization

    return text


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

    if df.empty:
        return (
            empty_plot("No data available for this period"),
            generate_error_prompt("DATA_UNAVAILABLE"),
        )

    fig = total_edu_figure(df)
    return fig, education_narrative(data, country)


def public_private_narrative(df, country):
    latest_year = df.year.max()
    earliest_year = df.year.min()
    text = ""
    try:
        latest_gov_share = df[df.year == latest_year].public_percentage.values[0]
        earliest_gov_share = df[df.year == earliest_year].public_percentage.values[0]
        trend = 'increased' if latest_gov_share > earliest_gov_share else 'decreased'
        household_ratio = (
            df[df.year == latest_year].real_expenditure_private.values[0]
            / df.real_expenditure_public.values[0]
        )
        if earliest_year != latest_year:
            text += f"In {country}, the government's share of spending on education {trend} from {earliest_gov_share:.0%} to {latest_gov_share:.0%} between {earliest_year} and {latest_year}. "

        text += f"For every unit of spending on education by the government, households spent {household_ratio:.1f} units in {latest_year}. "

    except IndexError:
        return generate_error_prompt("DATA_UNAVAILABLE")
    except:
        return generate_error_prompt("GENERIC_ERROR")
    return text


@callback(
    Output("education-public-private", "figure"),
    Output("education-public-private-narrative", "children"),
    Input("stored-data-education-private", "data"),
    Input("stored-data-education-total", "data"),
    Input("country-select", "value"),
)
def render_public_private_figure(private_data, public_data, country):
    if not private_data or not public_data:
        return

    fig_title = "What % was spent by the govt vs household?"

    private = pd.DataFrame(private_data["edu_private_expenditure"])
    private = filter_country_sort_year(private, country)

    public_data = pd.DataFrame(public_data["edu_public_expenditure"])
    public = filter_country_sort_year(public_data, country)

    merged = pd.merge(
        private,
        public,
        on=["year", "country_name"],
        how="inner",
        suffixes=["_private", "_public"],
    )
    merged = merged.dropna(subset=["real_expenditure_public", "real_expenditure_private"])

    if merged.empty:
        if public.empty:
            prompt = generate_error_prompt(
                "DATA_UNAVAILABLE_DATASET_NAME",
                dataset_name="Education public spending"
            )
        elif private.empty:
            prompt = generate_error_prompt(
                "DATA_UNAVAILABLE_DATASET_NAME",
                dataset_name="Education private spending"
            )
        else:
            prompt = "Available public and private spending data on education do not have an overlapping time period."
        return (empty_plot(prompt, fig_title=fig_title), prompt)

    merged["private_percentage"] = merged["real_expenditure_private"] / (
        merged["real_expenditure_private"] + merged["real_expenditure_public"]
    )
    merged["public_percentage"] = 1 - merged["private_percentage"]

    merged["real_expenditure_private_formatted"] = merged[
        "real_expenditure_private"
    ].apply(millify)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Private Expenditure",
            y=merged["year"].astype(str),
            x=merged.private_percentage,
            orientation="h",
            customdata=merged.real_expenditure_private_formatted,
            hovertemplate="%{customdata}",
            marker=dict(
                color="rgb(255, 191, 0)",
            ),
            text=merged.private_percentage,
            texttemplate="%{text:.0%}",
            textposition="auto",
        )
    )

    merged["real_expenditure_public_formatted"] = merged[
        "real_expenditure_public"
    ].apply(millify)
    fig.add_trace(
        go.Bar(
            name="Public Expenditure",
            y=merged["year"].astype(str),
            x=merged.public_percentage,
            orientation="h",
            customdata=merged.real_expenditure_public_formatted,
            hovertemplate="$%{customdata}",
            marker=dict(
                color="darkblue",
            ),
            text=merged.public_percentage,
            texttemplate="%{text:.0%}",
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1),
        title=fig_title,
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    narrative = public_private_narrative(merged, country)
    return fig, narrative

def outcome_measure(country):
    return f"To check if this is the case for {country}, we can use inflation-adjusted per capita public spending as a measure for public financial resource allocation per person on education, use school attendance rate of 6-17 year-old children to proximate access to education, and use learning poverty rate as an indicator for education quality."

def outcome_narrative(outcome_df, pov_df, expenditure_df, country):
    try:
        start_year = expenditure_df.year.min()
        end_year = expenditure_df.year.max()

        merged = pd.merge(outcome_df, expenditure_df, on=["year"], how="inner")
        x_col = {"display": "6-17 year-old school attendance", "col_name": "attendance_6to17yo"}
        y_col = {"display": "per capita public spending", "col_name": "per_capita_real_expenditure"}
        PCC = get_correlation_text(merged, x_col, y_col)

        text = f"From {start_year} to {end_year}, {PCC}"

        merged = pd.merge(pov_df, expenditure_df, on=["year"], how="inner")
        x_col = {"display": "learning poverty rate", "col_name": "learning_poverty_rate"}
        y_col = {"display": "per capita public spending", "col_name": "per_capita_real_expenditure"}
        PCC = get_correlation_text(merged, x_col, y_col)

        text += f" Meanwhile, {PCC}"
    except:
        traceback.print_exc()
        return generate_error_prompt("GENERIC_ERROR")
    return text


@callback(
    Output("education-outcome", "figure"),
    Output("education-outcome-measure", "children"),
    Output("education-outcome-narrative", "children"),
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
            name="6-17yo attendance rate",
            x=indicator.year,
            y=indicator.attendance_6to17yo,
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
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text="How has education outcome changed?",
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0,
                y=-0.2,
                text="Source: Education index measured by years of education: UNDP through GDL. <br>"
                "BOOST, CPI, Learning Poverty: World Bank; Population: UN, Eurostat",
                showarrow=False,
                font=dict(size=12),
            )
        ],
        hoverlabel_namelength=-1,
    )

    fig.update_yaxes(
        range=[0, max(pub_exp.per_capita_real_expenditure) * 1.2], secondary_y=False
    )
    fig.update_yaxes(range=[0, 1.2], tickformat='.0%', secondary_y=True)

    measure = outcome_measure(country)
    narrative = outcome_narrative(
        indicator, learning_poverty, pub_exp, country
    )
    return fig, measure, narrative


def education_sub_func_narrative(data, country):
    try:
        max_row = data.loc[data["real_expenditure"].idxmax()]
        max_sector = max_row.func_sub
        max_amount = max_row.real_expenditure

        percentage = (max_amount / data["real_expenditure"].sum()) * 100

        text = f"""
            In {country}, the government spent the most on {max_sector}, totalling {millify(max_amount)}. This is {percentage:.2f}% of the total education expenditure.
    """
    except:
        return generate_error_prompt("GENERIC_ERROR")

    return text


@callback(
    Output("education-sub-func", "figure"),
    Output("education-sub-func-narrative", "children"),
    Input("stored-data-education-sub-func", "data"),
    Input("country-select", "value"),
)
def render_education_sub_func(sub_func_data, country):
    try:
        if not sub_func_data or not country:
            return

        data = pd.DataFrame(sub_func_data["expenditure_by_country_sub_func_year"])
        data = data.loc[(data.func == "Education") & (data.year == data.latest_year)]
        data = filter_country_sort_year(data, country)

        fig = go.Figure()
        ids = ["total"]
        parents = [""]
        labels = ["Total"]
        real_expenditures = [data["real_expenditure"].sum()]
        data["func_sub"] = data["func_sub"].fillna(value="Others")
        parents_values = data.groupby("func_sub").sum(numeric_only=True).reset_index()
        customData = [millify(real_expenditures[0])]
        for _, row in parents_values.iterrows():
            parents.append("total")
            ids.append(row["func_sub"] + "-" + "all")
            labels.append(row["func_sub"])
            real_expenditures.append(row["real_expenditure"])
            customData.append(millify(row["real_expenditure"]))

        for _, row in data.iterrows():
            ids.append(row["admin0"] + "-" + row["func_sub"])
            parents.append(row["func_sub"] + "-" + "all")
            labels.append(row["admin0"])
            real_expenditures.append(row["real_expenditure"])
            customData.append(millify(row["real_expenditure"]))

        fig.add_trace(
            go.Icicle(
                ids=ids,
                labels=labels,
                parents=parents,
                values=real_expenditures,
                branchvalues="total",
                root_color="lightgrey",
                customdata=np.stack(customData),
                hovertemplate="<b>Real expenditure</b>: $%{customdata}<br>"
                + "<extra></extra>",
            )
        )
        source = f"Calculated from the latest available year: {data.year.values[0]}"
        fig.update_layout(
            plot_bgcolor="white",
            title="How much did the gov spend on different levels of education?",
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=-0,
                    y=-0.25,
                    text=source,
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
        )
    except:
        return empty_plot("No data available for this period"), generate_error_prompt(
            "DATA_UNAVAILABLE"
        )

    narrative = education_sub_func_narrative(parents_values, country)
    return fig, narrative



@callback(
    Output("education-subnational", "figure"),
    Output("education-subnational-narrative", "children"),
    Input("stored-data-education-subnational", "data"),
    Input("country-select", "value"),
    Input("year_slider_edu", "value"),
)
def render_education_sub_outcome(subnational_outcome_data, country, base_year):
    if not subnational_outcome_data or not country:
        return

    data = pd.DataFrame(subnational_outcome_data["edu_subnational_expenditure"])
    data = filter_country_sort_year(data, country)
    data = data[data["attendance"].notna()]
    data = data.loc[(data.func == "Education") & (data.year == base_year)]
    if data.empty:
        return empty_plot(
            "No attendance data available for this period"
        ), generate_error_prompt("DATA_UNAVAILABLE")
    n = data.shape[0]
    data_expenditure_sorted = data[["adm1_name", "per_capita_expenditure"]].sort_values(
        "per_capita_expenditure", ascending=False
    )
    data_outcome_sorted = data[["attendance", "adm1_name"]].sort_values(
        "attendance", ascending=False
    )
    source = list(data_expenditure_sorted.adm1_name)
    dest = list(data_outcome_sorted.adm1_name)
    node_custom_data = [
        (
            f"${millify(data_expenditure_sorted.iloc[i]['per_capita_expenditure'])}",
            data_expenditure_sorted.iloc[i]["adm1_name"],
        )
        for i in range(n)
    ]
    node_custom_data += [
        (
            f"{'{:.2f}'.format(data_outcome_sorted.iloc[i]['attendance'])}%",
            data_outcome_sorted.iloc[i]["adm1_name"],
        )
        for i in range(n)
    ]

    gradient_n = 1 if n < 6 else 2 if n < 11 else 3

    color_highs = px.colors.sequential.Oranges[-1 * gradient_n :]
    colors_lows = px.colors.sequential.Blues[-1 * gradient_n :]
    node_colors = (
        color_highs[::-1] + ["rgb(169,169,169)"] * (n - 2 * gradient_n) + colors_lows
    )
    node_colors_opaque = [add_opacity(color, 0.7) for color in node_colors]
    node_colors += [node_colors[source.index(dest[i])] for i in range(n)]

    fig = go.Figure()
    fig.add_trace(
        go.Sankey(
            node=dict(
                thickness=20,
                line=dict(color="black", width=0.2),
                label=list(source) + [name + "-" for name in list(dest)],
                y=[(i + 1) / (n + 1) for i in range(n)]
                + [(i + 1) / (n + 1) for i in range(n)],
                x=[0.1 for i in range(n)] + [0.9 for i in range(n)],
                color=node_colors,
                customdata=node_custom_data,
                hovertemplate="%{customdata[1]}:  %{customdata[0]}<extra></extra>",
            ),
            link=dict(
                source=[i for i in range(data.shape[0])],
                target=[data.shape[0] + dest.index(source[i]) for i in range(n)],
                color=node_colors_opaque,
                value=[1 for i in range(n)],
                hovertemplate="Expenditure: %{source.customdata[0]} <br /> Attendance: %{target.customdata[0]}<extra></extra>",
            ),
        )
    )

    fig.add_annotation(
        x=0.1,
        y=1,
        arrowcolor="rgba(0, 0, 0, 0)",
        text=f"<b>Per Capita Expenditure on Education</b><br> <b>{base_year}</b>",
    )
    fig.add_annotation(
        x=0.9,
        y=1,
        arrowcolor="rgba(0, 0, 0, 0)",
        text=f"<b>Attendance</b> <br> <b>{base_year}</b>",
    )

    rank_mapping = {0: "1st", 10: "10th", 20: "20th", 30: "30th", 40: "40th"}
    for i in range(0, n + 1, 10):
        fig.add_annotation(
            y=1 - ((i + 1) / (n + 1)),
            x=0.075,
            yshift=10,
            text=f"<b>{rank_mapping[i]}</b>",
            showarrow=False,
        )

    narrative = education_sub_narrative(base_year, data)
    return fig, narrative


def education_sub_narrative(year, data):

    data["ROI"] = data.attendance / data.per_capita_expenditure
    PCC = get_correlation_text(
        data,
        {"col_name": "ROI", "display": "ROI"},
        {
            "col_name": "per_capita_expenditure",
            "display": "per capita expenditure on education",
        },
    )

    narrative = f"In {year}, {PCC}"
    best_ROI = data[data["ROI"] == data.ROI.max()].adm1_name.values[0]
    worst_ROI = data[data["ROI"] == data.ROI.min()].adm1_name.values[0]

    narrative += f" Among the subnational regions, in terms of return on public spending on education measured by attendance, {best_ROI} had the highest return on investment (ROI) while {worst_ROI} had the lowest."
    return narrative


@callback(
    Output("year_slider_edu_container", "style"),
    Output("year_slider_edu", "marks"),
    Output("year_slider_edu", "value"),
    Output("year_slider_edu", "min"),
    Output("year_slider_edu", "max"),
    Output("year_slider_edu", "tooltip"),
    Input("stored-data-education-subnational", "data"),
    Input("country-select", "value"),
)
def update_education_year_range(data, country):
    data = pd.DataFrame(data["edu_subnational_expenditure"])
    data = data.loc[(data.func == "Education")]

    data = filter_country_sort_year(data, country)
    
    if data.empty:
        return {"display": "block"}, {}, 0, 0, 0, {}

    expenditure_years = list(data.year.astype("int").unique())
    data = data[data["attendance"].notna()]
    outcome_years = list(data.year.astype("int").unique())
    return get_slider_config(expenditure_years, outcome_years)
