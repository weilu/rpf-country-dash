import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils import (
    filter_country_sort_year,
    map_center,
    filter_geojson_by_country,
    zoom,
    empty_plot,
    remove_accents,
)

dash.register_page(__name__)

layout = html.Div(
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Tabs(
                        id="overview-tabs",
                        active_tab="overview-tab-time",
                        children=[
                            dbc.Tab(label="Over Time", tab_id="overview-tab-time"),
                            dbc.Tab(label="Across Space", tab_id="overview-tab-space"),
                        ],
                        style={"marginBottom": "2rem"},
                    ),
                    html.Div(id="overview-content"),
                ]
            )
        )
    ]
)


@callback(
    Output("overview-content", "children"),
    Input("overview-tabs", "active_tab"),
)
def render_overview_content(tab):
    if tab == "overview-tab-time":
        return html.Div(
            [
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
                            dcc.Graph(
                                id="overview-total", config={"displayModeBar": False}
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 6, "offset": 0},
                        ),
                        # How has per capita expenditure changed over time?
                        dbc.Col(
                            dcc.Graph(
                                id="overview-per-capita",
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
                            dcc.Graph(
                                id="functional-breakdown",
                                config={"displayModeBar": False},
                            ),
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
                        ),
                    ],
                ),
            ]
        )
    elif tab == "overview-tab-space":
        return html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        html.H3(
                            children="Regional Expenditure",
                        )
                    )
                ),
                dbc.Row(style={"height": "20px"}),
                # "Geospatial choropleths"
                dbc.Row(
                    [
                        dbc.Col(width=1),
                        dbc.Col(
                            dcc.Slider(
                                id="year-slider",
                                min=0,
                                max=0,
                                value=None,
                                step=None,
                                included=False,
                            ),
                            width=10,
                        ),
                    ]
                ),
                dbc.Row(style={"height": "20px"}),
                dbc.Row(
                    [
                        dbc.RadioItems(
                            id="expenditure-plot-radio",
                            options=[
                                {
                                    "label": "  Per capita expenditure",
                                    "value": "percapita",
                                },
                                {
                                    "label": "  Total expenditure",
                                    "value": "total",
                                },
                            ],
                            value="percapita",
                            inline=True,
                            style={"padding": "10px"},
                            labelStyle={
                                "margin-right": "20px",
                            },
                        ),
                        # How much was spent in each region?
                        dbc.Col(
                            dcc.Graph(
                                id="subnational-spending",
                                config={"displayModeBar": False},
                            ),
                            xs={"size": 12, "offset": 0},
                            sm={"size": 12, "offset": 0},
                            md={"size": 12, "offset": 0},
                            lg={"size": 6, "offset": 0},
                        ),
                        # visualization of poverty by region
                        dbc.Col(
                            dcc.Graph(
                                id="subnational-poverty",
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
                    dbc.Col(
                        [
                            html.Br(),
                            html.P(
                                id="subnational-spending-narrative",
                                children="loading ...",
                            ),
                        ]
                    )
                ),
                html.Div(style={"height": "20px"}),
            ]
        )


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
        legend=dict(orientation="h", yanchor="bottom", y=1.03),
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


def per_capita_figure(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            name="Poverty Rate",
            x=df.year,
            y=df.poor215,
            mode="lines+markers",
            line=dict(color="darkred", shape="spline", dash="dot"),
            connectgaps=True,
            hovertemplate=("%{x}: %{y:.2f}%"),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            name="Inflation Adjusted",
            x=df.year,
            y=df.per_capita_real_expenditure,
            mode="lines+markers",
            marker_color="darkblue",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            name="Per Capita",
            x=df.year,
            y=df.per_capita_expenditure,
            marker_color="#686dc3",
        ),
        secondary_y=False,
    )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(title_text="Per Capita Expenditure", secondary_y=False)
    fig.update_yaxes(
        title_text="Poverty Rate (%)",
        secondary_y=True,
        range=[0, 100],
    )
    fig.update_layout(
        barmode="stack",
        title="How has per capita expenditure changed over time?",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.03),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.2,
                text="Source: BOOST & CPI: World Bank; Population: UN, Eurostat",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


def overview_narrative(df):
    country = df.country_name.iloc[0]
    earliest = df[df.year == df.earliest_year].iloc[0].to_dict()
    latest = df[df.year == df.latest_year].iloc[0].to_dict()
    start_year = earliest["year"]
    end_year = latest["year"]
    latest_year_with_real_exp = df[df.real_expenditure.notnull()].year.max()
    latest_real_exp = df[df.year == latest_year_with_real_exp].iloc[0].to_dict()

    total_percent_diff = (
        100
        * (latest_real_exp["real_expenditure"] - earliest["real_expenditure"])
        / earliest["real_expenditure"]
    )
    total_trend = "increased" if total_percent_diff > 0 else "decreased"

    per_capita_percent_diff = (
        100
        * (
            latest_real_exp["per_capita_real_expenditure"]
            - earliest["per_capita_real_expenditure"]
        )
        / earliest["per_capita_real_expenditure"]
    )
    per_capita_trend = "increased" if per_capita_percent_diff > 0 else "decreased"

    text = f"After accounting for inflation, total public spending has {total_trend} by {total_percent_diff:.1f}% and per capita spending has {per_capita_trend} by {per_capita_percent_diff:.1f}% between {start_year} and {latest_year_with_real_exp}. "

    decentral_mean = df.expenditure_decentralization.mean() * 100
    decentral_latest = latest["expenditure_decentralization"] * 100
    decentral_text = f"On average, {decentral_mean:.1f}% of total public spending is executed by local/regional government. "
    if decentral_latest > 0:
        decentral_text += f"In {end_year}, which is the latest year with data available, expenditure decentralization is {decentral_latest:.1f}%. "
    text += (
        decentral_text
        if decentral_mean > 0
        else f"BOOST does not have any local/regional spending data for {country}. "
    )

    return text


COFOG_CATS = [
    "Social protection",
    "Recreation, culture and religion",
    "Public order and safety",
    "Housing and community amenities",
    "Health",
    "General public services",
    "Environmental protection",
    "Education",
    "Economic affairs",
    "Defence",
]
FUNC_PALETTE = px.colors.qualitative.T10
FUNC_COLORS = {
    cat: FUNC_PALETTE[i % len(FUNC_PALETTE)] for i, cat in enumerate(COFOG_CATS)
}


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
                customdata=cat_df["expenditure"],
                hovertemplate=(
                    "<b>Year</b>: %{x}<br>"
                    "<b>Expenditure</b>: %{customdata:,} (%{y:.1f}%)"
                ),
            )
        )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        barmode="stack",
        title="How has sector prioritization changed over time?",
        plot_bgcolor="white",
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.1,
                y=-0.2,
                text="Expenditure % by COFOG categories. Source: BOOST",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


def functional_narrative(df):
    country = df.country_name.iloc[0]
    categories = df.func.unique().tolist()
    text = f"For {country}, BOOST provides functional spending data on {len(categories)} categories, based on Classification of the Functions of Government (COFOG). "

    if len(categories) < len(COFOG_CATS):
        missing_cats = set(COFOG_CATS) - set(categories)
        if len(missing_cats) == 1:
            text += f"The cartegory we do not have data on is {list(missing_cats)[0]}. "
        else:
            text += f'The cartegories we do not have data on include {", ".join(missing_cats)}. '

    mean_percentage = df.groupby("func")["percentage"].mean().reset_index()
    n = 3
    top_funcs = mean_percentage.sort_values(by="percentage", ascending=False).head(n)
    text += f"On average, the top {n} spending functional categories are "
    text += (
        ", ".join(
            [
                f"{row['func']} ({row['percentage']:.1f}%)"
                for _, row in top_funcs.iterrows()
            ]
        )
        + "; "
    )

    bottom_funcs = mean_percentage.sort_values(by="percentage", ascending=True).head(n)
    text += f"while the bottom {n} spenders are "
    text += (
        ", ".join(
            [
                f"{row['func']} ({row['percentage']:.1f}%)"
                for _, row in bottom_funcs.iterrows()
            ]
        )
        + ". "
    )

    std_percentage = df.groupby("func")["percentage"].std().reset_index()
    m = 2
    stable_funcs = std_percentage.sort_values(by="percentage", ascending=True).head(m)
    text += f"Relatively, public expenditure remain the most stable in "
    text += (
        " and ".join(
            [
                f"{row['func']} (std={row['percentage']:.1f})"
                for _, row in stable_funcs.iterrows()
            ]
        )
        + "; "
    )

    flux_funcs = std_percentage.sort_values(by="percentage", ascending=False).head(m)
    text += f"while spending in "
    text += " and ".join(
        [
            f"{row['func']} (std={row['percentage']:.1f})"
            for _, row in flux_funcs.iterrows()
        ]
    )
    text += f" fluctuate the most over time. "

    return text


def subnational_spending_narrative(
    df_spending,
    df_poverty,
    top_n=3,
    exp_thresh=0.5,
    per_capita_thresh=1000,
    corr_thresholds=(0.3, 0.7),
):
    total_expenditure = (
        df_spending.groupby("adm1_name")["expenditure"]
        .sum()
        .sort_values(ascending=False)
    )
    top_n_total = total_expenditure.head(top_n)
    total_expenditure_sum = total_expenditure.sum()
    top_n_percentage = top_n_total.sum() / total_expenditure_sum
    per_capita_expenditure = df_spending.groupby("adm1_name")[
        "per_capita_expenditure"
    ].mean()
    per_capita_range = per_capita_expenditure.max() - per_capita_expenditure.min()
    per_capita_median = per_capita_expenditure.median()
    if not df_poverty.empty:
        poverty_rates = df_poverty.groupby("region_name")["poor215"].mean()
        correlation = per_capita_expenditure.corr(poverty_rates)

    if top_n_percentage > exp_thresh:
        exp_narrative = f"The top {top_n} regions—{', '.join(top_n_total.index)}—account for {top_n_percentage:.1%} of the total government expenditure,\
            indicating a significant concentration in these areas."
    else:
        exp_narrative = f"The top {top_n} regions—{', '.join(top_n_total.index)}—account for {top_n_percentage:.1%} of the total government expenditure."

    if per_capita_range > per_capita_thresh:
        per_capita_narrative = f"Per capita spending varies widely across regions, ranging from {per_capita_expenditure.min():,.2f} \
            to {per_capita_expenditure.max():,.2f}, with a median of {per_capita_median:,.2f}. This indicates substantial variation in resource allocation per person."
    else:
        per_capita_narrative = f"Per capita spending ranges from {per_capita_expenditure.min():,.2f} to {per_capita_expenditure.max():,.2f}, \
            with a median of {per_capita_median:,.2f}. The distribution is relatively even across regions."
    if not df_poverty.empty:
        if abs(correlation) > corr_thresholds[1]:
            corr_narrative = f"The correlation between per capita spending and poverty rates is {correlation:.2f} (Pearson correlation coefficient),\
                indicating a strong inverse relationship. Higher per capita spending is generally associated with lower poverty rates."
        elif abs(correlation) > corr_thresholds[0]:
            corr_narrative = f"The correlation between per capita spending and poverty rates is {correlation:.2f} (Pearson correlation coefficient), \
                suggesting a moderate inverse relationship. Generally, higher per capita spending is associated with lower poverty, though exceptions exist."
        else:
            corr_narrative = f"The correlation between per capita spending and poverty rates is {correlation:.2f} (Pearson correlation coefficient), \
                indicating a weak inverse relationship. There is little consistent pattern between higher per capita spending and poverty rates."
    else:
        corr_narrative = ""

    return f"{exp_narrative} {per_capita_narrative} {corr_narrative}"


def regional_spending_choropleth(geojson, df, zmin, zmax, lat, lon):
    all_regions = [feature["properties"]["region"] for feature in geojson["features"]]
    regions_without_data = [r for r in all_regions if r not in df.adm1_name.values]
    df_no_data = pd.DataFrame({"region_name": regions_without_data})
    df_no_data["adm1_name"] = None
    if df.empty:
        return empty_plot("Sub-national expenditure data not available")
    country_name = df.country_name.iloc[0]
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        color="expenditure",
        locations="adm1_name",
        featureidkey="properties.region",
        # center=map_center(geojson),
        center={"lat": lat, "lon": lon},
        mapbox_style="carto-positron",
        zoom=zoom.get(country_name, 6),
        range_color=[zmin, zmax],
    )
    fig.add_trace(
        px.choropleth_mapbox(
            df_no_data,
            geojson=geojson,
            color_discrete_sequence=["rgba(211, 211, 211, 0.3)"],
            locations="region_name",
            featureidkey="properties.region",
            zoom=zoom.get(country_name, 6),
        ).data[0]
    )
    fig.update_layout(
        title="How much was spent in each region?",
        plot_bgcolor="white",
        coloraxis_colorbar=dict(
            title="",
            orientation="v",
            thickness=10,
        ),
        legend=dict(orientation="h", x=1.02, y=1, xanchor="left", yanchor="top"),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.1,
                y=-0.2,
                text="Regional spending. Source: BOOST",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    fig.update_traces(
        hovertemplate="<b>Region:</b> %{location}<br>" + "<b>Expenditure:</b> %{z}<br>"
    )

    return fig


def regional_percapita_spending_choropleth(geojson, df, zmin, zmax, lat, lon):
    all_regions = [feature["properties"]["region"] for feature in geojson["features"]]
    regions_without_data = [r for r in all_regions if r not in df.adm1_name.values]
    df_no_data = pd.DataFrame({"region_name": regions_without_data})
    df_no_data["adm1_name"] = None
    if df.empty:
        return empty_plot("Sub-national population data not available ")
    country_name = df.country_name.iloc[0]
    df = df[df.adm1_name != "Central Scope"]
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        color="per_capita_expenditure",
        locations="adm1_name",
        featureidkey="properties.region",
        center={"lat": lat, "lon": lon},
        # center=map_center(geojson),
        mapbox_style="carto-positron",
        zoom=zoom.get(country_name, 6),
        range_color=[zmin, zmax],
    )
    fig.add_trace(
        px.choropleth_mapbox(
            df_no_data,
            geojson=geojson,
            color_discrete_sequence=["rgba(211, 211, 211, 0.3)"],
            locations="region_name",
            featureidkey="properties.region",
            zoom=zoom.get(df.country_name.iloc[0], 6),
        ).data[0]
    )
    fig.update_layout(
        title="How much was spent per person in each region?",
        plot_bgcolor="white",
        coloraxis_colorbar=dict(
            title="",
            orientation="v",
            thickness=10,
        ),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0,
                y=-0.2,
                text="Per capita regional spending. Source: BOOST",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    fig.update_traces(
        hovertemplate="<b>Region:</b> %{location}<br>"
        + "<b>Per capita expenditure:</b> %{z}<br>"
        + "<extra></extra>"
    )

    return fig


def subnational_poverty_choropleth(geojson, df, zmin, zmax, lat, lon):

    if df[df.region_name != "National"].empty:
        return empty_plot("Sub-national poverty data not available")
    # TODO align accents across all datasets
    df["region_name"] = df.region_name.map(lambda x: remove_accents(x))
    poverty_col = "poor215"
    country_name = df.country_name.iloc[0]
    year = df.year.iloc[0]
    all_regions = [feature["properties"]["region"] for feature in geojson["features"]]
    regions_without_data = [r for r in all_regions if r not in df.region_name.values]
    df_no_data = pd.DataFrame({"region_name": regions_without_data})
    df_no_data[poverty_col] = None
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        color=poverty_col,
        locations="region_name",
        featureidkey="properties.region",
        center={"lat": lat, "lon": lon},
        zoom=zoom.get(country_name, 3),
        range_color=[zmin, zmax],
        mapbox_style="carto-positron",
    )
    fig.add_trace(
        px.choropleth_mapbox(
            df_no_data,
            geojson=geojson,
            color_discrete_sequence=["rgba(211, 211, 211, 0.3)"],
            locations="region_name",
            featureidkey="properties.region",
            zoom=zoom.get(country_name, 3),
        ).data[0]
    )
    fig.update_layout(
        title="What percent of the population is living in poverty?",
        plot_bgcolor="white",
        coloraxis_colorbar=dict(
            title="",
            orientation="v",
            thickness=10,
        ),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0,
                y=-0.13,
                xanchor="left",
                text=f"Displaying data from {year}",
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0,
                y=-0.2,
                xanchor="left",
                text=f"Poverty rate at $2.15 (2017 PPP). Source: SPID and GSAP, World Bank",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )
    fig.update_traces(
        hovertemplate="<b>Region:</b> %{location}<br>"
        + "<b>Poverty rate (2.15):</b> %{z}<br>"
    )

    return fig


@callback(
    Output("overview-total", "figure"),
    Output("overview-per-capita", "figure"),
    Output("overview-narrative", "children"),
    Input("stored-data", "data"),
    Input("country-select", "value"),
)
def render_overview_total_figure(data, country):
    all_countries = pd.DataFrame(data["expenditure_w_poverty_by_country_year"])
    df = filter_country_sort_year(all_countries, country)
    return total_figure(df), per_capita_figure(df), overview_narrative(df)


@callback(
    Output("functional-breakdown", "figure"),
    Output("functional-narrative", "children"),
    Input("stored-data-func", "data"),
    Input("country-select", "value"),
)
def render_overview_total_figure(data, country):
    all_countries = pd.DataFrame(data["expenditure_by_country_func_econ_year"])
    df = filter_country_sort_year(all_countries, country)
    func_df = (
        df.groupby(["year", "country_name", "func"])["expenditure"].sum().reset_index()
    )

    total_per_year = func_df.groupby("year")["expenditure"].sum().reset_index()
    func_df = func_df.merge(total_per_year, on="year", suffixes=("", "_total"))
    func_df["percentage"] = (
        func_df["expenditure"] / func_df["expenditure_total"]
    ) * 100

    return functional_figure(func_df), functional_narrative(func_df)


@callback(
    Output("year-slider", "marks"),
    Output("year-slider", "value"),
    Output("year-slider", "min"),
    Output("year-slider", "max"),
    Input("stored-basic-country-data", "data"),
    Input("country-select", "value"),
)
def update_year_range(data, country):
    data = data["basic_country_info"]
    expenditure_years = data[country].get("expenditure_years", [])
    poverty_years = data[country].get("poverty_years", [])
    common_years = set(poverty_years).intersection(set(expenditure_years))
    min_year, max_year = expenditure_years[0], expenditure_years[-1]

    marks = {
        year: (
            {"label": str(year), "style": {"color": "white"}}
            if year in common_years
            else {"label": str(year), "style": {"color": "black"}}
        )
        for year in expenditure_years
    }

    selected_year = max_year
    return marks, selected_year, min_year, max_year


@callback(
    Output("subnational-spending", "figure"),
    Input("stored-data-subnational", "data"),
    Input("stored-basic-country-data", "data"),
    Input("country-select", "value"),
    Input("expenditure-plot-radio", "value"),
    Input("year-slider", "value"),
)
def render_subnational_spending_figures(data, country_data, country, plot_type, year):
    geojson = data["boundaries"]
    lat, lon = [
        country_data["basic_country_info"][country][k]
        for k in ["latitude", "longitude"]
    ]

    filtered_geojson = filter_geojson_by_country(geojson, country)
    df = pd.DataFrame(data["expenditure_by_country_geo1_year"])
    df = filter_country_sort_year(df, country)
    df = df[df.adm1_name != "Central Scope"]
    legend_percapita_min, legend_percapita_max = (
        df.per_capita_expenditure.min(),
        df.per_capita_expenditure.max(),
    )
    legend_expenditure_min, legend_expenditure_max = (
        df.expenditure.min(),
        df.expenditure.max(),
    )
    if plot_type == "percapita":
        return regional_percapita_spending_choropleth(
            filtered_geojson,
            df[df.year == year],
            legend_percapita_min,
            legend_percapita_max,
            lat,
            lon,
        )
    else:
        return regional_spending_choropleth(
            filtered_geojson,
            df[df.year == year],
            legend_expenditure_min,
            legend_expenditure_max,
            lat,
            lon,
        )


@callback(
    Output("subnational-poverty", "figure"),
    Input("stored-data-subnational", "data"),
    Input("stored-basic-country-data", "data"),
    Input("country-select", "value"),
    Input("year-slider", "value"),
)
def render_subnational_poverty_figure(subnational_data, country_data, country, year):
    geojson = subnational_data["boundaries"]
    filtered_geojson = filter_geojson_by_country(geojson, country)
    df = pd.DataFrame(subnational_data["subnational_poverty_index"])
    df = filter_country_sort_year(df, country)
    legend_min, legend_max = country_data["basic_country_info"][country][
        "poverty_bounds"
    ]
    lat, lon = [
        country_data["basic_country_info"][country][k]
        for k in ["latitude", "longitude"]
    ]
    available_years = country_data["basic_country_info"][country]["poverty_years"]
    relevant_years = [x for x in available_years if x <= year]
    if not relevant_years:
        return empty_plot("Poverty data not available for this time period")
    return subnational_poverty_choropleth(
        filtered_geojson,
        df[df.year == relevant_years[-1]],
        legend_min,
        legend_max,
        lat,
        lon,
    )


@callback(
    Output("subnational-spending-narrative", "children"),
    Input("stored-data-subnational", "data"),
    Input("stored-basic-country-data", "data"),
    Input("country-select", "value"),
    Input("year-slider", "value"),
)
def render_subnational_spending_narrative(
    subnational_data, country_data, country, year
):
    df_poverty = pd.DataFrame(subnational_data["subnational_poverty_index"])
    df_poverty = filter_country_sort_year(df_poverty, country)
    available_years = country_data["basic_country_info"][country]["poverty_years"]
    relevant_years = [x for x in available_years if x <= year]
    if not relevant_years:
        df_poverty = pd.DataFrame()
    else:
        df_poverty = df_poverty[df_poverty.year == relevant_years[-1]]

    df_spending = pd.DataFrame(subnational_data["expenditure_by_country_geo1_year"])
    df_spending = filter_country_sort_year(df_spending, country)
    df_spending = df_spending[
        (df_spending.adm1_name != "Central Scope") & (df_spending.year == year)
    ]
    return subnational_spending_narrative(df_spending, df_poverty)
