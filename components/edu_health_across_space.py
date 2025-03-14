import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
from dash import html
from utils import (
    empty_plot,
    filter_country_sort_year,
    filter_geojson_by_country,
    generate_error_prompt,
    get_correlation_text,
    millify,
    add_opacity,
)


def render_func_subnat_overview(func_data, sub_func_data, country, selected_year):
    if not func_data or not sub_func_data or not country:
        return

    func = "Education"

    total_data = _subset_data(
        func_data['edu_public_expenditure'], selected_year, country, func
    )

    data = _subset_data(
        sub_func_data["expenditure_by_country_sub_func_year"], 
        selected_year, country, func
    ).sort_values(by='func_sub')

    if total_data.empty and data.empty:
        return (
            empty_plot("No data available for this period"),
            empty_plot("No data available for this period"),
            generate_error_prompt("DATA_UNAVAILABLE"),
        )

    fig1 = _central_vs_regional_fig(data)
    fig2 = _sub_func_fig(data)

    narrative = _education_sub_func_narrative(total_data, data, country, selected_year)
    return fig1, fig2, narrative

def _subset_data(stored_data, year, country, func):
    data = pd.DataFrame(stored_data)
    data = filter_country_sort_year(data, country)
    return data.loc[(data.func == func) & (data.year == year)]

def _central_vs_regional_fig(data):
    fig_title = "Where was education spending directed?"
    central_vs_regional = (
        data.groupby("geo0").sum(numeric_only=True).reset_index()
    )
    if central_vs_regional.empty:
        return empty_plot("No data available for this period", fig_title)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=central_vs_regional["geo0"],
                values=central_vs_regional["real_expenditure"],
                hole=0.5,
                marker=dict(colors=["rgb(17, 141, 255)", "rgb(160, 209, 255)"]),
                customdata=[
                    millify(x)
                    for x in np.stack(central_vs_regional["real_expenditure"])
                ],
                hovertemplate="<b>Real expenditure</b>: $%{customdata}<br>"
                + "<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=fig_title,
        showlegend=True,
        plot_bgcolor="white",
    )
    return fig

def _sub_func_fig(data):
    fig_title = "How much did the gov spend on different levels of education?"
    education_values = data.groupby("func_sub", sort=False).sum(numeric_only=True).reset_index()
 
    if education_values.empty:
        return empty_plot("No data available for this period", fig_title)

    fig = go.Figure()
    total = data.expenditure.sum()
    ids = []
    parents = []
    labels = []
    values = []
    hover_texts = []
    parent_totals = {}
    for _, row in education_values.iterrows():
        percent_of_total = (row["expenditure"] / total) * 100
        parent_totals[row["func_sub"]] = row["expenditure"]
        ids.append(row["func_sub"])
        parents.append("")
        labels.append(f"{row['func_sub']}<br>{millify(row['expenditure'])} ({percent_of_total:.0f}%)")
        values.append(row["expenditure"])
        hover_texts.append(f"Real expenditure: {millify(row['real_expenditure'])}")

    data_grouped = (
        data.groupby(["func_sub", "geo0"], sort=False).sum(numeric_only=True).reset_index()
    )

    for _, row in data_grouped.iterrows():
        parent = row["func_sub"]
        percent_of_parent = (row["expenditure"] / parent_totals[parent]) * 100 \
                if parent_totals[parent] > 0 else 0

        ids.append(f"{row['func_sub']} - {row['geo0']}")
        parents.append(parent)
        values.append(row["expenditure"])

        labels.append(f"{row['geo0']}<br>{millify(row['expenditure'])} ({percent_of_parent:.0f}%)")
        hover_texts.append(f"Real expenditure: {millify(row['real_expenditure'])}")

    fig.add_trace(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            textinfo="label",
            hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
            customdata=hover_texts,
        )
    )

    fig.update_layout(
        autosize=True,
        plot_bgcolor="white",
        title=fig_title,
        margin=dict(l=15, r=15, b=15),
    )

    return fig

def _education_sub_func_narrative(total_data, data, country, selected_year):
    try:
        total_spending = data["real_expenditure"].sum()
        regional_spending = data[data.geo0 == 'Regional'].real_expenditure.sum()
        geo_tagged = regional_spending / total_spending * 100
        decentralization = total_data.expenditure_decentralization.values[0] * 100

        text = f"In {country}, as of {selected_year}, "

        subnat_exp_available_text = f"{decentralization:.1f}% of education spending is executed by regional or local governments (decentralized spending)"
        subnat_exp_not_available_text = "we do not have data on education spending executed by regional or local governments (decentralized spending)"

        geo_exp_available_text = f", while {geo_tagged:.1f}% of education spending is geographically allocated, meaning it may be funded either centrally or regionally but is directed toward specific regions. To explore disparities in spending and education outcomes across subnational regions, we will focus on geographically allocated spending, as it provides a more complete picture of resources benefiting each region."
        geo_exp_not_available_text = ". However, data on geographically allocated spending—which would capture both central and regional spending benefiting specific locations—is not available. Ideally, we would use geographically allocated spending to analyze subnational disparities, but due to data limitations, we will use decentralized spending as a proxy."

        subnat_exp_available = not math.isnan(decentralization) and not math.isclose(decentralization, 0)
        geo_exp_available =  not math.isnan(geo_tagged) and not math.isclose(geo_tagged, decentralization)
        if subnat_exp_available and geo_exp_available:
            text += subnat_exp_available_text + geo_exp_available_text
        elif subnat_exp_available and not geo_exp_available:
            text += subnat_exp_available_text + geo_exp_not_available_text
        elif not subnat_exp_available and geo_exp_available:
            text += subnat_exp_not_available_text + geo_exp_available_text
        else:
            text += "we do not have education spending at subnational level."
    except:
        traceback.print_exc()
        return generate_error_prompt("GENERIC_ERROR")

    return text


def update_func_expenditure_map(
    edu_subnational_data,
    subnational_data,
    country_data,
    country,
    year,
    expenditure_type,
):
    if (
        not edu_subnational_data
        or not subnational_data
        or not country_data
        or not country
        or year is None
    ):
        return empty_plot("Data not available")

    df = _subset_data(
        edu_subnational_data['edu_subnational_expenditure'], year, country, 'Education'
    )

    if df.empty:
        return empty_plot("No data available for the selected year")

    if expenditure_type not in df.columns:
        return empty_plot(f"{expenditure_type} data not available")

    geojson = subnational_data["boundaries"]
    filtered_geojson = filter_geojson_by_country(geojson, country)

    lat, lon = [
        country_data["basic_country_info"][country].get(k)
        for k in ["display_lat", "display_lon"]
    ]
    zoom = country_data["basic_country_info"][country]["zoom"]

    # Identify regions without data
    all_regions = [
        feature["properties"]["region"] for feature in filtered_geojson["features"]
    ]
    regions_without_data = [r for r in all_regions if r not in df.adm1_name.values]
    df_no_data = pd.DataFrame({"region_name": regions_without_data})
    df_no_data["adm1_name"] = None

    fig = px.choropleth_mapbox(
        df,
        geojson=filtered_geojson,
        color=expenditure_type,
        locations="adm1_name",
        featureidkey="properties.region",
        center={"lat": lat, "lon": lon},
        zoom=zoom,
        mapbox_style="carto-positron",
    )

    fig.add_trace(
        px.choropleth_mapbox(
            df_no_data,
            geojson=filtered_geojson,
            color_discrete_sequence=["rgba(211, 211, 211, 0.3)"],
            locations="region_name",
            featureidkey="properties.region",
            zoom=zoom,
        ).data[0]
    )

    hover_template_str = (
        "<b>Region:</b> %{location}<br>"
        f"<b>{expenditure_type.replace('_', ' ').title()}:</b> %{{z:,.2f}}<br>"
        "<extra></extra>"
    )

    fig.update_traces(hovertemplate=hover_template_str)

    fig.update_layout(
        title="Subnational Education Spending",
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
                text="Source: BOOST Database, World Bank",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )

    return fig

def update_hd_index_map(
    edu_outcome_data, subnational_data, country_data, country, year
):
    if (
        not edu_outcome_data
        or not subnational_data
        or not country_data
        or not country
        or year is None
    ):
        return empty_plot("Data not available")

    print(
        f"[DEBUG]: {pd.DataFrame(edu_outcome_data['edu_subnational_expenditure']).columns}"
    )

    df = pd.DataFrame(edu_outcome_data["edu_subnational_expenditure"])
    df = df[(df["country_name"] == country) & (df["year"] == year)]

    if df.empty:
        return empty_plot("No data available for the selected year")

    geojson = subnational_data["boundaries"]
    filtered_geojson = filter_geojson_by_country(geojson, country)

    lat, lon = [
        country_data["basic_country_info"][country].get(k)
        for k in ["display_lat", "display_lon"]
    ]
    zoom = country_data["basic_country_info"][country]["zoom"]

    # Identify regions without data
    all_regions = [
        feature["properties"]["region"] for feature in filtered_geojson["features"]
    ]
    regions_without_data = [r for r in all_regions if r not in df.adm1_name.values]
    df_no_data = pd.DataFrame({"region_name": regions_without_data})
    df_no_data["adm1_name"] = None

    # Create the choropleth for education index
    fig = px.choropleth_mapbox(
        df,
        geojson=filtered_geojson,
        color="outcome_index",
        locations="adm1_name",
        featureidkey="properties.region",
        center={"lat": lat, "lon": lon},
        zoom=zoom,
        mapbox_style="carto-positron",
    )

    fig.add_trace(
        px.choropleth_mapbox(
            df_no_data,
            geojson=filtered_geojson,
            color_discrete_sequence=["rgba(211, 211, 211, 0.3)"],
            locations="region_name",
            featureidkey="properties.region",
            zoom=zoom,
        ).data[0]
    )

    fig.update_traces(
        hovertemplate="<b>Region:</b> %{location}<br>"
        + "<b>Attendance:</b> %{z}<br>"
        + "<extra></extra>"
    )

    fig.update_layout(
        title="Subnational School Attendance",
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
                text="Source: UNDP through Global Data Lab",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )

    return fig


def render_func_subnat_rank(subnational_outcome_data, country, base_year):
    if not subnational_outcome_data or not country:
        return

    data = pd.DataFrame(subnational_outcome_data["edu_subnational_expenditure"])
    data = filter_country_sort_year(data, country)
    data = data[data["outcome_index"].notna()]
    data = data.loc[(data.func == "Education") & (data.year == base_year)]
    if data.empty:
        return empty_plot(
            "No attendance data available for this period"
        ), generate_error_prompt("DATA_UNAVAILABLE")

    data['attendance'] = data.outcome_index * 100

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

    narrative = _func_subnat_rank_narrative(base_year, data)
    return fig, narrative


def _func_subnat_rank_narrative(year, data):
    PCC = get_correlation_text(
        data,
        {
            "col_name": "outcome_index",
            "display": "6-17yo school attendance"
        },
        {
            "col_name": "per_capita_expenditure",
            "display": "per capita expenditure on education",
        },
    )

    narrative = f"In {year}, {PCC}"
    data["ROI"] = data.outcome_index / data.per_capita_expenditure
    best_ROI = data[data["ROI"] == data.ROI.max()].adm1_name.values[0]
    worst_ROI = data[data["ROI"] == data.ROI.min()].adm1_name.values[0]

    narrative += f" Among the subnational regions, in terms of return on public spending on education measured by attendance, {best_ROI} had the highest return on investment (ROI) while {worst_ROI} had the lowest."
    return narrative


