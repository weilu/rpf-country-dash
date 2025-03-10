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


def render_func_subnat_overview(sub_func_data, country, selected_year):
    try:
        if not sub_func_data or not country:
            return

        data = pd.DataFrame(sub_func_data["expenditure_by_country_sub_func_year"])
        data = data.loc[(data.func == "Education") & (data.year == selected_year)]
        data = filter_country_sort_year(data, country)

        central_vs_regional = (
            data.groupby("geo0").sum(numeric_only=True).reset_index()
        )
        fig1 = go.Figure(
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
        fig1.update_layout(
            title="Where was education spending directed?",
            showlegend=True,
            plot_bgcolor="white",
        )
        parents_values = data.groupby("func_sub").sum(numeric_only=True).reset_index()

        fig2 = go.Figure()
        education_values = data.groupby("func_sub").sum(numeric_only=True).reset_index()
        ids = []
        parents = []
        labels = []
        values = []
        for _, row in education_values.iterrows():
            ids.append(row["func_sub"])
            parents.append("")
            labels.append(row["func_sub"])
            values.append(row["real_expenditure"])

        data_grouped = (
            data.groupby(["func_sub", "geo0"]).sum(numeric_only=True).reset_index()
        )

        for _, row in data_grouped.iterrows():
            ids.append(f"{row['func_sub']} - {row['geo0']}")
            parents.append(row["func_sub"])
            labels.append(row["geo0"])
            values.append(row["real_expenditure"])
        formatted_values = [millify(v) for v in values]
        fig2.add_trace(
            go.Treemap(
                ids=ids,
                labels=[f"{l}<br>{v}" for l, v in zip(labels, formatted_values)],
                parents=parents,
                values=values,
                branchvalues="total",
                textinfo="label",
                hovertemplate="<b>%{label}</b><br>Real expenditure: %{label}<extra></extra>",
            )
        )

        fig2.update_layout(
            autosize=True,
            plot_bgcolor="white",
            title="How much did the gov spend on different levels of education?",
            margin=dict(l=15, r=15, b=15),
        )

    except:
        traceback.print_exc()
        return (
            empty_plot("No data available for this period"),
            empty_plot("No data available for this period"),
            generate_error_prompt("DATA_UNAVAILABLE"),
        )

    narrative = _education_sub_func_narrative(parents_values, country)
    return fig1, fig2, narrative

def _education_sub_func_narrative(data, country):
    try:
        # Identify the highest spending education sector
        max_row = data.loc[data["real_expenditure"].idxmax()]
        max_sector = max_row.func_sub
        max_amount = max_row.real_expenditure
        total_spending = data["real_expenditure"].sum()
        percentage = (max_amount / total_spending) * 100

        # Breakdown of spending by sector
        sector_shares = (
            data.groupby("func_sub")["real_expenditure"]
            .sum()
            .apply(lambda x: (x / total_spending) * 100)
            .to_dict()
        )

        # Construct text dynamically based on available sectors
        sector_text = ". ".join(
            [
                f"{sector}: {share:.1f}%"
                for sector, share in sorted(sector_shares.items(), key=lambda x: -x[1])
            ]
        )

        # Generate the narrative
        text = f"""
        In {country}, the largest share of education expenditure is allocated to {max_sector}, 
        amounting to {millify(max_amount)}, which represents {percentage:.2f}% of the total education budget.\n


        These allocations reflect government priorities in expanding access and improving education quality. 
        Understanding how resources are distributed across different levels of education helps assess their impact 
        on outcomes such as school attendance.
        """
    except:
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

    df = pd.DataFrame(edu_subnational_data["edu_subnational_expenditure"])
    df = df[(df["country_name"] == country) & (df["year"] == year)]

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
        color="attendance",
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
        title="Subnational Attendance",
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


def render_func_subnat_rank(subnational_outcome_data, country, base_year):
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

    narrative = _func_subnat_rank_narrative(base_year, data)
    return fig, narrative


def _func_subnat_rank_narrative(year, data):
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


