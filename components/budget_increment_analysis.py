import numpy as np
import pandas as pd
import plotly.graph_objects as go
from constants import FUNC_COLORS
from utils import (
    filter_country_sort_year,
    empty_plot,
    generate_error_prompt,
    calculate_cagr,
)

DEFAULT_VISIBLE_CATEGORIES = [
    "Health",
    "Education",
    "General public services",
    "Overall budget",
]

def render_fig_and_narrative(data, country, exp_type, num_years):
    country_budget_changes_df = pd.DataFrame(data["expenditure_by_country_func_year"])
    country_budget_changes_df = filter_country_sort_year(
        country_budget_changes_df, country
    )
    country_budget_changes_df = country_budget_changes_df[
        country_budget_changes_df["expenditure"].notna()
        & (round(country_budget_changes_df["expenditure"]) != 0)
    ]

    overall_budget_df = country_budget_changes_df.groupby(
        ["country_name", "year"], as_index=False
    ).agg(
        {
            "expenditure": "sum",
            "domestic_funded_budget": "sum",
            "real_domestic_funded_budget": "sum",
        }
    )
    overall_budget_df["func"] = "Overall budget"
    country_budget_changes_df = pd.concat(
        [country_budget_changes_df, overall_budget_df], ignore_index=True
    )

    country_budget_changes_df = country_budget_changes_df.sort_values(
        ["country_name", "func", "year"]
    )

    for col in ["domestic_funded_budget", "real_domestic_funded_budget"]:
        prev_col = f"prev_{col}"
        yoy_col = f"yoy_{col}"
        country_budget_changes_df[prev_col] = country_budget_changes_df.groupby(
            ["country_name", "func"]
        )[col].shift(1)
        country_budget_changes_df[yoy_col] = (
            (country_budget_changes_df[col] - country_budget_changes_df[prev_col])
            / country_budget_changes_df[prev_col]
        ) * 100

    country_budget_changes_df.drop(
        columns=[
            "prev_domestic_funded_budget",
            "prev_real_domestic_funded_budget",
        ],
        inplace=True,
    )

    end_year = country_budget_changes_df["year"].max()
    start_year = end_year - num_years + 1
    country_budget_changes_df = country_budget_changes_df[
        (country_budget_changes_df["year"] >= start_year)
        & (country_budget_changes_df["year"] <= end_year)
    ]
    start_year = max(country_budget_changes_df.year.min(), end_year - num_years + 1)
    num_years = end_year - start_year + 1

    foreign_funding_isnull = (
        overall_budget_df["domestic_funded_budget"] == overall_budget_df["expenditure"]
    ).all()

    func_cagr_dict = {
        func: calculate_cagr(
            group.loc[group["year"] == start_year, exp_type].sum(),
            group.loc[group["year"] == end_year, exp_type].sum(),
            num_years,
        )
        for func, group in country_budget_changes_df.groupby("func")
    }

    valid_cagr_dict = {
        k: v for k, v in func_cagr_dict.items() if v is not None and not np.isnan(v)
    }

    if (not valid_cagr_dict) & (exp_type == "real_domestic_funded_budget"):
        return (
            empty_plot("Inflation-adjusted budget data unavailable"),
            generate_error_prompt(
                "DATA_UNAVAILABLE_DATASET_NAME",
                dataset_name="Inflation adjusted domestic funded budget",
            ),
        )

    fig = create_func_growth_figure(country_budget_changes_df, exp_type, num_years)

    highest_func_cat = max(
        (k for k in valid_cagr_dict if k != "Overall budget"), key=valid_cagr_dict.get
    )
    other_candidates = [
        k for k in valid_cagr_dict if k != "Overall budget" and k != highest_func_cat
    ]

    if other_candidates:
        lowest_func_cat = min(other_candidates, key=valid_cagr_dict.get)
    else:
        lowest_func_cat = highest_func_cat

    cagr_data = {
        "Overall budget": func_cagr_dict["Overall budget"],
        "highest": (highest_func_cat, func_cagr_dict[highest_func_cat]),
        "lowest": (lowest_func_cat, func_cagr_dict[lowest_func_cat]),
    }
    narrative = format_budget_increment_narrative(
        cagr_data, foreign_funding_isnull, exp_type, num_years=num_years
    )

    return fig, narrative

def create_func_growth_figure(df, exp_type, num_years):
    color_mapping = {
        func: FUNC_COLORS.get(func, "gray")
        for func in df["func"].unique()
    }
    color_mapping["Overall budget"] = "rgba(150, 150, 150, 0.8)"

    df.dropna(subset=[f"yoy_{exp_type}"], inplace=True)

    fig = go.Figure()
    for func, group in df.groupby("func"):
        fig.add_trace(
            go.Scatter(
                x=group["year"],
                y=group[f"yoy_{exp_type}"],
                mode="lines+markers",
                name=func,
                line=dict(
                    color=color_mapping.get(func, "gray"),
                    width=2,
                    dash="dot" if func == "Overall budget" else "solid",
                ),
                marker=dict(size=4, opacity=0.8),
                hovertemplate=(
                    "<b>Functional Category:</b> %{fullData.name}<br>"
                    "<b>Year:</b> %{x}<br>"
                    "<b>Growth Rate:</b> %{y:.1f}%<extra></extra>"
                ),
                visible="legendonly"
                if func not in DEFAULT_VISIBLE_CATEGORIES
                else True,
            )
        )

    fig.update_layout(
        title="How do budgets for functional categories fluctuate over time?",
        yaxis_title="Year-on-year growth rate (%)",
        legend_title_text="",
        hovermode="closest",
        template="plotly_white",
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.2,
                text="Source: BOOST, World Bank",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


def format_budget_increment_narrative(
    data, foreign_funding_isnull, exp_type, num_years=5, threshold=0.75
):
    budget_cagr = data["Overall budget"]
    highest_func_cat, highest_cagr = data["highest"]
    lowest_func_cat, lowest_cagr = data["lowest"]

    real_terms_phrase = (
        " in real terms. " if exp_type == "real_domestic_funded_budget" else ". "
    )

    if lowest_cagr < 0:
        lowest_phrase = f"declined by {abs(lowest_cagr):.1f}%"
    else:
        lowest_phrase = f"grew at a modest rate of {lowest_cagr:.1f}%"

    if highest_cagr > 10:
        highest_phrase = f"expanded significantly at {highest_cagr:.1f}%"
    else:
        highest_phrase = f"grew at a steady rate of {highest_cagr:.1f}%"

    if abs(highest_cagr - lowest_cagr) < threshold:
        func_comparison = (
            f"Both the {highest_func_cat} and {lowest_func_cat} categories have grown at similar rates, "
            f"with {highest_func_cat} growing at {highest_cagr:.1f}% and {lowest_func_cat} at {lowest_cagr:.1f}% per year."
        )
    elif highest_cagr > lowest_cagr:
        func_comparison = (
            f"The {highest_func_cat} category {highest_phrase}, "
            f"while the {lowest_func_cat} category {lowest_phrase}. "
            f"This might suggest a policy shift towards prioritizing the {highest_func_cat} category, if resource deployment is in line with public policy priorities."
        )
    else:
        func_comparison = (
            f"The {lowest_func_cat} category {lowest_phrase}, "
            f"outpacing the {highest_func_cat} category, which {highest_phrase}. "
            f"This suggests a greater focus on the {highest_func_cat} category, if resource deployment is in line with public policy priorities."
        )

    if foreign_funding_isnull:
        external_financing_note = (
            "This analysis currently includes external financing as the budget data used has limited granularity. "
            "It would ideally exclude external financing due to its volatility."
        )
    else:
        external_financing_note = (
            "This analysis excludes external financing as it tends to be volatile."
        )

    return (
        (
            f"Over the past {num_years} years, the national budget has grown at an average rate of {budget_cagr:.1f}% per year{real_terms_phrase}"
            f"{func_comparison} "
            f"{external_financing_note}"
        ),
    )

