from dash import Output, Input, callback, html
import pandas as pd
import plotly.graph_objects as go


def format_econ_narrative(data, country_name, category):
    data = data.sort_values("year")
    start_year, latest_year = data["year"].iloc[0], data["year"].iloc[-1]

    latest_data = data[data["year"] == latest_year].squeeze()
    start_data = data[data["year"] == start_year].squeeze()
    cap_spending_pct = latest_data["Capital Spending"]
    op_spenging_pct = latest_data["Operational Spending"]
    gs_spending_pct = latest_data["Goods and Services"]
    emp_comp_pct = latest_data["Employee Compensation"]
    categories = [
        "Operational Spending",
        "Employee Compensation",
        "Goods and Services",
        "Capital Spending",
    ]
    emp_comp_threshold = 5
    near_zero_threshold = 1
    changes = {cat: latest_data[cat] - start_data[cat] for cat in categories}
    trends = {
        cat: (
            "remained stable"
            if abs(changes[cat]) < near_zero_threshold
            else "increased"
            if changes[cat] > 0
            else "decreased"
        )
        for cat in categories
    }
    operational_resources = {
        "Education": "school materials",
        "Health": "healthcare supplies",
    }
    capital_investment_targets = {
        "Education": "new schools",
        "Health": "new healthcare facilities",
    }
    essential_resources = {
        "Education": "essential teaching materials and classroom resources",
        "Health": "medical supplies and patient care resources",
    }
    support_materials = {
        "Education": "learning materials",
        "Health": "treatment support materials",
    }
    employee_compensation_narrative = (
        f" This indicates that a significant portion of spending is directed towards salaries and benefits, leaving limited room for non-salary operational costs such as {operational_resources[category]} and facility maintenance."
        if latest_data["Employee Compensation"] > 70
        else " This indicates a balanced allocation between salaries and operational resources, potentially enabling enhanced investment in resources and services that directly impact service delivery."
    )

    capital_spending_narrative = (
        f"Meanwhile, capital spending represented {cap_spending_pct}% of total {category} spending in {latest_year}, "
        + (
            "indicating potential under-investment in long-term infrastructure, which could affect future service delivery."
            if cap_spending_pct < 10
            else "which is within the expected range for social sectors but may require further prioritization based on infrastructure needs."
            if 10 <= cap_spending_pct <= 25
            else "suggesting a strong emphasis on infrastructure and capacity expansion."
        )
    )

    capital_spending_change_narrative = (
        f"Capital spending has {trends['Capital Spending']}"
        + (
            "."
            if trends["Capital Spending"] == "remained stable"
            else f" by {abs(changes['Capital Spending'])}%, "
            + (
                f"reflecting reduced investment in {capital_investment_targets[category]} and infrastructure."
                if changes["Capital Spending"] < 0
                else f"suggesting a stronger commitment to expanding and upgrading {category} facilities."
            )
        )
    )
    emp_comp_spending_change_narrative = (
        f"Employee compensation has {trends['Employee Compensation']}"
        + (
            "."
            if trends["Employee Compensation"] == "remained stable"
            else f" by {abs(changes['Employee Compensation'])}%, "
            + (
                "possibly driven by wage increases and workforce expansion."
                if changes["Employee Compensation"] > emp_comp_threshold
                else "remaining stable relative to overall spending trends."
            )
        )
    )
    gs_spending_change_narrative = (
        f"Goods and services spending has {trends['Goods and Services']}"
        + (
            "."
            if trends["Goods and Services"] == "remained stable"
            else f" by {abs(changes['Goods and Services'])}%, "
            + (
                f"potentially affecting the availability of {essential_resources[category]}."
                if changes["Goods and Services"] < 0
                else f"allowing for enhanced support for {support_materials[category]} and maintenance needs."
            )
        )
    )

    return html.Div(
        [
            html.P(
                f"In {country_name}, operational spending accounted for {op_spenging_pct}% of total {category.lower()} spending in {latest_year}, "
                f"with {emp_comp_pct}% allocated to employee compensation and {gs_spending_pct}% to goods and services."
                f"{employee_compensation_narrative}"
            ),
            html.P(f"{capital_spending_narrative}"),
            html.P(
                f"Between {start_year} and {latest_year}, the earliest and latest year for which data is available, spending patterns have changed:"
            ),
            html.Ul(
                [
                    html.Li(f"{capital_spending_change_narrative}"),
                    html.Li(f"{emp_comp_spending_change_narrative}"),
                    html.Li(f"{gs_spending_change_narrative}"),
                ]
            ),
        ]
    )


def generate_econ_figure(data, category):
    fig = go.Figure()
    for econ_category in data.columns[1:]:
        fig.add_trace(
            go.Scatter(
                x=data["year"],
                y=data[econ_category],
                mode="lines",
                line=dict(width=0.5),
                stackgroup="one",
                name=econ_category,
            )
        )
    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(
        fixedrange=True,
        showticklabels=True,
        tickmode="array",
        tickvals=[20, 40, 60, 80, 100],
    )
    fig.update_layout(
        barmode="stack",
        hovermode="x unified",
        title=f"How have expenditure priorities within {category} changed over time?",
        plot_bgcolor="white",
        xaxis_title="Year",
        yaxis_title=f"Percentage of total {category.lower()} expenditure",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=50, b=80),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0,
                y=-0.13,
                xanchor="left",
                text="Source: BOOST Database, World Bank",
                showarrow=False,
                font=dict(size=12, color="grey"),
            ),
        ],
    )
    return fig


@callback(
    [
        Output("econ-breakdown-func", "figure"),
        Output("econ-breakdown-func-narrative", "children"),
        Input("stored-data-func-econ", "data"),
        Input("country-select", "value"),
        Input("page-selector", "data"),
    ],
)
def render_econ_breakdown(data, country_name, page_category):
    df = pd.DataFrame(data["econ_expenditure_prop_by_func_country_year"])
    filtered_df = df[
        (df["country_name"] == country_name) & (df["func"] == page_category)
    ]
    pivot_df = filtered_df.pivot_table(
        index="year", columns="econ", values="proportion", aggfunc="sum", fill_value=0
    ).reset_index()

    fig = generate_econ_figure(pivot_df, page_category)
    narrative = format_econ_narrative(pivot_df, country_name, page_category)

    return fig, narrative
