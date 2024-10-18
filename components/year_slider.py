from dash import dcc, html
import dash_bootstrap_components as dbc


def slider(id, container_id):
    return html.Div(
        id=container_id,
        children=[
            dcc.Slider(
                id=id,
                min=0,
                max=0,
                value=None,
                step=None,
                included=False,
            ),
        ],
    )


## Helper function to create the slider configuration
# @param expenditure_years: list of years from the expenditure dataset
# @param outcome_years: list of years from the outcome dataset
# @return: slider configuration dictionary with the following
#           - style: style configuration for the slider
#           - marks: marks configuration for the slider
#           - selected_year: the selected year
#           - min_year: the minimum year
#           - max_year: the maximum year
#           - tooltip: tooltip configuration for the slider
def get_slider_config(expenditure_years, outcome_years):
    expenditure_years.sort()
    outcome_years.sort()
    if not expenditure_years:
        # default years if no data
        marks = {
            2015: {"label": "2015", "style": {"color": "black"}},
            2010: {"label": "2010", "style": {"color": "black"}},
            2021: {"label": "2021", "style": {"color": "black"}},
        }

        return (
            {"opacity": 0.5, "pointer-events": "none"},
            marks,
            2015,
            2010,
            2021,
            {"template": "data not available", "always_visible": True},
        )

    common_years = [year for year in expenditure_years if year in outcome_years]
    min_year, max_year = expenditure_years[0], expenditure_years[-1]

    marks = {
        str(year): ({"label": str(year), "style": {"color": "white"}})
        if year in common_years else {"label": str(year), "style": {"color": "black"}}
        for year in expenditure_years
    }

    selected_year = max(common_years) if common_years else max_year
    return (
        {"display": "block"},
        marks,
        selected_year,
        min_year,
        max_year,
        {},
    )
