import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager
import pandas as pd
import queries
import json
import diskcache
from auth import setup_basic_auth

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.QUARTZ, dbc_css],
    long_callback_manager=long_callback_manager,
    suppress_callback_exceptions=True,
    use_pages=True,
)

auth = setup_basic_auth(app)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE = {
    "marginLeft": "14rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}


def get_relative_path(page_name):
    return dash.page_registry[f"pages.{page_name}"]["relative_path"]


sidebar = html.Div(
    [
        dbc.Row(
            [
                html.Img(
                    src=app.get_asset_url("rpf_logo.png"), style={"height": "100"}
                ),
            ]
        ),
        html.Hr(),
        dbc.Select(
            id="country-select",
            size="sm",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Overview", href=get_relative_path("home"), active="exact"),
                dbc.NavLink(
                    "Education", href=get_relative_path("education"), active="exact"
                ),
                dbc.NavLink("Health", href=get_relative_path("health"), active="exact"),
                dbc.NavLink("About", href=get_relative_path("about"), active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(dash.page_container, id="page-content", style=CONTENT_STYLE)

dummy_div = html.Div(id="div-for-redirect")

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        sidebar,
        content,
        dummy_div,
        dcc.Store(id="stored-data"),
        dcc.Store(id="stored-basic-country-data"),
        dcc.Store(id="stored-data-subnational"),
        dcc.Store(id="stored-data-func"),
    ]
)


@app.callback(Output("div-for-redirect", "children"), Input("url", "pathname"))
def redirect_default(url_pathname):
    known_paths = list(p["relative_path"] for p in dash.page_registry.values())
    if url_pathname not in known_paths:
        return dcc.Location(pathname=get_relative_path("home"), id="redirect-me")
    else:
        return ""


@app.callback(Output("stored-data", "data"), Input("stored-data", "data"))
def fetch_data_once(data):
    if data is None:
        df = queries.get_expenditure_w_poverty_by_country_year()
        countries = sorted(df["country_name"].unique())
        return {
            "countries": countries,
            "expenditure_w_poverty_by_country_year": df.to_dict("records"),
        }
    return dash.no_update


@app.callback(Output("stored-data-func", "data"), Input("stored-data-func", "data"))
def fetch_func_data_once(data):
    if data is None:
        func_df = queries.get_expenditure_by_country_func_year()
        func_econ_df = queries.get_expenditure_by_country_func_econ_year()
        return {
            "expenditure_by_country_func_econ_year": func_econ_df.to_dict("records"),
            "expenditure_by_country_func_year": func_df.to_dict("records"),
        }
    return dash.no_update


@app.callback(
    Output("stored-data-subnational", "data"),
    Input("stored-data-subnational", "data"),
    Input("stored-data", "data"),
)
def fetch_subnational_data_once(data, country_data):
    countries = country_data["countries"]
    if data is None:
        df = queries.get_adm_boundaries(countries)

        boundaries_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "properties": {"country": x[0], "region": x[1]},
                    "geometry": json.loads(x[2]),
                }
                for x in zip(df.country_name, df.admin1_region, df.boundary)
            ],
        }

        subnational_poverty_df = queries.get_subnational_poverty_index(countries)
        geo1_year_df = queries.get_expenditure_by_country_geo1_year()
        return {
            "subnational_poverty_index": subnational_poverty_df.to_dict("records"),
            "boundaries": boundaries_geojson,
            "expenditure_by_country_geo1_year": geo1_year_df.to_dict("records"),
        }
    return dash.no_update


@app.callback(
    Output("country-select", "options"),
    Output("country-select", "value"),
    Input("stored-data", "data"),
)
def display_data(data):
    def get_country_select_options(countries):
        options = list({"label": c, "value": c} for c in countries)
        options[0]["selected"] = True
        return options

    if data is not None:
        countries = data["countries"]
        return get_country_select_options(countries), countries[0]
    return ["No data available"], ""


@app.callback(
    Output("stored-basic-country-data", "data"),
    Input("country-select", "options"),
    Input("stored-data-subnational", "data"),
    Input("stored-basic-country-data", "data"),
)
def fetch_country_data_once(countries, subnational_data, country_data):
    countries = [x["label"] for x in countries]
    if country_data is None:
        country_df = queries.get_basic_country_data(countries)
        country_info = country_df.set_index("country_name").T.to_dict()

        expenditure_df = pd.DataFrame(
            subnational_data["expenditure_by_country_geo1_year"],
            columns=["country_name", "year"],
        )
        poverty_df = pd.DataFrame(
            subnational_data["subnational_poverty_index"],
            columns=["country_name", "year", "poor215"],
        )

        expenditure_years = (
            expenditure_df.groupby("country_name")["year"]
            .apply(lambda x: sorted(x.unique()))
            .to_dict()
        )
        poverty_years = (
            poverty_df.groupby("country_name")["year"]
            .apply(lambda x: sorted(x.unique()))
            .to_dict()
        )

        poverty_level_stats = (
            pd.merge(country_df, poverty_df, on="country_name")
            .groupby("income_level")["poor215"]
            .agg(["min", "max"])
            .reset_index()
        )
        poverty_level_stats = (
            poverty_level_stats.set_index("income_level").apply(tuple, axis=1).to_dict()
        )

        for country, years in expenditure_years.items():
            country_info[country]["expenditure_years"] = years

        for country, years in poverty_years.items():
            country_info[country]["poverty_years"] = years

        for country, info in country_info.items():

            country_income_level = info["income_level"]
            info["poverty_bounds"] = poverty_level_stats[country_income_level]

        return {"basic_country_info": country_info}
    return dash.no_update


server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
