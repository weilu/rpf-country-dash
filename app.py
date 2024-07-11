import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager

import queries

import diskcache
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
    return dash.page_registry[f'pages.{page_name}']['relative_path']


sidebar = html.Div(
    [
        dbc.Row([
            html.Img(
                src=app.get_asset_url('rpf_logo.png'),
                style={'height': '100'}
            ),
        ]),

        html.Hr(),
        dbc.Select(
            id="country-select",
            size="sm",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Overview", href=get_relative_path('home'), active="exact"),
                dbc.NavLink("Education", href=get_relative_path('education'), active="exact"),
                dbc.NavLink("Health", href=get_relative_path('health'), active="exact"),
                dbc.NavLink("About", href=get_relative_path('about'), active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(dash.page_container,
    id="page-content", style=CONTENT_STYLE
)

dummy_div = html.Div(id="div-for-redirect")

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content,
    dummy_div,
    dcc.Store(id='stored-data'),
])

@app.callback(
    Output('div-for-redirect', 'children'),
    Input('url', 'pathname')
)
def redirect_default(url_pathname):
    known_paths = list(p['relative_path'] for p in dash.page_registry.values())
    if url_pathname not in known_paths:
        return dcc.Location(pathname=get_relative_path('home'), id="redirect-me")
    else:
        return ""

@app.callback(
    Output('stored-data', 'data'),
    Input('stored-data', 'data')
)
def fetch_data_once(data):
    if data is None:
        df = queries.get_expenditure_w_porverty_by_country_year()
        countries = sorted(df['country_name'].unique())
        func_econ_df = queries.get_expenditure_by_country_func_econ_year()
        return ({
            'countries': countries,
            'expenditure_w_poverty_by_country_year': df.to_dict('records'),
            'expenditure_by_country_func_econ_year': func_econ_df.to_dict('records'),
        })
    return dash.no_update

@app.callback(
    Output('country-select', 'options'),
    Output('country-select', 'value'),
    Input('stored-data', 'data')
)
def display_data(data):
    def get_country_select_options(countries):
        options = list({"label": c, "value": c} for c in countries)
        options[0]["selected"] = True
        return options

    if data is not None:
        countries = data['countries']
        return get_country_select_options(countries), countries[0]
    return ["No data available"], ""


@app.long_callback(
    Output('education-content', 'children'),
    Input('education-tabs', 'active_tab'),
    running=[
        (
            Output("education-spinner", "style"),
            {"display": "block"},
            {"display": "none"},
        ),
        (
            Output("education-content", "style"),
            {"display": "none"},
            {"display": "block"},
        ),
    ],
)
def render_education_content(tab):
    if tab == 'edu-tab-time':
        return html.Div([
            'Time series viz'
            # dcc.Graph(id='edu-plot', figure=make_edu_plot(gdp, country))
        ])
    elif tab == 'edu-tab-space':
        return html.Div([
            'Geospatial viz'
            # dcc.Graph(id='health-plot', figure=make_health_plot(gdp, country))
        ])


@app.long_callback(
    Output('health-content', 'children'),
    Input('health-tabs', 'active_tab'),
    running=[
        (
            Output("health-spinner", "style"),
            {"display": "block"},
            {"display": "none"},
        ),
        (
            Output("health-content", "style"),
            {"display": "none"},
            {"display": "block"},
        ),
    ],
)
def render_health_content(tab):
    if tab == 'health-tab-time':
        return html.Div([
            'Time series viz'
            # dcc.Graph(id='edu-plot', figure=make_edu_plot(gdp, country))
        ])
    elif tab == 'health-tab-space':
        return html.Div([
            'Geospatial viz'
            # dcc.Graph(id='health-plot', figure=make_health_plot(gdp, country))
        ])


if __name__ == '__main__':
    app.run_server(debug=True)

