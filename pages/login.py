import dash
from dash import html, dcc, callback, Output, Input, State
from auth import authenticate

dash.register_page(__name__)

def layout():
    return html.Div(
    [
        html.H2("Please log in to continue:", id="h1"),
        html.Div(id="hidden_div_for_redirect_callback"),
        dcc.Input(placeholder="Enter your username", type="text", id="uname-box", style={"marginRight": "10px"}),
        dcc.Input(placeholder="Enter your password", type="password", id="pwd-box", style={"marginRight": "10px"}),
        html.Button(children="Login", n_clicks=0, type="submit", id="login-button"),
        html.Br(),
    ]
)

@callback(
    Output("hidden_div_for_redirect_callback", "children"),
    Input("login-button", "n_clicks"),
    State("uname-box", "value"),
    State("pwd-box", "value"),
)
def login_button_click(n_clicks, username, password):
    if n_clicks > 0:
        if authenticate(username, password):
            return dcc.Location(pathname=f"/home", id="home")
        else:
            return "Invalid credentials"
    return dash.no_update
