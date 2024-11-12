from dash_auth import BasicAuth
import bcrypt
import os

USER_NAME = os.getenv("USER_NAME")
SALTED_PASSWORD = os.getenv("SALTED_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
CREDENTIAL_STORE = {
    USER_NAME: SALTED_PASSWORD
}


def authenticate(username, password):
    if username not in CREDENTIAL_STORE:
        return False
    salted_password = CREDENTIAL_STORE[username]
    return bcrypt.checkpw(password.encode("utf-8"), salted_password.encode("utf-8"))


def setup_basic_auth(app):
    if USER_NAME and SALTED_PASSWORD:
        return BasicAuth(app, auth_func=authenticate, secret_key=SECRET_KEY)
