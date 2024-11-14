import bcrypt
import os
from flask_login import login_user, UserMixin
from dash import dcc

USER_NAME = os.getenv("USER_NAME")
SALTED_PASSWORD = os.getenv("SALTED_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
CREDENTIAL_STORE = {
    USER_NAME: SALTED_PASSWORD
}

class User(UserMixin):
    # User data model. It has to have at least self.id as a minimum
    def __init__(self, username):
        self.id = username

def attempt_login(username, password):
        if CREDENTIAL_STORE.get(username) is None:
            return "Invalid username"
        salted_password = CREDENTIAL_STORE[username]
        if bcrypt.checkpw(password.encode("utf-8"), salted_password.encode("utf-8")):
            login_user(User(username))
            return dcc.Location(pathname=f"/home", id="home")
        return "Incorrect  password"