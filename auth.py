import bcrypt
import os
from flask_login import login_user, UserMixin

USER_NAME = os.getenv("USER_NAME")
SALTED_PASSWORD = os.getenv("SALTED_PASSWORD")
CREDENTIAL_STORE = {
    USER_NAME: SALTED_PASSWORD
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

def authenticate(username, password):
        if not USER_NAME:
            login_user(User(username))
            return True

        if CREDENTIAL_STORE.get(username) is None:
            return False

        salted_password = CREDENTIAL_STORE[username]
        if bcrypt.checkpw(password.encode("utf-8"), salted_password.encode("utf-8")):
            login_user(User(username))
            return True

        return False
