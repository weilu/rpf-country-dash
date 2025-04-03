import bcrypt
import os
from flask_login import login_user, UserMixin, current_user
from queries import QueryService

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "False").lower() in ("true", "1", "yes")

class User(UserMixin):
    def __init__(self, username):
        self.id = username

def authenticate(username, password):
    if not AUTH_ENABLED:
        login_user(User(username))
        return True

    credential_store = QueryService.get_instance().get_user_credentials()

    salted_password = credential_store.get(username)
    if not salted_password:
        return False

    if bcrypt.checkpw(password.encode("utf-8"), salted_password.encode("utf-8")):
        login_user(User(username))
        return True

    return False

