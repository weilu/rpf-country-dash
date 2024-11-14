from flask import Flask
from flask_login import LoginManager
from auth import User
import os

server = Flask(__name__)
server.secret_key = os.getenv("SECRET_KEY")

login_manager = LoginManager()
login_manager.login_view = "/login"
login_manager.init_app(server)

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)
