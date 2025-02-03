import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = os.getenv("APP_PORT", "8070")

# Gunicorn configuration
bind = f"{APP_HOST}:{APP_PORT}"
workers = 4
