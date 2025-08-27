# app/routes/config_routes.py

from flask import Blueprint, jsonify
from app.core.config import CONFIG

routes = Blueprint("routes", __name__)

@routes.route("/config", methods=["GET"])
def get_config():
    return jsonify(CONFIG)
