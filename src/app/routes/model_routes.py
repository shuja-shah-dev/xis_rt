from flask import Blueprint, request, jsonify
from app.core.config import AppConfig

model_bp = Blueprint("model", __name__)
app_config = AppConfig()


@model_bp.route("/select_model", methods=["POST"])
def select_model():
    data = request.get_json()
    model_name = data.get("model_name")
    valid_model_names = ["raw_dough", "baked_baguette", "generic", "donut"]
    if model_name not in valid_model_names:
        return jsonify({"status": "error", "message": "Invalid model selection"})
    try:
        app_config.set_model_selection(model_name)
        return jsonify({"status": "success", "message": f"Model {model_name} selected"})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)})
