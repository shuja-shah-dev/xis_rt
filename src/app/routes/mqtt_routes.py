from flask import Blueprint, jsonify, request
from app.core.mqtt_client import publish, get_messages
import json

mqtt_bp = Blueprint("mqtt", __name__)

@mqtt_bp.route("/status")
def status():
    return jsonify({"messages": get_messages()})

@mqtt_bp.route("/publish_detection", methods=["POST"])
def publish_detection():
    """
    Example payload:
    {
        "object": "person",
        "confidence": 0.92,
        "timestamp": "2025-08-26T15:00:00"
    }
    """
    data = request.json
    publish("detection/results", json.dumps(data))
    return jsonify({"status": "published", "data": data}), 200
