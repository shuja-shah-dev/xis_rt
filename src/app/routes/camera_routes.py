from flask import Blueprint, request, jsonify
import os
from app.core.config import AppConfig

camera_bp = Blueprint("camera", __name__)
app_config = AppConfig()


@camera_bp.route("/configure/camera", methods=["POST"])
def configure_camera():
    data = request.get_json()
    cti_file_location = data.get("cti_file_location")
    app_config.set_camera_config(cti_file_location)
    return jsonify({"status": "success", "message": "Camera configured"})


@camera_bp.route("/configure/video", methods=["POST"])
def configure_video():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"})

    video_file = request.files["video"]
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    app_config.set_video_config(video_path)
    return jsonify({"status": "success", "message": "Video configured"})
