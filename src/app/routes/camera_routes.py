from flask import Blueprint, request, jsonify
import os
from app.core.config import AppConfig
from werkzeug.utils import secure_filename
import uuid

camera_bp = Blueprint("camera", __name__)
app_config = AppConfig()


@camera_bp.route("/configure/camera", methods=["POST"])
def configure_camera():
    if "cti_file" in request.files:
        cti_file = request.files["cti_file"]
        if cti_file.filename != "":
            os.makedirs("cti", exist_ok=True)
            for old_file in os.listdir("cti"):
                if old_file.endswith(".cti"):
                    try:
                        os.remove(os.path.join("cti", old_file))
                    except Exception as e:
                        print(f"Could not remove {old_file}: {e}")
            # cti_filename = secure_filename(cti_file.filename)
            cti_filename = f"{uuid.uuid4().hex}_{secure_filename(cti_file.filename)}"
            cti_file_location = os.path.join("cti", cti_filename)
            cti_file.save(cti_file_location)
            app_config.set_camera_config(cti_file_location)

            return jsonify(
                {
                    "status": "success",
                    "message": "CTI file uploaded and camera configured",
                    "cti_file_location": cti_file_location,
                }
            )
    return jsonify({"status": "error", "message": "No valid CTI file provided"})


@camera_bp.route("/configure/video", methods=["POST"])
def configure_video():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"})

    video_file = request.files["video"]
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    app_config.set_video_config(video_path)
    return jsonify({"status": "success", "message": "Video configured"})
