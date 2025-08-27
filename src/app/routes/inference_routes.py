from flask import Blueprint, jsonify
from flask_socketio import join_room, leave_room, emit
from app.core.config import AppConfig
from app.core.genicam_service import GenICamService

inference_bp = Blueprint("inference", __name__)
app_config = AppConfig()
genicam_service = GenICamService()


def run_video_inference(engine_path):
    genicam_service.run_with_inference(engine_path)

@inference_bp.route("/stream/", methods=["POST"])
def start_normal_stream():
    try:
        is_valid, message = app_config.validate_config()
        if not is_valid:
            return jsonify({"status": "error", "message": message}), 400

        config = app_config.get_config()

        if config["input_type"] == "camera":
            genicam_service.run_normal()
            result = "Normal camera stream started"
        else:
            result = "Normal stream only available for camera input"

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500



@inference_bp.route("/stream/inference", methods=["POST"])
def start_inference_stream():
    is_valid, message = app_config.validate_config()
    if not is_valid:
        return jsonify({"status": "error", "message": message})

    config = app_config.get_config()

    if config["input_type"] == "camera":
        if config["model_selection"]:
            engine_path = f"models/{config['model_selection']}.engine"
            genicam_service.run_with_inference(engine_path)
            result = f"Inference stream started with model {config['model_selection']}"
        else:
            result = "No model selected for inference"
    else:
        result = run_video_inference(config)

    return jsonify({"status": "success", "result": result})


@inference_bp.route("/stream/stop", methods=["POST"])
def stop_stream():
    genicam_service.stop()
    return jsonify({"status": "success", "result": "Stream stopped"})
