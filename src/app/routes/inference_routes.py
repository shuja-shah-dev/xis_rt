from flask import Blueprint, jsonify
from app.core.config import AppConfig
from app.core.genicam_service import GenICamService

inference_bp = Blueprint("inference", __name__)
app_config = AppConfig()
genicam_service = GenICamService()

def run_video_inference(config):
    return f"Running video inference with model {config['model_selection']} on video {config['video_path']}"

@inference_bp.route("/stream/normal", methods=["POST"])
def start_normal_stream():
    is_valid, message = app_config.validate_config()
    if not is_valid:
        return jsonify({"status": "error", "message": message})

    config = app_config.get_config()

    if config["input_type"] == "camera":
        genicam_service.run_normal()
        result = "Normal camera stream started"
    else:
        result = "Normal stream only available for camera input"

    return jsonify({"status": "success", "result": result})

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

def register_socketio_events(socketio_instance, genicam_service_instance):
    from flask_socketio import join_room, leave_room, emit
    
    @socketio_instance.on("join_stream", namespace="/ws")
    def on_join_stream():
        join_room("stream")
        clients = genicam_service_instance.client_joined()
        emit("status", {"message": "Joined stream room", "clients": clients})
        print(f"Client joined stream room. Total clients: {clients}")
    
    @socketio_instance.on("connect", namespace="/ws")
    def on_connect():
        join_room("stream")
        clients = genicam_service_instance.client_joined()
        emit("status", {"message": "Connected", "clients": clients})
        print(f"Client connected. Total clients: {clients}")
    
    @socketio_instance.on("disconnect", namespace="/ws")
    def on_disconnect():
        leave_room("stream")
        clients = genicam_service_instance.client_left()
        print(f"Client disconnected. Total clients: {clients}")
    
    @socketio_instance.on("set_confidence", namespace="/ws")
    def on_set_confidence(data):
        try:
            value = float(data.get("value", genicam_service_instance.conf_threshold))
            new_val = genicam_service_instance.set_confidence(value)
            emit("status", {"message": "confidence_updated", "value": new_val}, to="stream")
        except Exception as e:
            emit("status", {"message": f"error: {e}"})
    
    @socketio_instance.on("test_frame", namespace="/ws")
    def on_test_frame():
        test_payload = {
            "frame": "test_base64_string",
            "metrics": {
                "camera_fps": 30.0,
                "display_fps": 30.0,
                "infer_ms": 15.5,
                "conf_threshold": 0.6
            }
        }
        emit("stream_frame", test_payload, to="stream")