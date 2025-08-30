# app/routes/inference_routes.py
from flask import Blueprint, jsonify, request
from app.core.config import AppConfig
import threading
import time
import os
import traceback
from app import start_genicam_service_async, stop_genicam_service


inference_bp = Blueprint("inference", __name__)
app_config = AppConfig()


def get_genicam_service():
    """Get the GenICam service from app initialization"""
    from app import get_genicam_service as get_app_service

    return get_app_service()


def get_video_service_rd():
    from app import video_service

    return video_service()


def run_video_inference(config):
    """Run video inference - placeholder for video processing"""
    return f"Running video inference with model {config['model_selection']} on video {config['video_path']}"


@inference_bp.route("/stream/normal", methods=["POST"])
def start_normal_stream():
    """Start normal camera stream"""
    try:
        print("Starting normal stream request...")

        is_valid, message = app_config.validate_config()
        if not is_valid:
            print(f"Configuration validation failed: {message}")
            return jsonify({"status": "error", "message": message}), 400

        config = app_config.get_config()

        if config["input_type"] == "video":

            RUNTIME_CONFIG = {
                "engine_path": config.get("VIDEO_ENGINE_PATH"),
                "video_path": config.get("VIDEO_INPUT_PATH"),
                "score_threshold": 0.4,
                "score_class0": None,
                "score_class1": None,
                "nms_threshold": 0.5,
                "mask_threshold": 0.4,
                "canvas_size": 640,
                "alpha": 0.3,
                "min_inference_frames": 30,
                "target_fps": 30.0,
                "mqtt_topic": "detection/results",
                "video_mode": config.get("VIDEO_MODE"),
            }
            if RUNTIME_CONFIG.get("video_mode") == "raw_dough":
                video_service = get_video_service_rd()
                if video_service.initialize_from_config(RUNTIME_CONFIG):
                    if video_service.start_processing():
                        return (
                            jsonify(
                                {
                                    "status": "success",
                                    "message": "Video processing started",
                                }
                            ),
                            200,
                        )
                    else:
                        return (
                            jsonify(
                                {
                                    "status": "error",
                                    "message": "Could not start processing",
                                }
                            ),
                            500,
                        )
                else:
                    return (
                        jsonify(
                            {"status": "error", "message": "Initialization failed"}
                        ),
                        500,
                    )
            elif RUNTIME_CONFIG.get("video_mode") == "baked_baguette":
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Initialization failed, raw_dough is not defined yet",
                        }
                    ),
                    400,
                )

        else:

            model_path = app_config.get_modlel_path()
            cti_file_path = app_config.get_cti()

            genicam_service = get_genicam_service()
            if genicam_service is None:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Failed to initialize GenICam service",
                        }
                    ),
                    500,
                )

            if genicam_service.app_config is None:
                print("Warning: app_config still None, setting it now...")
                genicam_service.set_app_config(app_config)

            if config["input_type"] == "camera":
                print("Attempting to start normal camera stream...")
                success = start_genicam_service_async(model_path, cti_file_path)

                if success:
                    result = "Normal camera stream started successfully"
                    status_code = 200

                else:
                    result = "Failed to start normal camera stream"
                    status_code = 500
                    print(result)
            else:
                result = "Normal stream only available for camera input"
                status_code = 400
                print(result)

            return (
                jsonify(
                    {
                        "status": "success" if status_code == 200 else "error",
                        "message": result,
                        "service_status": genicam_service.get_status(),
                    }
                ),
                status_code,
            )

    except Exception as e:
        error_msg = f"Failed to start normal stream: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"status": "error", "message": error_msg}), 500


@inference_bp.route("/stream/stop", methods=["POST"])
def stop_stream():
    """Stop camera streaming"""
    try:
        print("Stopping stream request...")

        genicam_service = get_genicam_service()
        success = False
        try:

            def stop_async():
                try:
                    stop_genicam_service()
                except Exception as e:
                    print(f"Background stop error: {e}")

            stop_thread = threading.Thread(target=stop_async, daemon=True)
            stop_thread.start()
            success = True

        except Exception as e:
            error_msg = f"Failed to stop GenICam service: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return jsonify({"status": "error", "message": error_msg}), 500

        if success:
            result = "Stream stopped successfully"
            status_code = 200

        else:
            result = "Failed to stop stream"
            status_code = 500

        return (
            jsonify(
                {
                    "status": "success" if success else "error",
                    "message": result,
                    "service_status": genicam_service.get_status(),
                }
            ),
            status_code,
        )

    except Exception as e:
        error_msg = f"Error stopping stream: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"status": "error", "message": error_msg}), 500


@inference_bp.route("/set_video", methods=["POST"])
def set_src_video():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        if not data or "video" not in data:
            return jsonify({"error": "Missing 'video' field in request body"}), 400

        video_value = data["video"]

        allowed_videos = ["raw_dough", "baked_baguette"]
        if video_value not in allowed_videos:
            return (
                jsonify(
                    {"error": f"Invalid video value. Must be one of: {allowed_videos}"}
                ),
                400,
            )

        app_config.set_video_config(video_value)

        return (
            jsonify(
                {
                    "message": "Video configuration updated successfully",
                    "video": video_value,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def register_socketio_events(socketio_instance, genicam_service_instance=None):
    """Register SocketIO events with improved error handling"""
    from flask_socketio import join_room, leave_room, emit

    print("Registering SocketIO events...")

    if genicam_service_instance is None:
        genicam_service_instance = get_genicam_service()

    if genicam_service_instance is None:
        print("Warning: Could not get GenICam service for SocketIO events")
        return

    genicam_service_instance.set_socketio(socketio_instance)
    if genicam_service_instance.app_config is None:
        genicam_service_instance.set_app_config(app_config)

    @socketio_instance.on("join_stream", namespace="/ws")
    def on_join_stream():
        try:
            join_room("stream")
            clients = genicam_service_instance.client_joined()
            emit(
                "status",
                {
                    "message": "Joined stream room",
                    "clients": clients,
                    "service_status": genicam_service_instance.get_status(),
                },
            )
            print(f"Client joined stream room. Total clients: {clients}")
        except Exception as e:
            emit(
                "status", {"message": f"Error joining stream: {str(e)}", "error": True}
            )
            print(f"Error in join_stream: {e}")
            traceback.print_exc()

    @socketio_instance.on("connect", namespace="/ws")
    def on_connect():
        try:
            join_room("stream")
            clients = genicam_service_instance.client_joined()
            emit(
                "status",
                {
                    "message": "Connected",
                    "clients": clients,
                    "service_status": genicam_service_instance.get_status(),
                },
            )
            print(f"Client connected. Total clients: {clients}")
        except Exception as e:
            emit("status", {"message": f"Connection error: {str(e)}", "error": True})
            print(f"Error in connect: {e}")
            traceback.print_exc()

    @socketio_instance.on("disconnect", namespace="/ws")
    def on_disconnect():
        try:
            leave_room("stream")
            clients = genicam_service_instance.client_left()
            print(f"Client disconnected. Total clients: {clients}")
        except Exception as e:
            print(f"Error in disconnect: {e}")
            traceback.print_exc()

    @socketio_instance.on("set_confidence", namespace="/ws")
    def on_set_confidence(data):
        try:
            value = float(data.get("value", genicam_service_instance.conf_threshold))
            new_val = genicam_service_instance.set_confidence(value)
            emit(
                "status",
                {
                    "message": "confidence_updated",
                    "value": new_val,
                    "service_status": genicam_service_instance.get_status(),
                },
                to="stream",
            )
            print(f"Confidence threshold updated to: {new_val}")
        except Exception as e:
            emit(
                "status",
                {"message": f"Error setting confidence: {str(e)}", "error": True},
            )
            print(f"Error setting confidence: {e}")
            traceback.print_exc()

    @socketio_instance.on("get_status", namespace="/ws")
    def on_get_status():
        try:
            status = genicam_service_instance.get_status()
            emit("status", {"message": "status_update", "service_status": status})
        except Exception as e:
            emit(
                "status", {"message": f"Error getting status: {str(e)}", "error": True}
            )
            traceback.print_exc()

    @socketio_instance.on("test_frame", namespace="/ws")
    def on_test_frame():
        try:
            test_payload = {
                "frame": "test_base64_string",
                "metrics": {
                    "camera_fps": 30.0,
                    "display_fps": 30.0,
                    "infer_ms": 15.5,
                    "conf_threshold": genicam_service_instance.conf_threshold,
                    "connected_clients": genicam_service_instance.connected_clients,
                },
                "test": True,
            }
            emit("stream_frame", test_payload, to="stream")
            print("Test frame sent")
        except Exception as e:
            emit(
                "status",
                {"message": f"Error sending test frame: {str(e)}", "error": True},
            )
            print(f"Error sending test frame: {e}")
            traceback.print_exc()

    print("SocketIO events registered successfully")
