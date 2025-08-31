from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
from config.settings import settings
from app.core.mqtt_client import init_mqtt, mqtt
from app.routes.mqtt_routes import mqtt_bp
from app.routes.camera_routes import camera_bp
from app.routes.model_routes import model_bp
import os
import atexit
import signal
import sys
from app.core.config import AppConfig
import threading
from app.core.video_service.raw_dough import VideoInferenceService
from app.core.video_service.baked_baguette import VideoInferenceService_baked
from app.core.video_service.donut import VideoInferenceService_dont


app_config = AppConfig()
__PREFIX__ = "/api"
socketio = SocketIO()

_genicam_service = None
_socketio_instance = None
_genicam_thread = None
_genicam_lock = threading.Lock()


def _cleanup_handler():
    global _genicam_service
    if _genicam_service:
        print("Performing application-level cleanup...")
        try:
            _genicam_service.cleanup_all()
        except Exception as e:
            print(f"Cleanup error: {e}")


def _signal_handler(signum, frame):
    print(f"Received signal {signum}, shutting down...")
    _cleanup_handler()
    sys.exit(0)


def get_socketio():
    global _socketio_instance
    return _socketio_instance


def create_app():
    global _genicam_service, _socketio_instance

    app = Flask(__name__)

    CORS(
        app,
        resources={
            r"/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )

    app.config["MQTT_BROKER_URL"] = settings.MQTT_BROKER_URL
    app.config["MQTT_BROKER_PORT"] = settings.MQTT_BROKER_PORT
    app.config["MQTT_USERNAME"] = settings.MQTT_USERNAME
    app.config["MQTT_PASSWORD"] = settings.MQTT_PASSWORD
    app.config["MQTT_CLIENT_ID"] = settings.MQTT_CLIENT_ID
    app.config["MQTT_KEEPALIVE"] = settings.MQTT_KEEPALIVE
    app.config["MQTT_TLS_ENABLED"] = settings.MQTT_TLS_ENABLED
    app.config["MQTT_LOGGING"] = settings.MQTT_LOGGING

    try:
        init_mqtt(app)
    except Exception as e:
        print(f"MQTT initialization failed: {e}")

    app.register_blueprint(mqtt_bp, url_prefix=__PREFIX__)
    app.register_blueprint(camera_bp, url_prefix=__PREFIX__)
    app.register_blueprint(model_bp, url_prefix=__PREFIX__)

    directories = ["uploads", "cti", "models", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Failed to create directory {directory}: {e}")

    socketio.init_app(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        ping_timeout=60,
        ping_interval=25,
        logger=True,  # CHANGED: Enable logging for debugging
        engineio_logger=True,  # CHANGED: Enable engineio logging
    )

    _socketio_instance = socketio
    print("SocketIO initialized and stored globally")

    # ============ SOCKETIO EVENT HANDLERS MUST BE DEFINED AFTER INIT ============
    @socketio.on("connect", namespace="/ws")
    def handle_connect():
        emit("status", {"message": "Connected to WebSocket", "connected": True})

    @socketio.on("disconnect", namespace="/ws")
    def handle_disconnect():
        print("❌ Client disconnected from /ws namespace")

    @socketio.on("join_stream", namespace="/ws")
    def handle_join_stream():
        join_room("stream")
        print("👥 Client joined stream room")
        emit("status", {"message": "Joined stream room", "room": "stream"})

    @socketio.on("leave_stream", namespace="/ws")
    def handle_leave_stream():
        print("👋 Client left stream room")
        emit("status", {"message": "Left stream room"})

    # ============================================================================

    try:
        from app.core.geni_inference import TensorRTGenICamDetector

        _genicam_service = TensorRTGenICamDetector(websocket_mode=True, lazy_init=True)
        _genicam_service.set_socketio(socketio)
        _genicam_service.set_app_config(app_config)
        print("GenICam service partially initialized - waiting for configuration")
    except Exception as e:
        print(f"GenICam service partial initialization failed: {e}")
        _genicam_service = None
        print("GenICamService initialized successfully with SocketIO")
    except Exception as e:
        print(f"GenICamService initialization failed: {e}")
        _genicam_service = None

    from app.routes.inference_routes import inference_bp, register_socketio_events

    app.register_blueprint(inference_bp, url_prefix=__PREFIX__)

    try:
        register_socketio_events(socketio, _genicam_service)
        print("SocketIO events registered successfully")
    except Exception as e:
        print(f"SocketIO event registration failed: {e}")

    @app.route("/health")
    def health_check():
        health_status = {
            "status": "healthy",
            "services": {
                "flask": True,
                "socketio": True,
                "mqtt": mqtt is not None,
                "genicam": _genicam_service is not None,
            },
        }

        if _genicam_service:
            try:
                service_status = _genicam_service.get_status()
                health_status["genicam_status"] = service_status
            except Exception as e:
                health_status["services"]["genicam"] = False
                health_status["genicam_error"] = str(e)

        if not all(health_status["services"].values()):
            health_status["status"] = "degraded"

        return health_status

    @app.route("/system/info")
    def system_info():
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            info = {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "python_version": sys.version,
                "platform": sys.platform,
            }

            try:
                info["open_files"] = len(process.open_files())
                info["connections"] = len(process.connections())
            except (psutil.AccessDenied, AttributeError):
                pass

            return {"status": "success", "info": info}
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    @app.route("/system/cleanup", methods=["POST"])
    def manual_cleanup():
        try:
            if _genicam_service:
                _genicam_service.cleanup_all()

            import gc

            gc.collect()

            return {"status": "success", "message": "System cleanup completed"}
        except Exception as e:
            return {"status": "error", "message": f"Cleanup failed: {str(e)}"}, 500

    @app.errorhandler(404)
    def not_found(error):
        return {"status": "error", "message": "Endpoint not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"status": "error", "message": "Internal server error"}, 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled exception: {str(e)}")
        return {"status": "error", "message": "An unexpected error occurred"}, 500

    atexit.register(_cleanup_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    @app.context_processor
    def inject_config():
        return {
            "app_config": app_config,
            "genicam_status": (
                _genicam_service.get_status() if _genicam_service else None
            ),
        }

    @app.before_request
    def before_request():
        pass

    @app.after_request
    def after_request(response):
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type,Authorization"
        )
        response.headers.add(
            "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
        )
        return response

    print("Flask application created successfully")
    print(f"GenICam service: {'Initialized' if _genicam_service else 'Failed'}")
    init_video_service_rd(app=app, socketio=socketio)
    init_video_service_bk(app=app, socketio=socketio)
    init_video_service_dnt(app=app, socketio=socketio)

    return app


video_service = None
bked_service = None


def init_video_service_rd(app, socketio):
    global video_service
    video_service = VideoInferenceService(websocket_mode=True)
    video_service.set_socketio(socketio)
    video_service.set_mqtt(mqtt)

    @socketio.on("start_video_processing", namespace="/ws")
    def handle_start_video(data):
        config = {
            "engine_path": data.get("engine_path", app.config.get("VIDEO_ENGINE_PATH")),
            "video_path": data.get("video_path", app.config.get("VIDEO_INPUT_PATH")),
            "score_threshold": data.get("score_threshold", 0.4),
            "score_class0": data.get("score_class0"),
            "score_class1": data.get("score_class1"),
            "nms_threshold": data.get("nms_threshold", 0.5),
            "mask_threshold": data.get("mask_threshold", 0.4),
            "canvas_size": data.get("canvas_size", 640),
            "alpha": data.get("alpha", 0.3),
            "min_inference_frames": data.get("min_inference_frames", 30),
            "target_fps": data.get("target_fps", 30.0),
            "mqtt_topic": data.get("mqtt_topic", "detection/results"),
        }

        if video_service.initialize_from_config(config):
            if video_service.start_processing():
                socketio.emit("video_status", {"status": "started"}, namespace="/ws")
            else:
                socketio.emit(
                    "video_status",
                    {"status": "failed", "error": "Could not start processing"},
                    namespace="/ws",
                )
        else:
            socketio.emit(
                "video_status",
                {"status": "failed", "error": "Initialization failed"},
                namespace="/ws",
            )

    @socketio.on("stop_video_processing", namespace="/ws")
    def handle_stop_video():
        if video_service.stop_processing():
            socketio.emit("video_status", {"status": "stopped"}, namespace="/ws")
        else:
            socketio.emit("video_status", {"status": "error"}, namespace="/ws")

    @socketio.on("get_video_status", namespace="/ws")
    def handle_get_status():
        status = video_service.get_status() if video_service else {"is_running": False}
        socketio.emit("video_status", status, namespace="/ws")

    return video_service


def init_video_service_bk(app, socketio):
    global bked_service
    bked_service = VideoInferenceService_baked(websocket_mode=True)
    bked_service.set_socketio(socketio)
    bked_service.set_mqtt(mqtt)

    @socketio.on("start_video_processing", namespace="/ws")
    def handle_start_video(data):
        config = {
            "engine_path": data.get("engine_path", app.config.get("VIDEO_ENGINE_PATH")),
            "video_path": data.get("video_path", app.config.get("VIDEO_INPUT_PATH")),
            "score_threshold": data.get("score_threshold", 0.4),
            "score_class0": data.get("score_class0"),
            "score_class1": data.get("score_class1"),
            "nms_threshold": data.get("nms_threshold", 0.5),
            "mask_threshold": data.get("mask_threshold", 0.4),
            "canvas_size": data.get("canvas_size", 640),
            "alpha": data.get("alpha", 0.3),
            "min_inference_frames": data.get("min_inference_frames", 30),
            "target_fps": data.get("target_fps", 30.0),
            "mqtt_topic": data.get("mqtt_topic", "detection/results"),
        }

        if bked_service.initialize_from_config(config):
            if bked_service.start_processing():
                socketio.emit("video_status", {"status": "started"}, namespace="/ws")
            else:
                socketio.emit(
                    "video_status",
                    {"status": "failed", "error": "Could not start processing"},
                    namespace="/ws",
                )
        else:
            socketio.emit(
                "video_status",
                {"status": "failed", "error": "Initialization failed"},
                namespace="/ws",
            )

    @socketio.on("stop_video_processing", namespace="/ws")
    def handle_stop_video():
        if bked_service.stop_processing():
            socketio.emit("video_status", {"status": "stopped"}, namespace="/ws")
        else:
            socketio.emit("video_status", {"status": "error"}, namespace="/ws")

    @socketio.on("get_video_status", namespace="/ws")
    def handle_get_status():
        status = bked_service.get_status() if bked_service else {"is_running": False}
        socketio.emit("video_status", status, namespace="/ws")

    return bked_service


def init_video_service_dnt(app, socketio):
    global dont_service
    dont_service = VideoInferenceService_dont(websocket_mode=True)
    dont_service.set_socketio(socketio)
    dont_service.set_mqtt(mqtt)

    @socketio.on("start_video_processing", namespace="/ws")
    def handle_start_video(data):
        config = {
            "engine_path": data.get("engine_path", app.config.get("VIDEO_ENGINE_PATH")),
            "video_path": data.get("video_path", app.config.get("VIDEO_INPUT_PATH")),
            "score_threshold": data.get("score_threshold", 0.4),
            "score_class0": data.get("score_class0"),
            "score_class1": data.get("score_class1"),
            "nms_threshold": data.get("nms_threshold", 0.5),
            "mask_threshold": data.get("mask_threshold", 0.4),
            "canvas_size": data.get("canvas_size", 640),
            "alpha": data.get("alpha", 0.3),
            "min_inference_frames": data.get("min_inference_frames", 30),
            "target_fps": data.get("target_fps", 30.0),
            "mqtt_topic": data.get("mqtt_topic", "detection/results"),
        }

        if dont_service.initialize_from_config(config):
            if dont_service.start_processing():
                socketio.emit("video_status", {"status": "started"}, namespace="/ws")
            else:
                socketio.emit(
                    "video_status",
                    {"status": "failed", "error": "Could not start processing"},
                    namespace="/ws",
                )
        else:
            socketio.emit(
                "video_status",
                {"status": "failed", "error": "Initialization failed"},
                namespace="/ws",
            )

    @socketio.on("stop_video_processing", namespace="/ws")
    def handle_stop_video():
        if dont_service.stop_processing():
            socketio.emit("video_status", {"status": "stopped"}, namespace="/ws")
        else:
            socketio.emit("video_status", {"status": "error"}, namespace="/ws")

    @socketio.on("get_video_status", namespace="/ws")
    def handle_get_status():
        status = dont_service.get_status() if dont_service else {"is_running": False}
        socketio.emit("video_status", status, namespace="/ws")

    return dont_service


def start_genicam_service_async(model_path, cti_file_path):
    global _genicam_thread, _genicam_lock

    with _genicam_lock:
        if _genicam_thread and _genicam_thread.is_alive():
            print("GenICam service already running")
            return {"status": "already_running", "message": "Service is already active"}

        def service_worker():
            try:
                if _genicam_service:
                    success = _genicam_service.complete_initialization(
                        model_path, cti_file_path
                    )
                    if success:
                        _genicam_service.run_service_loop()
                    else:
                        print("Failed to complete GenICam initialization")
            except Exception as e:
                print(f"GenICam service thread error: {e}")
                import traceback

                traceback.print_exc()

        _genicam_thread = threading.Thread(
            target=service_worker, daemon=True, name="GenICamService"
        )
        _genicam_thread.start()

        return {"status": "started", "message": "Service started in background"}


def stop_genicam_service():
    global _genicam_thread, _genicam_lock

    service_ref = None
    thread_ref = None

    with _genicam_lock:
        service_ref = _genicam_service
        thread_ref = _genicam_thread

    if service_ref:
        service_ref.stop_streaming_completely()
    if thread_ref and thread_ref.is_alive():
        print("Service thread will finish cleanup in background")

    with _genicam_lock:
        _genicam_thread = None

    return {"status": "stopped", "message": "Service stop initiated"}


def get_genicam_service():
    global _genicam_service
    return _genicam_service


__all__ = ["create_app", "socketio", "get_genicam_service", "get_socketio"]
