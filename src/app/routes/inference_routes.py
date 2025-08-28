# app/routes/inference_routes.py
from flask import Blueprint, jsonify, request
from app.core.config import AppConfig
import threading
import time
import os
import traceback

inference_bp = Blueprint("inference", __name__)
app_config = AppConfig()

def get_genicam_service():
    """Get the GenICam service from app initialization"""
    from app import get_genicam_service as get_app_service
    return get_app_service()

# def get_genicam_service():
#     """Get the GenICam service instance from the app"""
#     try:
#         from app import get_genicam_service as get_app_service
#         service = get_app_service()
        
#         if service is None:
#             print("No GenICamService available from app")
#             return None
            
#         # Ensure it has the necessary configurations
#         if service.app_config is None:
#             service.set_app_config(app_config)
#             print("Set app config on app service")
            
#         if service.socketio is None:
#             from app import get_socketio
#             socketio_instance = get_socketio()
#             if socketio_instance:
#                 service.set_socketio(socketio_instance)
#                 print("Set socketio on app service")
        
#         print(f"Using app GenICamService - socketio available: {service.socketio is not None}")
#         return service
        
#     except Exception as e:
#         print(f"Failed to get app service: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
def run_video_inference(config):
    """Run video inference - placeholder for video processing"""
    return f"Running video inference with model {config['model_selection']} on video {config['video_path']}"

@inference_bp.route("/stream/normal", methods=["POST"])
def start_normal_stream():
    """Start normal camera stream"""
    try:
        print("Starting normal stream request...")
        
        # Validate configuration
        is_valid, message = app_config.validate_config()
        if not is_valid:
            print(f"Configuration validation failed: {message}")
            return jsonify({"status": "error", "message": message}), 400

        config = app_config.get_config()
        print(f"Config: {config}")
        
        # Get service instance (this will set the app_config)
        genicam_service = get_genicam_service()
        if genicam_service is None:
            return jsonify({
                "status": "error", 
                "message": "Failed to initialize GenICam service"
            }), 500

        # Double-check that config is set
        if genicam_service.app_config is None:
            print("Warning: app_config still None, setting it now...")
            genicam_service.set_app_config(app_config)

        if config["input_type"] == "camera":
            print("Attempting to start normal camera stream...")
            success = genicam_service.run_normal()
            
            if success:
                result = "Normal camera stream started successfully"
                status_code = 200
                print(result)
            else:
                result = "Failed to start normal camera stream"
                status_code = 500
                print(result)
        else:
            result = "Normal stream only available for camera input"
            status_code = 400
            print(result)

        return jsonify({
            "status": "success" if status_code == 200 else "error", 
            "message": result,
            "service_status": genicam_service.get_status()
        }), status_code

    except Exception as e:
        error_msg = f"Failed to start normal stream: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": error_msg
        }), 500

@inference_bp.route("/stream/inference", methods=["POST"])
def start_inference_stream():
    """Start inference camera stream"""
    try:
        print("Starting inference stream request...")
        
        # Validate configuration
        is_valid, message = app_config.validate_config()
        if not is_valid:
            print(f"Configuration validation failed: {message}")
            return jsonify({"status": "error", "message": message}), 400

        config = app_config.get_config()
        print(f"Config: {config}")
        
        # Get service instance (this will set the app_config)
        genicam_service = get_genicam_service()
        if genicam_service is None:
            return jsonify({
                "status": "error", 
                "message": "Failed to initialize GenICam service"
            }), 500

        # Double-check that config is set
        if genicam_service.app_config is None:
            print("Warning: app_config still None, setting it now...")
            genicam_service.set_app_config(app_config)

        if config["input_type"] == "camera":
            if config["model_selection"]:
                engine_path = f"models/{config['model_selection']}.engine"
                print(f"Looking for engine at: {engine_path}")
                
                # Verify engine file exists
                if not os.path.exists(engine_path):
                    error_msg = f"Model engine not found: {engine_path}"
                    print(error_msg)
                    return jsonify({
                        "status": "error", 
                        "message": error_msg
                    }), 404
                
                print("Attempting to start inference stream...")
                success = genicam_service.run_with_inference(engine_path)
                
                if success:
                    result = f"Inference stream started with model {config['model_selection']}"
                    status_code = 200
                    print(result)
                else:
                    result = f"Failed to start inference stream with model {config['model_selection']}"
                    status_code = 500
                    print(result)
            else:
                result = "No model selected for inference"
                status_code = 400
                print(result)
        else:
            # Video inference placeholder
            result = run_video_inference(config)
            status_code = 200
            print(result)

        return jsonify({
            "status": "success" if status_code == 200 else "error",
            "message": result,
            "service_status": genicam_service.get_status() if config["input_type"] == "camera" else None
        }), status_code

    except Exception as e:
        error_msg = f"Failed to start inference stream: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": error_msg
        }), 500

@inference_bp.route("/stream/stop", methods=["POST"])
def stop_stream():
    """Stop camera streaming"""
    try:
        print("Stopping stream request...")
        
        genicam_service = get_genicam_service()
        if genicam_service is None:
            return jsonify({
                "status": "error", 
                "message": "GenICam service not available"
            }), 500
            
        success = genicam_service.stop()
        
        if success:
            result = "Stream stopped successfully"
            status_code = 200
            print(result)
        else:
            result = "Failed to stop stream"
            status_code = 500
            print(result)
            
        return jsonify({
            "status": "success" if success else "error",
            "message": result,
            "service_status": genicam_service.get_status()
        }), status_code

    except Exception as e:
        error_msg = f"Error stopping stream: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": error_msg
        }), 500

@inference_bp.route("/stream/status", methods=["GET"])
def get_stream_status():
    """Get current streaming status and metrics"""
    try:
        genicam_service = get_genicam_service()
        if genicam_service is None:
            return jsonify({
                "status": "error",
                "message": "GenICam service not available"
            }), 500
            
        status = genicam_service.get_status()
        
        return jsonify({
            "status": "success",
            "data": status
        }), 200

    except Exception as e:
        error_msg = f"Error getting status: {str(e)}"
        print(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@inference_bp.route("/stream/test", methods=["POST"])
def test_service():
    """Test service initialization"""
    try:
        print("Testing service initialization...")
        
        # Test config first
        is_valid, message = app_config.validate_config()
        config = app_config.get_config()
        
        print(f"Config validation: {is_valid}, message: {message}")
        print(f"Config data: {config}")
        
        genicam_service = get_genicam_service()
        if genicam_service is None:
            return jsonify({
                "status": "error",
                "message": "Failed to create GenICam service",
                "test_result": "FAILED"
            }), 500
        
        # Check if app config is properly set
        service_has_config = genicam_service.app_config is not None
        print(f"Service has app_config: {service_has_config}")
        
        status = genicam_service.get_status()
        
        return jsonify({
            "status": "success",
            "message": "Service initialized successfully",
            "service_status": status,
            "config_validation": {"valid": is_valid, "message": message},
            "config_data": config,
            "service_has_config": service_has_config,
            "test_result": "PASSED"
        }), 200
        
    except Exception as e:
        error_msg = f"Service test failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": error_msg,
            "test_result": "FAILED"
        }), 500

def register_socketio_events(socketio_instance, genicam_service_instance=None):
    """Register SocketIO events with improved error handling"""
    from flask_socketio import join_room, leave_room, emit
    
    print("Registering SocketIO events...")
    
    # Use the singleton service if none provided
    if genicam_service_instance is None:
        genicam_service_instance = get_genicam_service()
    
    if genicam_service_instance is None:
        print("Warning: Could not get GenICam service for SocketIO events")
        return
    
    # Set SocketIO instance on the service
    genicam_service_instance.set_socketio(socketio_instance)
    # Ensure app_config is set
    if genicam_service_instance.app_config is None:
        genicam_service_instance.set_app_config(app_config)
    
    @socketio_instance.on("join_stream", namespace="/ws")
    def on_join_stream():
        try:
            join_room("stream")
            clients = genicam_service_instance.client_joined()
            emit("status", {
                "message": "Joined stream room", 
                "clients": clients,
                "service_status": genicam_service_instance.get_status()
            })
            print(f"Client joined stream room. Total clients: {clients}")
        except Exception as e:
            emit("status", {"message": f"Error joining stream: {str(e)}", "error": True})
            print(f"Error in join_stream: {e}")
            traceback.print_exc()
    
    @socketio_instance.on("connect", namespace="/ws")
    def on_connect():
        try:
            join_room("stream")
            clients = genicam_service_instance.client_joined()
            emit("status", {
                "message": "Connected", 
                "clients": clients,
                "service_status": genicam_service_instance.get_status()
            })
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
            emit("status", {
                "message": "confidence_updated", 
                "value": new_val,
                "service_status": genicam_service_instance.get_status()
            }, to="stream")
            print(f"Confidence threshold updated to: {new_val}")
        except Exception as e:
            emit("status", {"message": f"Error setting confidence: {str(e)}", "error": True})
            print(f"Error setting confidence: {e}")
            traceback.print_exc()
    
    @socketio_instance.on("get_status", namespace="/ws")
    def on_get_status():
        try:
            status = genicam_service_instance.get_status()
            emit("status", {
                "message": "status_update",
                "service_status": status
            })
        except Exception as e:
            emit("status", {"message": f"Error getting status: {str(e)}", "error": True})
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
                    "connected_clients": genicam_service_instance.connected_clients
                },
                "test": True
            }
            emit("stream_frame", test_payload, to="stream")
            print("Test frame sent")
        except Exception as e:
            emit("status", {"message": f"Error sending test frame: {str(e)}", "error": True})
            print(f"Error sending test frame: {e}")
            traceback.print_exc()
    
    print("SocketIO events registered successfully")