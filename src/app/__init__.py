# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
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

app_config = AppConfig()

__PREFIX__ = "/api"

socketio = SocketIO()

# Global service instance for proper cleanup
_genicam_service = None
_socketio_instance = None  # Store socketio globally

def _cleanup_handler():
    """Global cleanup handler"""
    global _genicam_service
    if _genicam_service:
        print("Performing application-level cleanup...")
        try:
            _genicam_service.cleanup_all()
        except Exception as e:
            print(f"Cleanup error: {e}")

def _signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}, shutting down...")
    _cleanup_handler()
    sys.exit(0)

def get_socketio():
    """Get the global socketio instance"""
    global _socketio_instance
    return _socketio_instance

def create_app():
    global _genicam_service, _socketio_instance
    
    app = Flask(__name__)
    
    # Configure CORS with specific settings
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # MQTT Configuration
    app.config["MQTT_BROKER_URL"] = settings.MQTT_BROKER_URL
    app.config["MQTT_BROKER_PORT"] = settings.MQTT_BROKER_PORT
    app.config["MQTT_USERNAME"] = settings.MQTT_USERNAME
    app.config["MQTT_PASSWORD"] = settings.MQTT_PASSWORD
    app.config["MQTT_CLIENT_ID"] = settings.MQTT_CLIENT_ID
    app.config["MQTT_KEEPALIVE"] = settings.MQTT_KEEPALIVE
    app.config["MQTT_TLS_ENABLED"] = settings.MQTT_TLS_ENABLED
    app.config["MQTT_LOGGING"] = settings.MQTT_LOGGING

    # Initialize MQTT
    try:
        init_mqtt(app)
    except Exception as e:
        print(f"MQTT initialization failed: {e}")

    # Register blueprints
    app.register_blueprint(mqtt_bp, url_prefix=__PREFIX__)
    app.register_blueprint(camera_bp, url_prefix=__PREFIX__)
    app.register_blueprint(model_bp, url_prefix=__PREFIX__)
    
    # Create necessary directories
    directories = ["uploads", "cti", "models", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Failed to create directory {directory}: {e}")

    # Initialize SocketIO with improved configuration
    socketio.init_app(
        app, 
        cors_allowed_origins="*",
        async_mode='threading',  # Use threading for better performance
        ping_timeout=60,
        ping_interval=25,
        logger=False,  # Disable verbose logging
        engineio_logger=False
    )
    
    # Store socketio globally so services can access it
    _socketio_instance = socketio
    print("SocketIO initialized and stored globally")

    # Initialize GenICamService with proper cleanup registration
    try:
        from app.core.genicam_service import GenICamService
        _genicam_service = GenICamService()
        _genicam_service.set_socketio(socketio)  # Set socketio immediately
        _genicam_service.set_app_config(app_config)
        
        print("GenICamService initialized successfully with SocketIO")
    except Exception as e:
        print(f"GenICamService initialization failed: {e}")
        _genicam_service = None

    # Register inference routes with the service
    from app.routes.inference_routes import inference_bp, register_socketio_events
    app.register_blueprint(inference_bp, url_prefix=__PREFIX__)
    
    # Register SocketIO events with the service instance
    try:
        register_socketio_events(socketio, _genicam_service)
        print("SocketIO events registered successfully")
    except Exception as e:
        print(f"SocketIO event registration failed: {e}")

    # Health check endpoint with detailed status
    @app.route("/health")
    def health_check():
        health_status = {
            "status": "healthy",
            "services": {
                "flask": True,
                "socketio": True,
                "mqtt": mqtt is not None,
                "genicam": _genicam_service is not None
            }
        }
        
        # Add service-specific health info
        if _genicam_service:
            try:
                service_status = _genicam_service.get_status()
                health_status["genicam_status"] = service_status
            except Exception as e:
                health_status["services"]["genicam"] = False
                health_status["genicam_error"] = str(e)
        
        # Determine overall health
        if not all(health_status["services"].values()):
            health_status["status"] = "degraded"
            
        return health_status

    # System info endpoint
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
                "platform": sys.platform
            }
            
            try:
                info["open_files"] = len(process.open_files())
                info["connections"] = len(process.connections())
            except (psutil.AccessDenied, AttributeError):
                pass
                
            return {"status": "success", "info": info}
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    # Manual cleanup endpoint
    @app.route("/system/cleanup", methods=["POST"])
    def manual_cleanup():
        try:
            if _genicam_service:
                _genicam_service.cleanup_all()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return {"status": "success", "message": "System cleanup completed"}
        except Exception as e:
            return {"status": "error", "message": f"Cleanup failed: {str(e)}"}, 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"status": "error", "message": "Endpoint not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"status": "error", "message": "Internal server error"}, 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        # Log the error
        app.logger.error(f"Unhandled exception: {str(e)}")
        
        # Return JSON instead of HTML for consistency
        return {"status": "error", "message": "An unexpected error occurred"}, 500

    # Register cleanup handlers
    atexit.register(_cleanup_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Context processors for templates (if using)
    @app.context_processor
    def inject_config():
        return {
            'app_config': app_config,
            'genicam_status': _genicam_service.get_status() if _genicam_service else None
        }

    # Before request hooks
    @app.before_request
    def before_request():
        # Add any pre-request processing here
        pass

    # After request hooks
    @app.after_request
    def after_request(response):
        # Add CORS headers if not already present
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    print("Flask application created successfully")
    print(f"GenICam service: {'Initialized' if _genicam_service else 'Failed'}")
    print(f"Available routes: {[rule.rule for rule in app.url_map.iter_rules()]}")
    
    return app

def get_genicam_service():
    """Get the global GenICam service instance"""
    global _genicam_service
    return _genicam_service

# Export the service getter for use in other modules
__all__ = ['create_app', 'socketio', 'get_genicam_service', 'get_socketio']