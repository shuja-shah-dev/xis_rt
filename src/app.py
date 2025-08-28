# app.py
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from app import create_app, socketio, get_genicam_service

# Create the Flask application
app = create_app()

def setup_logging():
    """Setup logging for the application"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup file logging
    file_handler = RotatingFileHandler('logs/camera_server.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    # Setup console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)
    
    # Log startup info
    app.logger.info('Camera server startup')

def validate_environment():
    """Validate environment and dependencies"""
    required_dirs = ['models', 'uploads', 'cti', 'logs']
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")
    
    # Check if models directory has any engine files
    models_dir = 'models'
    if os.path.exists(models_dir):
        engine_files = [f for f in os.listdir(models_dir) if f.endswith('.engine')]
        if engine_files:
            print(f"Found {len(engine_files)} TensorRT engine files in models directory")
        else:
            print("Warning: No TensorRT engine files found in models directory")
    
    # Check for CUDA availability
    try:
        import pycuda.driver as cuda
        cuda.init()
        device_count = cuda.Device.count()
        print(f"CUDA initialized successfully. Found {device_count} GPU(s)")
        
        if device_count > 0:
            device = cuda.Device(0)
            print(f"GPU 0: {device.name()}")
            
            # Check memory
            context = device.make_context()
            try:
                free, total = cuda.mem_get_info()
                print(f"GPU Memory: {free // 1024 // 1024}MB free / {total // 1024 // 1024}MB total")
            finally:
                context.pop()
                context.detach()
        
    except Exception as e:
        print(f"Warning: CUDA not available or not working properly: {e}")

def main():
    """Main application entry point"""
    try:
        # Setup logging
        setup_logging()
        
        # Validate environment
        validate_environment()
        
        # Get environment variables
        host = os.environ.get('FLASK_HOST', '0.0.0.0')
        port = int(os.environ.get('FLASK_PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        
        print(f"Starting camera server on {host}:{port}")
        print(f"Debug mode: {debug}")
        
        if debug:
            print("Warning: Running in debug mode. This should not be used in production!")
        
        # Check if GenICam service is available
        genicam_service = get_genicam_service()
        if genicam_service:
            print("GenICam service is ready")
            service_status = genicam_service.get_status()
            print(f"Service status: {service_status}")
        else:
            print("Warning: GenICam service not available")
        
        # Start the server
        socketio.run(
            app, 
            host=host, 
            port=port, 
            debug=debug,
            use_reloader=False,  # Disable reloader to prevent double initialization
            log_output=True
        )
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        app.logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Cleanup on exit
        print("Performing final cleanup...")
        try:
            genicam_service = get_genicam_service()
            if genicam_service:
                genicam_service.cleanup_all()
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Application shutdown complete")

if __name__ == '__main__':
    main()