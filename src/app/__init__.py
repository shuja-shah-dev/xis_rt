from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from config.settings import settings
from app.core.mqtt_client import init_mqtt, mqtt
from app.routes.mqtt_routes import mqtt_bp
from app.routes.camera_routes import camera_bp
from app.routes.model_routes import model_bp



__PREFIX__ = "/api"


socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config["MQTT_BROKER_URL"] = settings.MQTT_BROKER_URL
    app.config["MQTT_BROKER_PORT"] = settings.MQTT_BROKER_PORT
    app.config["MQTT_USERNAME"] = settings.MQTT_USERNAME
    app.config["MQTT_PASSWORD"] = settings.MQTT_PASSWORD
    app.config["MQTT_CLIENT_ID"] = settings.MQTT_CLIENT_ID
    app.config["MQTT_KEEPALIVE"] = settings.MQTT_KEEPALIVE
    app.config["MQTT_TLS_ENABLED"] = settings.MQTT_TLS_ENABLED
    app.config["MQTT_LOGGING"] = settings.MQTT_LOGGING

    init_mqtt(app)
    app.register_blueprint(mqtt_bp, url_prefix=__PREFIX__)
    app.register_blueprint(camera_bp, url_prefix=__PREFIX__)
    app.register_blueprint(model_bp, url_prefix=__PREFIX__)

    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("cti"):
        os.makedirs("cti")

    socketio.init_app(app, cors_allowed_origins="*")

    from app.core.genicam_service import GenICamService

    genicam_service = GenICamService()
    genicam_service.set_socketio(socketio)

    @app.route("/health")
    def health_check():
        return {"status": "healthy"}

    return app
