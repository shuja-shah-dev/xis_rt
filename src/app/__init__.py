from flask import Flask, jsonify
from flask_cors import CORS
from config.settings import settings
from app.core.mqtt_client import init_mqtt, mqtt
from app.routes.mqtt_routes import mqtt_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Load MQTT config
    app.config['MQTT_BROKER_URL'] = settings.MQTT_BROKER_URL
    app.config['MQTT_BROKER_PORT'] = settings.MQTT_BROKER_PORT
    app.config['MQTT_USERNAME'] = settings.MQTT_USERNAME
    app.config['MQTT_PASSWORD'] = settings.MQTT_PASSWORD
    app.config['MQTT_CLIENT_ID'] = settings.MQTT_CLIENT_ID
    app.config['MQTT_KEEPALIVE'] = settings.MQTT_KEEPALIVE
    app.config['MQTT_TLS_ENABLED'] = settings.MQTT_TLS_ENABLED
    app.config['MQTT_LOGGING'] = settings.MQTT_LOGGING

    # Init MQTT
    init_mqtt(app)

    # Register blueprints
    app.register_blueprint(mqtt_bp, url_prefix="/mqtt")

    # Root route
    @app.route("/")
    def root():
        return jsonify({
            "message": "Welcome to Flask MQTT Application",
            "endpoints": ["/mqtt/status", "/mqtt/publish_detection"]
        })

    @app.route("/health")
    def health_check():
        return {"status": "healthy"}

    return app
