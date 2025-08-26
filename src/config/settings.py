class Settings:
    MQTT_BROKER_URL = "localhost"
    MQTT_BROKER_PORT = 1883
    MQTT_USERNAME = None
    MQTT_PASSWORD = None
    MQTT_CLIENT_ID = "flask-mqtt-client"
    MQTT_KEEPALIVE = 60
    MQTT_TLS_ENABLED = False
    MQTT_LOGGING = True

settings = Settings()
