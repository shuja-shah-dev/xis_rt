class Settings:
    def __init__(self):
        self.app_name = "FastAPI MQTT Application"
        self.mqtt_broker_host = "localhost"
        self.mqtt_broker_port = 1883
        self.mqtt_topic = "sensor/data"
        self.mqtt_client_id = "fastapi_client"

settings = Settings()