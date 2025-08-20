from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FastAPI MQTT Application"
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topic: str = "sensor/data"
    mqtt_client_id: str = "fastapi_client"
    
    class Config:
        env_file = ".env"

settings = Settings()