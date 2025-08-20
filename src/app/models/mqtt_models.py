from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class MQTTMessage(BaseModel):
    topic: str
    payload: str
    timestamp: Optional[datetime] = None
    qos: int = 0

class SensorData(BaseModel):
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: Optional[datetime] = None