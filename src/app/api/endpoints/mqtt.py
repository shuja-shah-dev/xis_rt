from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.mqtt_models import MQTTMessage, SensorData
from app.core.mqtt_client import MQTTClientManager
from datetime import datetime

router = APIRouter(prefix="/mqtt", tags=["MQTT"])

def get_mqtt_client():
    return MQTTClientManager()

@router.get("/status")
async def mqtt_status(mqtt_client: MQTTClientManager = Depends(get_mqtt_client)):
    return {
        "connected": mqtt_client.is_connected(),
        "broker": mqtt_client.broker_host,
        "port": mqtt_client.broker_port
    }

@router.get("/messages", response_model=List[MQTTMessage])
async def get_recent_messages(mqtt_client: MQTTClientManager = Depends(get_mqtt_client)):
    """Get recent MQTT messages"""
    return mqtt_client.get_recent_messages()

@router.post("/publish")
async def publish_message(
    message: MQTTMessage,
    mqtt_client: MQTTClientManager = Depends(get_mqtt_client)
):
    try:
        mqtt_client.publish(message.topic, message.payload, message.qos)
        return {"status": "success", "message": "Message published successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.mqtt_models import MQTTMessage, SensorData
from app.core.mqtt_client import MQTTClientManager
from datetime import datetime

router = APIRouter(prefix="/mqtt", tags=["MQTT"])

def get_mqtt_client():
    return MQTTClientManager()

@router.get("/status")
async def mqtt_status(mqtt_client: MQTTClientManager = Depends(get_mqtt_client)):
    return {
        "connected": mqtt_client.is_connected(),
        "broker": mqtt_client.broker_host,
        "port": mqtt_client.broker_port
    }

@router.get("/messages", response_model=List[MQTTMessage])
async def get_recent_messages(mqtt_client: MQTTClientManager = Depends(get_mqtt_client)):
    return mqtt_client.get_recent_messages()

@router.post("/publish")
async def publish_message(
    message: MQTTMessage,
    mqtt_client: MQTTClientManager = Depends(get_mqtt_client)
):
  
    try:
        mqtt_client.publish(message.topic, message.payload, message.qos)
        return {"status": "success", "message": "Message published successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
