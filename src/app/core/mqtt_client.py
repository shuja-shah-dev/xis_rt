import paho.mqtt.client as mqtt
import json
import threading
import time
from typing import List, Callable
from datetime import datetime
from config.settings import settings
from app.models.mqtt_models import MQTTMessage

class MQTTClientManager:
    def __init__(self):
        self.broker_host = settings.mqtt_broker_host
        self.broker_port = settings.mqtt_broker_port
        self.client_id = settings.mqtt_client_id
        self.topic = settings.mqtt_topic
        
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self.connected = False
        self.messages = []
        self.max_messages = 100  
        
        self._connect_async()
    
    def _connect_async(self):
        def connect():
            try:
                self.client.connect(self.broker_host, self.broker_port, 60)
                self.client.loop_start()
            except Exception as e:
                print(f"Failed to connect to MQTT broker: {e}")
        
        thread = threading.Thread(target=connect, daemon=True)
        thread.start()
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
            self.connected = True
         
            client.subscribe(self.topic)
            client.subscribe("sensor/+")  
        else:
            print(f"Failed to connect to MQTT broker with code {rc}")
            self.connected = False
    
    def _on_message(self, client, userdata, msg):
    
        try:
            payload = msg.payload.decode('utf-8')
            message = MQTTMessage(
                topic=msg.topic,
                payload=payload,
                timestamp=datetime.now(),
                qos=msg.qos
            )
            

            self.messages.append(message)
            
 
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
            
            print(f"Received message on topic {msg.topic}: {payload}")
        
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        print("Disconnected from MQTT broker")
        self.connected = False
    
    def publish(self, topic: str, payload: str, qos: int = 0):
        if not self.connected:
            raise Exception("Not connected to MQTT broker")
        
        result = self.client.publish(topic, payload, qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise Exception(f"Failed to publish message: {result.rc}")
    
    def subscribe(self, topic: str, qos: int = 0):
        if not self.connected:
            raise Exception("Not connected to MQTT broker")
        
        result = self.client.subscribe(topic, qos)
        if result[0] != mqtt.MQTT_ERR_SUCCESS:
            raise Exception(f"Failed to subscribe to topic: {result[0]}")
    
    def is_connected(self) -> bool:
        return self.connected
    
    def get_recent_messages(self) -> List[MQTTMessage]:
        return self.messages.copy()
    
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()


mqtt_client_manager = MQTTClientManager()
