import paho.mqtt.client as mqtt
import json
import threading
import time
import socket
from typing import List, Callable
from datetime import datetime

class MQTTClientManager:
    def __init__(self):
        self.broker_host = "localhost"
        self.broker_port = 1883
        self.client_id = f"mqtt_client_{int(time.time())}"
        self.topic = "sensor/data"
        
        self.client = None
        self.connected = False
        self.connection_lost = False
        self.messages = []
        self.max_messages = 100
        
        self._lock = threading.Lock()
        self._should_reconnect = True
        self._reconnect_delay = 5
        self._max_reconnect_delay = 60
        
        self._initialize_client()
        self._connect()

    def _initialize_client(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except:
                pass
        
        self.client = mqtt.Client(
            client_id=self.client_id, 
            clean_session=True,
            protocol=mqtt.MQTTv311
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        self.client.keepalive = 60
        self.client.reconnect_delay_set(min_delay=5, max_delay=60)

    def _connect(self):
        try:
            print(f"Attempting to connect to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.client.connect_async(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Connection error: {e}")
            self._schedule_reconnect()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Successfully connected to MQTT broker")
            self.connected = True
            self.connection_lost = False
            self._reconnect_delay = 5
            
            # Subscribe to topics
            client.subscribe(self.topic)
            client.subscribe("sensor/+")
        else:
            print(f"Connection failed with code {rc}")
            self.connected = False
            if self._should_reconnect:
                self._schedule_reconnect()

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        print(f"Successfully subscribed (mid: {mid}, QoS: {granted_qos})")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            print(f"Received message on {msg.topic}: {payload}")
            
            # Store message
            with self._lock:
                self.messages.append({
                    'topic': msg.topic,
                    'payload': payload,
                    'timestamp': datetime.now(),
                    'qos': msg.qos
                })
                if len(self.messages) > self.max_messages:
                    self.messages = self.messages[-self.max_messages:]
                    
        except Exception as e:
            print(f"Error processing message: {e}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"Disconnected with result code {rc}")
        self.connected = False
        self.connection_lost = True
        
        if rc != 0 and self._should_reconnect:
            print(f"Unexpected disconnect, scheduling reconnect...")
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        def reconnect():
            print(f"Waiting {self._reconnect_delay} seconds before reconnect...")
            time.sleep(self._reconnect_delay)
            if self._should_reconnect:
                self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
                try:
                    self._initialize_client()
                    self._connect()
                except Exception as e:
                    print(f"Reconnect failed: {e}")
                    self._schedule_reconnect()
        
        threading.Thread(target=reconnect, daemon=True).start()

    # API METHODS - These are required by your endpoints
    def get_recent_messages(self) -> List[dict]:
        """Get recent messages (required by API endpoint)"""
        with self._lock:
            return self.messages.copy()

    def is_connected(self) -> bool:
        """Check if connected (required by API endpoint)"""
        return self.connected

    def publish(self, topic: str, payload: str, qos: int = 0):
        """Publish a message"""
        if not self.connected:
            raise Exception("Not connected to MQTT broker")
        
        try:
            result = self.client.publish(topic, payload, qos)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"Failed to publish message: {result.rc}")
            return result
        except Exception as e:
            print(f"Error publishing message: {e}")
            raise

    def subscribe(self, topic: str, qos: int = 0):
        """Subscribe to a topic"""
        if not self.connected:
            raise Exception("Not connected to MQTT broker")
        
        try:
            result = self.client.subscribe(topic, qos)
            if result[0] != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"Failed to subscribe to topic: {result[0]}")
            return result
        except Exception as e:
            print(f"Error subscribing to topic: {e}")
            raise

    def stop_reconnection(self):
        """Stop automatic reconnection"""
        self._should_reconnect = False

    def start_reconnection(self):
        """Start automatic reconnection"""
        self._should_reconnect = True
        if not self.connected:
            self._connect()

    def disconnect(self):
        """Disconnect from broker"""
        self._should_reconnect = False
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                print("Disconnected from MQTT broker")
            except:
                pass

# Global instance
mqtt_client_manager = MQTTClientManager()