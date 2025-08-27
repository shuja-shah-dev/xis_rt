from flask_mqtt import Mqtt
from datetime import datetime
import json

mqtt = Mqtt()
messages = []
MAX_MESSAGES = 100
connected = False

def init_mqtt(app):
    mqtt.init_app(app)

    @mqtt.on_connect()
    def handle_connect(client, userdata, flags, rc):
        global connected
        if rc == 0:
            connected = True
            print("Connected to MQTT broker")
            mqtt.subscribe("test/topic")
            mqtt.subscribe("detection/results")
        else:
            print(f"Failed to connect. Code={rc}")

    @mqtt.on_message()
    def handle_message(client, userdata, msg):
        payload = msg.payload.decode()
        mqtt_message = {
            "topic": msg.topic,
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
            "qos": msg.qos
        }
        messages.append(mqtt_message)
        if len(messages) > MAX_MESSAGES:
            messages.pop(0)
        print(f"📩 {msg.topic} -> {payload}")

    @mqtt.on_disconnect()
    def handle_disconnect():
        global connected
        connected = False
        print("🔌 Disconnected from MQTT broker")

def publish(topic, payload, qos=0):
    mqtt.publish(topic, payload, qos)

def get_messages():
    return messages

def is_connected():
    return connected
