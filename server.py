import paho.mqtt.client as mqtt
import json
import time
import random
import threading
from datetime import datetime
from typing import Dict, Any
import argparse


class MQTTPublisher:
    def __init__(
        self, broker_host="localhost", broker_port=1883, client_id="mqtt_publisher"
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id

        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self.connected = False
        self.running = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(
                f"✅ Connected to MQTT broker at {self.broker_host}:{self.broker_port}"
            )
            self.connected = True
        else:
            print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc):
        print("🔌 Disconnected from MQTT broker")
        self.connected = False

    def _on_publish(self, client, userdata, mid):
        print(f"📤 Message {mid} published successfully")

    def connect(self):
        try:
            print(
                f"🔄 Connecting to MQTT broker at {self.broker_host}:{self.broker_port}..."
            )
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()

            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(1)
                timeout -= 1

            if not self.connected:
                raise Exception("Connection timeout")

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

        return True

    def disconnect(self):
        self.running = False
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()

    def publish_message(self, topic: str, payload: Dict[Any, Any], qos: int = 0):
        if not self.connected:
            print("❌ Not connected to broker")
            return False

        try:
            json_payload = json.dumps(payload)
            result = self.client.publish(topic, json_payload, qos)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📡 Published to {topic}: {json_payload}")
                return True
            else:
                print(f"❌ Failed to publish message: {result.rc}")
                return False

        except Exception as e:
            print(f"❌ Error publishing message: {e}")
            return False

    def generate_sensor_data(self, sensor_id: str) -> Dict[str, Any]:
        return {
            "sensor_id": sensor_id,
            "temperature": round(random.uniform(18.0, 35.0), 2),
            "humidity": round(random.uniform(30.0, 80.0), 2),
            "timestamp": datetime.now().isoformat(),
            "battery_level": random.randint(20, 100),
            "status": random.choice(["active", "warning", "normal"]),
        }

    def generate_device_status(self, device_id: str) -> Dict[str, Any]:
        return {
            "device_id": device_id,
            "status": random.choice(["online", "offline", "maintenance"]),
            "cpu_usage": round(random.uniform(10.0, 90.0), 2),
            "memory_usage": round(random.uniform(20.0, 85.0), 2),
            "uptime": random.randint(100, 100000),
            "timestamp": datetime.now().isoformat(),
        }

    def start_sensor_simulation(self, interval: float = 5.0, sensors: list = None):
        if sensors is None:
            sensors = ["sensor_001", "sensor_002", "sensor_003"]

        self.running = True
        print(
            f"🚀 Starting sensor simulation with {len(sensors)} sensors (interval: {interval}s)"
        )

        def simulation_loop():
            while self.running and self.connected:
                for sensor_id in sensors:
                    if not self.running:
                        break

                    sensor_data = self.generate_sensor_data(sensor_id)
                    self.publish_message(f"sensor/{sensor_id}/data", sensor_data)

                    if random.random() < 0.3:
                        device_status = self.generate_device_status(sensor_id)
                        self.publish_message(
                            f"device/{sensor_id}/status", device_status
                        )

                time.sleep(interval)

        thread = threading.Thread(target=simulation_loop, daemon=True)
        thread.start()
        return thread

    def publish_custom_data(self, topic: str, data: Dict[str, Any]):
        return self.publish_message(topic, data)


def run_simulation_mode(publisher, args):
    print(f"📊 Starting sensor simulation mode...")
    thread = publisher.start_sensor_simulation(args.interval, args.sensors)

    print("📋 Commands: 'status', 'sensors', 'quit'")
    print("-" * 50)

    while publisher.running:
        try:
            command = input(">>> ").strip().lower()

            if command == "quit":
                break
            elif command == "status":
                print(f"🔗 Connected: {publisher.connected}")
                print(f"🏃 Running: {publisher.running}")
                print(f"📡 Broker: {publisher.broker_host}:{publisher.broker_port}")
            elif command == "sensors":
                print(f"📊 Active sensors: {', '.join(args.sensors)}")
            elif command.startswith("publish"):
                parts = command.split(" ", 2)
                if len(parts) == 3:
                    try:
                        topic = parts[1]
                        data = json.loads(parts[2])
                        publisher.publish_custom_data(topic, data)
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON format")
                else:
                    print('Usage: publish topic_name {"key": "value"}')
            elif command == "help":
                print(
                    "📋 Commands: 'status', 'sensors', 'publish topic {\"data\": \"value\"}', 'quit'"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_demo_mode(publisher):
    print("🎮 Demo mode - Publishing sample data...")

    sample_data = [
        (
            "sensor/temperature",
            {"value": 23.5, "unit": "celsius", "location": "office"},
        ),
        ("sensor/humidity", {"value": 65.2, "unit": "percent", "location": "office"}),
        ("device/status", {"device_id": "dev_001", "status": "online", "uptime": 3600}),
        (
            "alert/warning",
            {"message": "High temperature detected", "severity": "warning"},
        ),
        ("system/info", {"version": "1.0.0", "build": "2024-01-01"}),
    ]

    for topic, data in sample_data:
        publisher.publish_message(topic, data)
        time.sleep(2)

    print("✅ Demo completed!")


def run_manual_mode(publisher):
    print("✋ Manual mode - Enter data manually")
    print('Format: topic_name {"key": "value"}')
    print("Type 'quit' to exit")

    while True:
        try:
            user_input = input("Enter topic and data: ").strip()

            if user_input.lower() == "quit":
                break

            parts = user_input.split(" ", 1)
            if len(parts) == 2:
                topic, json_data = parts
                data = json.loads(json_data)
                publisher.publish_custom_data(topic, data)
            else:
                print('Invalid format. Use: topic_name {"key": "value"}')

        except json.JSONDecodeError:
            print("❌ Invalid JSON format")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="MQTT Publisher Server")
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Publishing interval in seconds"
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["sensor_001", "sensor_002", "sensor_003"],
        help="List of sensor IDs",
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "manual", "demo"],
        default="simulation",
        help="Publishing mode",
    )

    args = parser.parse_args()

    publisher = MQTTPublisher(
        broker_host=args.host, broker_port=args.port, client_id="mqtt_data_publisher"
    )

    if not publisher.connect():
        print("❌ Failed to connect to MQTT broker. Exiting...")
        return

    try:
        if args.mode == "simulation":
            run_simulation_mode(publisher, args)
        elif args.mode == "demo":
            run_demo_mode(publisher)
        elif args.mode == "manual":
            run_manual_mode(publisher)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")

    finally:
        print("🔌 Disconnecting...")
        publisher.disconnect()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
