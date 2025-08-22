

## 📖 Table of Contents
- [Introduction](#introduction)
- [MQTT Protocol Overview](#mqtt-protocol-overview)
- [Paho MQTT Library](#paho-mqtt-library)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Testing Guide](#testing-guide)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🌟 Introduction

This project demonstrates a **FastAPI application** integrated with **MQTT messaging** for real-time IoT data processing. The application consists of:

- **FastAPI Server**: REST API with MQTT client integration
- **MQTT Publisher**: Simulates IoT devices sending sensor data
- **MQTT Broker**: Message broker for communication

## 🔌 MQTT Protocol Overview

### What is MQTT?

**MQTT (Message Queuing Telemetry Transport)** is a lightweight messaging protocol designed for IoT devices and low-bandwidth networks.

### Key Concepts

- **Publisher**: Sends messages to topics
- **Subscriber**: Receives messages from topics  
- **Broker**: Routes messages between publishers and subscribers
- **Topic**: Channel for message categorization (e.g., `sensor/temperature`)
- **QoS (Quality of Service)**: Message delivery guarantee levels

### MQTT QoS Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | At most once | Fire and forget (no guarantee) |
| 1 | At least once | Guaranteed delivery (may duplicate) |
| 2 | Exactly once | Guaranteed single delivery |

### MQTT Topics Structure

```
sensor/001/temperature    # Temperature from sensor 001
sensor/001/humidity      # Humidity from sensor 001
device/001/status        # Device status
alerts/critical          # Critical alerts
```

## 📚 Paho MQTT Library

### What is Paho?

**Paho** is the most popular Python MQTT client library, providing:

- ✅ **Full MQTT 3.1.1 support**
- ✅ **Asynchronous and synchronous APIs**
- ✅ **SSL/TLS encryption support**
- ✅ **Automatic reconnection**
- ✅ **Message persistence**

### Key Paho Features Used

```python
import paho.mqtt.client as mqtt

# Create client
client = mqtt.Client(client_id="unique_id")

# Set callbacks
client.on_connect = on_connect_callback
client.on_message = on_message_callback

# Connect and subscribe
client.connect("broker_host", 1883, 60)
client.subscribe("sensor/+/data")  # + is wildcard

# Publish message
client.publish("sensor/001/data", payload, qos=1)
```

### Callback Functions

- **`on_connect`**: Called when client connects to broker
- **`on_message`**: Called when message is received
- **`on_publish`**: Called when message is successfully published
- **`on_disconnect`**: Called when client disconnects

## 🏗️ Project Architecture

### Directory Structure

```
fastapi_mqtt_app/
├── src/
│   ├── main.py                     # FastAPI application entry point
│   ├── mqtt_publisher_server.py    # MQTT data publisher
│   ├── config/
│   │   └── settings.py            # Application configuration
│   └── app/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── endpoints/
│       │       ├── __init__.py
│       │       └── mqtt.py        # MQTT API endpoints
│       ├── core/
│       │   ├── __init__.py
│       │   └── mqtt_client.py     # MQTT client manager
│       └── models/
│           ├── __init__.py
│           └── mqtt_models.py     # Pydantic data models
├── requirements.txt               # Python dependencies
└── README.md
```

### Architecture Components

#### 1. **FastAPI Application** (`main.py`)
- REST API server with automatic documentation
- Health check endpoints
- MQTT integration layer

#### 2. **MQTT Client Manager** (`app/core/mqtt_client.py`)
- Handles MQTT broker connections
- Manages message subscriptions and publishing
- Stores recent messages in memory
- Provides thread-safe operations

#### 3. **API Endpoints** (`app/api/endpoints/mqtt.py`)
- REST endpoints for MQTT operations
- Real-time message retrieval
- Publishing interface

#### 4. **Data Models** (`app/models/mqtt_models.py`)
- Pydantic models for data validation
- Type safety and serialization
- Automatic API documentation

#### 5. **MQTT Publisher** (`mqtt_publisher_server.py`)
- Simulates IoT devices
- Generates realistic sensor data
- Multiple operation modes

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Docker (optional, for MQTT broker)

### Step 1: Create Project Structure

```bash
# Create project directory
mkdir fastapi_mqtt_app
cd fastapi_mqtt_app

# Create directory structure
mkdir -p src/config src/app/api/endpoints src/app/core src/app/models

# Create __init__.py files
touch src/app/__init__.py
touch src/app/api/__init__.py
touch src/app/api/endpoints/__init__.py
touch src/app/core/__init__.py
touch src/app/models/__init__.py
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install packages
pip install fastapi==0.104.1 uvicorn==0.24.0 paho-mqtt==1.6.1 pydantic==2.5.0 pydantic-settings==2.1.0 python-multipart==0.0.6
```

### Step 3: Create Configuration Files

Create all the necessary files as shown in the code artifacts provided earlier.

### Step 4: Setup MQTT Broker

Choose one of these options:

#### Option A: Docker (Recommended)
```bash
docker run -it -p 1883:1883 -p 9001:9001 eclipse-mosquitto
```

#### Option B: Local Installation (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

#### Option C: Public Broker (Testing Only)
Update `src/config/settings.py`:
```python
mqtt_broker_host: str = "test.mosquitto.org"
```

## 🏃‍♂️ Running the Application

### Terminal Setup (3 terminals required)

#### Terminal 1: MQTT Broker
```bash
# Using Docker
docker run -it -p 1883:1883 eclipse-mosquitto

# Or if installed locally
sudo systemctl start mosquitto
```

#### Terminal 2: FastAPI Server
```bash
cd fastapi_mqtt_app/src
python main.py
```

**Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
Connected to MQTT broker
```

#### Terminal 3: MQTT Publisher
```bash
cd fastapi_mqtt_app/src

# Simulation mode (auto-generates data)
python mqtt_publisher_server.py --mode simulation --interval 3

# Demo mode (sample data)
python mqtt_publisher_server.py --mode demo

# Manual mode (interactive)
python mqtt_publisher_server.py --mode manual
```

### Publisher Modes

#### Simulation Mode
```bash
python mqtt_publisher_server.py --mode simulation --interval 2 --sensors sensor_001 sensor_002
```
- Automatically generates sensor data
- Configurable interval and sensor IDs
- Interactive commands available

#### Demo Mode
```bash
python mqtt_publisher_server.py --mode demo
```
- Publishes predefined sample data
- Good for testing different data types

#### Manual Mode  
```bash
python mqtt_publisher_server.py --mode manual
```
- Interactive data entry
- Format: `topic_name {"key": "value"}`

## 🌐 API Endpoints

### Base URLs
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Available Endpoints

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy"
}
```

#### 2. Root Information
```http
GET /
```
**Response:**
```json
{
  "message": "Welcome to FastAPI MQTT Application",
  "docs": "/docs",
  "mqtt_status": "/api/v1/mqtt/status"
}
```

#### 3. MQTT Connection Status
```http
GET /api/v1/mqtt/status
```
**Response:**
```json
{
  "connected": true,
  "broker": "localhost",
  "port": 1883
}
```

#### 4. Recent MQTT Messages
```http
GET /api/v1/mqtt/messages
```
**Response:**
```json
[
  {
    "topic": "sensor/001/data",
    "payload": "{\"temperature\": 23.5, \"humidity\": 65.2}",
    "timestamp": "2024-01-15T10:30:00.123456",
    "qos": 0
  }
]
```

#### 5. Publish MQTT Message
```http
POST /api/v1/mqtt/publish
Content-Type: application/json

{
  "topic": "sensor/test",
  "payload": "Hello MQTT",
  "qos": 1
}
```
**Response:**
```json
{
  "status": "success",
  "message": "Message published successfully"
}
```

## 🧪 Testing Guide

### 1. Verify Setup
```bash
# Check if FastAPI is running
curl http://localhost:8000/health

# Check MQTT status
curl http://localhost:8000/api/v1/mqtt/status
```

### 2. Test Message Flow
```bash
# Terminal 1: Start publisher
python mqtt_publisher_server.py --mode simulation --interval 5

# Terminal 2: Check received messages
curl http://localhost:8000/api/v1/mqtt/messages
```

### 3. Test Publishing via API
```bash
curl -X POST "http://localhost:8000/api/v1/mqtt/publish" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "test/api",
    "payload": "API Test Message",
    "qos": 1
  }'
```

### 4. Interactive Testing
Use the FastAPI docs at http://localhost:8000/docs to test all endpoints interactively.

## ⚙️ Configuration

### Environment Variables
Create `.env` file in the project root:
```env
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC=sensor/data
MQTT_CLIENT_ID=fastapi_client
APP_NAME=My MQTT App
```

### Settings Class
```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FastAPI MQTT Application"
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topic: str = "sensor/data"
    mqtt_client_id: str = "fastapi_client"
    
    class Config:
        env_file = ".env"
```

### Publisher Configuration
```bash
# Custom broker
python mqtt_publisher_server.py --host 192.168.1.100 --port 1883

# Custom sensors
python mqtt_publisher_server.py --sensors sensor_01 sensor_02 sensor_03

# Custom interval
python mqtt_publisher_server.py --interval 1.5
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
**Error:** `ModuleNotFoundError: No module named 'config'`
**Solution:**
```bash
# Make sure you're running from the correct directory
cd src
python main.py

# Or set PYTHONPATH
PYTHONPATH=src python src/main.py
```

#### 2. Pydantic Import Error  
**Error:** `BaseSettings has been moved to pydantic-settings`
**Solution:**
```bash
pip install pydantic-settings
```

#### 3. MQTT Connection Failed
**Error:** `Failed to connect to MQTT broker`
**Solutions:**
- Check if broker is running: `docker ps` or `systemctl status mosquitto`
- Verify broker host/port in settings
- Check firewall/network connectivity
- Try public broker: `test.mosquitto.org`

#### 4. No Messages Received
**Possible Causes:**
- Publisher and client using different brokers
- Topic name mismatch
- Publisher not running
- Network connectivity issues

**Debug Steps:**
```bash
# Test broker connectivity
mosquitto_pub -h localhost -t test -m "hello"
mosquitto_sub -h localhost -t test

# Check FastAPI logs for connection status
# Check publisher logs for publish confirmations
```

#### 5. Port Already in Use
**Error:** `[Errno 98] Address already in use`
**Solutions:**
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or use different port
uvicorn main:app --port 8001
```

### Debug Tips

1. **Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Monitor MQTT Traffic:**
```bash
mosquitto_sub -h localhost -t '#' -v
```

3. **Check Network Connectivity:**
```bash
telnet localhost 1883
```

4. **Validate JSON Payloads:**
```python
import json
json.loads(payload)  # Will raise error if invalid
```

## 📝 Next Steps

### Enhancements
- Add authentication and authorization
- Implement message persistence
- Add WebSocket support for real-time updates
- Create Docker compose setup
- Add monitoring and metrics
- Implement message routing and filtering

### Production Considerations
- Use proper MQTT broker (not public ones)
- Implement SSL/TLS encryption
- Add proper logging and monitoring
- Use environment-specific configurations
- Implement proper error handling and retries
- Add unit and integration tests

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review FastAPI documentation: https://fastapi.tiangolo.com/
3. Check Paho MQTT documentation: https://www.eclipse.org/paho/index.php?page=clients/python/index.php
4. MQTT protocol specification: http://mqtt.org/

Happy coding! 🚀