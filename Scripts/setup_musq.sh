#!/bin/bash

echo "Setting up Mosquitto MQTT Broker..."
echo "Cleaning up existing containers..."
docker-compose down 2>/dev/null || true

echo "Creating directory structure..."
mkdir -p mosquitto/config
mkdir -p mosquitto/data
mkdir -p mosquitto/logs

echo "Creating mosquitto.conf file..."
cat > mosquitto/config/mosquitto.conf << 'EOF'
# Mosquitto Configuration File

# General Settings
persistence true
persistence_location /mosquitto/data/

# Logging
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
log_type error
log_type warning
log_type notice
log_type information
log_timestamp true

# Network Settings
listener 1883 0.0.0.0
protocol mqtt

# WebSocket Support (optional)
listener 9001 0.0.0.0
protocol websockets

# Security Settings
allow_anonymous true

# Connection Settings
max_connections -1
max_inflight_messages 20
max_queued_messages 100
message_size_limit 0

# Keep Alive Settings
keepalive_interval 60

# Client Settings
max_packet_size 0

# Persistence Settings
autosave_interval 1800
autosave_on_changes false

# Memory Settings
memory_limit 0

# Connection timeout
connection_messages true
EOF

echo "Setting permissions..."
sudo chown -R 1883:1883 mosquitto/data mosquitto/logs
sudo chown 1883:1883 mosquitto/config/mosquitto.conf
chmod 644 mosquitto/config/mosquitto.conf
chmod 755 mosquitto/data mosquitto/logs


echo "Starting Mosquitto Docker container..."
docker-compose up -d


sleep 10

if docker ps | grep -q mosquitto-broker; then
    echo "✅ Mosquitto MQTT Broker is running successfully!"
    echo "MQTT Port: 1883"
    echo "WebSocket Port: 9001"
    echo ""
    echo "Test the connection with:"
    echo "docker exec -it mosquitto-broker mosquitto_pub -h localhost -t test/topic -m 'Hello MQTT'"
    echo "docker exec -it mosquitto-broker mosquitto_sub -h localhost -t test/topic"
    echo ""
    echo "Container logs:"
    docker-compose logs --tail=20 mosquitto
else
    echo "❌ Failed to start Mosquitto container"
    echo "Container logs:"
    docker-compose logs mosquitto
fi

echo ""
echo "To view logs: docker-compose logs -f mosquitto"
echo "To stop: docker-compose down"
echo "To restart: docker-compose restart mosquitto"