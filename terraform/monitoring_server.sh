#!/bin/bash

# Redirect stdout and stderr to a log file
exec > /var/log/user-data.log 2>&1

echo "${priv_mykey}" > /home/ubuntu/.ssh/mykey.pem

# Ensure all commands are run with superuser privileges
echo "Running as user: $(whoami)"

# Update the system
apt update -y
apt upgrade -y

# Install wget
apt install -y wget software-properties-common

# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.37.0/prometheus-2.37.0.linux-amd64.tar.gz
tar xvfz prometheus-2.37.0.linux-amd64.tar.gz
mv prometheus-2.37.0.linux-amd64 /opt/prometheus

# Create a Prometheus user
useradd --no-create-home --shell /bin/false prometheus

# Set ownership for Prometheus directories
chown -R prometheus:prometheus /opt/prometheus

# Create a Prometheus service file
cat << EOF | sudo tee /etc/systemd/system/prometheus.service
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/opt/prometheus/prometheus \
    --config.file /opt/prometheus/prometheus.yml \
    --storage.tsdb.path /opt/prometheus/data

[Install]
WantedBy=multi-user.target
EOF

# Start and enable Prometheus service
systemctl daemon-reload
systemctl start prometheus
systemctl enable prometheus

# Install Grafana
# Add Grafana GPG key and repository
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
add-apt-repository "deb https://packages.grafana.com/oss/deb stable main" -y

# Update package list and install Grafana
apt update -y
apt install grafana -y

# Start and enable Grafana service
systemctl start grafana-server
systemctl enable grafana-server