#!/bin/bash

# Redirect stdout and stderr to a log file
exec > /var/log/user-data.log 2>&1

# Ensure all commands are run with superuser privileges
echo "Running as user: $(whoami)"

apt-get update -y 
apt-get upgrade -y
apt-get install -y redis-server 
apt-get install -y python3-pip 
apt-get install -y python3-venv
apt-get install -y gunicorn
pip install redis

# download repo
cd /home/ubuntu
git clone https://github.com/cgordon-dev/pneumonia_detection_deployment.git /home/ubuntu/CNN_deploy

# Set permissions on the repo
sudo chown -R ubuntu:ubuntu /home/ubuntu/CNN_deploy
