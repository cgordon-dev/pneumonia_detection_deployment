Kura Labs AI Workload 1
# Pneumonia Detection Application

## Overview

Welcome! You are an MLOps engineer working in a specialized team at Mount Sinai Hospital. Your team has developed a neural network application that allows doctors to upload X-ray images and receive a prediction on whether or not the X-ray indicates pneumonia. This application currently displays prediction results along with the percent accuracy for each diagnosis. The infrastructure, including backend, frontend, and monitoring, was manually configured to allow these components to interact seamlessly, and they are connected as shown in [this repo](https://github.com/elmorenox/CNN_deploy/blob/main/README.md). 

Initially, the application’s web server was accessible on a public subnet, where the UI was served on `public_ip:5001`. However, for enhanced security, there’s now a requirement to move the application to a private subnet and use Nginx on the public subnet to handle requests and serve the UI from `public_ip:80`. 

Additionally, concerns have been raised about model performance, as predictions show a tendency to classify everything as pneumonia. Your team is now tasked with creating the infrastructure and connections for this application, sending accurate metrics to Prometheus and Grafana, and retraining the model to reduce bias in predictions and align with the updated system architecture shown below. 

![Screenshot 2024-10-26 130236](https://github.com/user-attachments/assets/43d32683-b65d-471f-a58f-c87a33c8529b)

Please follow the steps below.

## Steps

A. **Set up Development Environment**
   - Spin up a `t3.medium` instance in your AWS account’s VPC.
   - Install **Terraform** (**VSCode** is optional but recommended).
   
B. **Clone and Prepare Repository**
   - Clone the project repository and upload it to your own GitHub account.
   - Update any file references to the repository URL to point to your new GitHub repository.
   
C. **Create Infrastructure**
   - Terraform apply the configurations for your application environment. 
   
D. **Configure**
   - In order to make sure all connections are made in the backend and frontend, please follow the steps below:

### Key Pair Creation
1. **Create a key pair** called `mykey` in your AWS account. This will allow you to SSH into the instances in the private subnets for configurations.
   - **Question:** Why is that necessary?

### Monitoring Server Setup
2. **Connect to the Monitoring Server**.
   - Navigate to `/opt/prometheus/prometheus.yml`.
   - Under `scrape_configs`, add the following:
     ```yaml
     - job_name: 'node_exporter'
       static_configs:
     - targets: ['${ML_TRAINING_SERVER_IP}:9100', '${ML_TRAINING_SERVER_IP}:8000']
     ```
   - In the terminal, execute:
     ```
     sudo systemctl restart prometheus
     ```
Make sure Prometheus and Grafana are configured correctly to grab information from the node exporter on the Application Server. 

### Nginx Server Configuration
3. **Connect to the Nginx Server**.
   - Navigate to `/etc/nginx/sites-enabled/default`.
   - After the server listen configuration, add the following (replace `{UI_SERVER_IP}` with UI server IP):
     ```nginx
     location / {
         proxy_pass http://${UI_SERVER_IP}:5001;
         proxy_set_header Host $host;
         proxy_set_header X-Real-IP $remote_addr;
         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
         proxy_set_header X-Forwarded-Proto $scheme;
     }
     ```
   - In the terminal, execute:
     ```
     sudo systemctl restart nginx
     ```
Make sure that Nginx routes traffic correctly. 

### Application Server Setup
4. **Exit the Monitoring Server and SSH into the Application Server**.
   - Execute:
     ```
     sudo reboot
     ```
   - After connecting again, in `/etc/redis/redis.conf`, modify these lines:
     ```
     bind 0.0.0.0
     protected-mode no
     ```
   - Run the following commands in the terminal:
     ```
     sudo systemctl restart redis
     ```

### ML Training Server Configuration
5. **Exit the Application Server and SSH into the ML Training Server**.
   - Navigate to `CNN_deploy/model/inference.py` and put the private IP of the Application Server where it says `backendapi_private_ip`.
     - **Question:** What connection are we making here and why?
   - In the terminal, execute the following commands:
     ```
     sudo reboot
     #### connect again ####
     cd CNN_deploy/model
     python3 -m venv venv
     source venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     python3 cnn.py
     python3 inference.py
     ```
   - Move the model (`best_model.keras`) to the application server by adding your `mykey.pem` and using SCP:
     ```
     scp -i ~/.ssh/mykey.pem /home/ubuntu/CNN_deploy/model/best_model.keras ubuntu@10.0.2.162:/home/ubuntu/CNN_deploy/pneumonia_api
     ```
Make sure that the saved model (post-training) is accessible in the Application Server for Redis.

### Application Server Final Setup
6. **Exit the ML Training server and SSH back into the Application Server**.
   - Execute:
     ```
     cd ~/CNN_deploy/pneumonia_api
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     gunicorn --bind 0.0.0.0:5000 app:app
     ```
Make sure that the application api is up. 

### Frontend Server Configuration
7. **Exit the ML Training Server and SSH into the UI Server**.
   - Navigate to `CNN_deploy/pneumonia_web/app.py` and replace `API_URL` with the private IP of the Application Server.
   - In the terminal, execute:
     ```
     cd ~/CNN_deploy/pneumonia_web
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     gunicorn --config gunicorn_config.py app:app
     ```
Make sure that the frontend api is up.
   
E. **Monitor and Fix Model**
   - Update the model to accurately detect pneumonia in scans. When you update your model (cnn.py), you have to do steps 5 and 6 again. 

---

## Documentation
You are responsible for creating a README.md in your repo with the following:

### Purpose
What is the purpose of this project?

### Steps
The application is designed to allow x-ray images to be uploaded via the frontend, processed by a neural network model in the backend, the results stored in Redis to be displayed on the UI. Explain the essential steps taken in order to make sure that the model is accurate and visible on the frontend to the user. Why did you have the steps you did in the order you did?

### Troubleshooting
Explain the issues you ran into and how you resolved them.

### Optimization
Explain how would you optimize this deployment?

### Conclusion
Share what you took away from this project. 

--- 
