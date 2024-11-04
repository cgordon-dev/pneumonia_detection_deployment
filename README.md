# Pneumonia Detection Application

## Purpose

The purpose of this project is to develop an application that enables doctors to upload X-ray images via a frontend interface to detect pneumonia. The application processes the uploaded images through a neural network model hosted on the backend. The model analyzes the images and stores the results in Redis, making them accessible and visible on the frontend for the user.

## Infrastructure Overview

![Pneumonia Detection Infrastructure Diagram](Diagram.png)

This project leverages AWS to deploy a secure, scalable infrastructure using Terraform. Below is a summary of the key components:

1. **Networking**:
   - **VPC**: A Virtual Private Cloud (VPC) named `ml_vpc` to isolate resources and provide secure networking.
   - **Subnets**:
     - Public subnet for resources requiring internet access (e.g., Nginx reverse proxy).
     - Private subnets for application servers and a dedicated training server.
   - **Internet Gateway and NAT Gateway**: The internet gateway allows public internet access, while the NAT gateway enables secure outbound internet access from private subnets.
   
2. **Security Groups**:
   - **Frontend Security Group**: Allows HTTP, HTTPS, and SSH access to the public Nginx server.
   - **Backend Security Group**: Restricts access to backend services, allowing only necessary traffic from frontend and internal services.
   - **Monitoring Security Group**: Manages access to Prometheus and Grafana for application health monitoring.
   - **UI Security Group**: Controls access to the application UI server for secure user interaction.

3. **EC2 Instances**:
   - **Nginx Server**: A public server acting as the frontend and reverse proxy.
   - **Monitoring Server**: Hosts Prometheus and Grafana for application health monitoring.
   - **UI Server**: The private server that handles user interactions.
   - **App Server**: Runs the backend API and services, including Redis.
   - **Training Server**: A GPU-enabled instance for model training and inference.

## Steps

The application involves several essential steps to ensure model accuracy, reliability, and efficient visibility of results on the frontend. Here’s a step-by-step breakdown of the workflow, technical components, and code snippets for the application.

### 1. Data Storage and Access (AWS S3)
   - **Setup**: All X-ray images are stored in an Amazon S3 bucket (`x-raysbucket`), which serves as a scalable and secure data storage solution.
   - **Purpose**: S3 provides a centralized and durable storage location where images can be accessed by the backend for processing and inference. This setup ensures that data remains secure and accessible across various services.
   - **Code**:
     ```bash
     # Install AWS CLI
     !pip install awscli
     !mkdir -p /content/chest_xray/
     # Download dataset from S3
     !aws s3 cp s3://x-raysbucket/chest_xray/ /content/chest_xray/ --recursive --no-sign-request
     ```

### 2. Data Preprocessing and Augmentation
   - **ImageDataGenerator**: For training, data augmentation is performed using Keras’s `ImageDataGenerator`, which applies transformations such as rotation, scaling, and flipping.
   - **Purpose**: Augmentation artificially expands the dataset, allowing the model to generalize better by learning from diverse variations of the same image. This reduces overfitting and improves accuracy, especially important when working with limited medical datasets.
   - **Code**:
     ```python
     from tensorflow.keras.preprocessing.image import ImageDataGenerator

     train_datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=10,
         width_shift_range=0.1,
         height_shift_range=0.1,
         horizontal_flip=True
     )
     val_datagen = ImageDataGenerator(rescale=1./255)
     test_datagen = ImageDataGenerator(rescale=1./255)
     ```

### 3. Model Selection and Transfer Learning (ResNet50)
   - **Base Model (ResNet50)**: The application uses a pre-trained ResNet50 model, which has been trained on the ImageNet dataset. The model’s final layers are replaced and fine-tuned for pneumonia detection.
   - **Layer Freezing**: The initial layers of ResNet50 are frozen to retain the pre-trained weights, which capture fundamental image features like edges and textures.
   - **Code**:
     ```python
     from tensorflow.keras.applications import ResNet50
     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

     # Unfreeze last 20 layers of ResNet50
     for layer in base_model.layers[-20:]:
         layer.trainable = True
     ```

### 4. Custom Layers for Binary Classification
   - **Dense Layers and Dropout**: Adds additional layers with dropout to prevent overfitting.
   - **Output Layer (Sigmoid Activation)**: A single neuron with sigmoid activation is used for binary classification.
   - **Code**:
     ```python
     from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

     x = base_model.output
     x = GlobalAveragePooling2D()(x)
     x = BatchNormalization()(x)
     x = Dense(128, activation='relu')(x)
     x = BatchNormalization()(x)
     x = Dense(256, activation='relu')(x)
     x = BatchNormalization()(x)
     x = Dense(128, activation='relu')(x)
     x = Dropout(0.5)(x)
     x = Dense(1, activation='sigmoid')(x)

     model = Model(inputs=base_model.input, outputs=x)
     ```

### 5. Model Training with Optimizations
   - **Early Stopping and Checkpointing**: Stops training if validation loss stops improving and saves the best model weights.
   - **Learning Rate Scheduler**: Reduces the learning rate when validation loss plateaus.
   - **Code**:
     ```python
     from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

     callbacks = [
         EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
         ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
         ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
     ]

     model.compile(
         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
         loss='binary_crossentropy',
         metrics=['accuracy']
     )

     history = model.fit(
         train_generator,
         steps_per_epoch=train_generator.samples // train_generator.batch_size,
         validation_data=val_generator,
         validation_steps=val_generator.samples // val_generator.batch_size,
         epochs=30,
         class_weight=class_weight,
         callbacks=callbacks
     )
     ```

### 6. Frontend-Backend Communication (Nginx Reverse Proxy)
   - **Nginx**: Acts as a reverse proxy for the application frontend, managing HTTP requests and providing secure communication with backend services.

### 7. Backend API and Redis Caching
   - **Backend API (Flask with Gunicorn)**: The backend API is built using Flask and deployed with Gunicorn. It processes images, retrieves prediction results, and stores them in Redis for quick access.

### 8. Model Inference (ResNet Model with GPU Support)
   - **p3.2xlarge EC2 Instance**: The model is deployed on a GPU-enabled EC2 instance, optimized for deep learning inference tasks.

### 9. Frontend Display and User Interface
   - **Frontend UI**: Built with HTML and JavaScript, the frontend enables doctors to upload X-ray images and view predictions in real-time.

## Troubleshooting

During development, several challenges were encountered, especially with server setup, model configuration, and system integration:

- **SSH Tunneling between Private and Public Servers**:
   - Solution: Configured SSH tunneling and security groups to ensure secure communication.
- **Configuring NVIDIA Drivers on the p3 EC2 Instance**:
   - Solution: Installed and configured CUDA and NVIDIA drivers.
- **Tuning the Model to Improve Accuracy**:
   - Solution: Adjusted learning rate, added dropout layers, and employed batch normalization.

## Optimization

To further optimize this deployment:
1. **Optimize Data Caching**: Improve Redis configurations or switch to an in-memory database for faster response times.
2. **Scale GPU Instances Dynamically**: Scale the p3 instance based on user demand to manage costs.
3. **Optimize Frontend Load Balancing**: Use an AWS ELB to handle high user loads.
4. **Use Managed Services for Monitoring**: Replace Prometheus and Grafana with Amazon CloudWatch.

## Conclusion

This project provided a comprehensive experience in deploying a machine learning model in a cloud environment, integrating frontend and backend components, and managing resources efficiently.

Key takeaways include:
- **Transfer Learning**: Using ResNet50 accelerated model training and improved accuracy.
- **Cloud Architecture**: Configuring secure communication between private and public servers, using Nginx as a reverse proxy, and managing data flow through Redis taught valuable lessons in cloud infrastructure.
- **Resource Management**: Setting up a GPU-enabled instance (p3.2xlarge) and troubleshooting NVIDIA drivers highlighted the challenges of handling specialized hardware in the cloud.
