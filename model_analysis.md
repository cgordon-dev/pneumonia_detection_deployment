Here’s the complete content in markdown format that you can save as a .md file for GitHub Pages.

# Pneumonia Detection Model: Using Transfer Learning with ResNet50

This notebook demonstrates the process of building, training, and evaluating a pneumonia detection model using a pre-trained ResNet50 network. The model is fine-tuned for binary classification to identify pneumonia from chest X-ray images.

## Prerequisites and Setup

- Install necessary libraries by running the following command:
  ```python
  !pip install tensorflow matplotlib seaborn awscli

	•	Set up AWS CLI to download data from S3. You will need AWS CLI configured with proper permissions.

Import Libraries

# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

Download Data from AWS S3

To download the chest X-ray dataset from AWS S3, ensure you have permission to access the bucket. Run the following code to download the dataset into a specified directory.

import subprocess

# Define the data directory
data_dir = "./chest_xray"

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Run AWS CLI command to download dataset from S3
subprocess.run([
    "aws", "s3", "cp", "s3://x-raysbucket/chest_xray/", data_dir,
    "--recursive", "--no-sign-request"
])

Model Architecture: Using ResNet50 for Transfer Learning

This model leverages the ResNet50 architecture, which is pre-trained on the ImageNet dataset. By fine-tuning the last 20 layers, we can adapt it for pneumonia detection.

from tensorflow.keras.applications import ResNet50

# Load the ResNet50 model (pre-trained on ImageNet) and set up for transfer learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last 20 layers of the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add custom layers on top of ResNet50 for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=output)

Model Compilation

The model is compiled with a reduced learning rate and uses binary cross-entropy as the loss function, suitable for binary classification tasks.

# Compile the model with a reduced learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

Data Preprocessing and Augmentation

We use ImageDataGenerator for data augmentation to enhance model generalization by applying transformations on the training data. Validation and test data are only rescaled to prevent data leakage.

# Set up data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

Model Training

We train the model using early stopping, model checkpointing, and learning rate reduction to optimize training and prevent overfitting.

# Adjusted class weights for better balance between classes
class_weight = {0: 1.2, 1: 1.3}

# Define callbacks for early stopping, model checkpointing, and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks
)

Training and Validation Metrics

The following plot displays the training and validation accuracy and loss over epochs to evaluate the model’s learning progress.

# Plot accuracy and loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

Save the Trained Model

The trained model is saved for later use in inference or evaluation.

model.save('/home/ubuntu/models/pneumonia_model.keras')

Model Inference

We test the model on individual images from the test set and print predictions with confidence scores.

test_dir = '/chest_xray/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])
            result = "Pneumonia" if confidence > 0.5 else "Normal"

            print(f"Image: {test_image_path}")
            print(f"Prediction: {result} (confidence: {confidence:.2%})")
            print("---")

Model Evaluation

Using the test set, we evaluate the model’s performance with a confusion matrix and classification report.

Y_true = test_generator.classes
Y_pred = (model.predict(test_generator) > 0.5).astype("int32")

conf_matrix = confusion_matrix(Y_true, Y_pred)
class_report = classification_report(Y_true, Y_pred, target_names=test_generator.class_indices.keys())

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

Conclusion

This notebook demonstrated the end-to-end process for building and evaluating a pneumonia detection model using a pre-trained ResNet50 with transfer learning. We covered data loading from AWS S3, data augmentation, training, evaluation, and prediction on new images. This pipeline is designed to be scalable and secure, leveraging cloud storage and best practices in deep learning.

Save this content as `pneumonia_detection_model.md` and upload it to your GitHub Pages repository. You can then view it as a webpage through GitHub Pages.