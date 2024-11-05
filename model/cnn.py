import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# Define the data directory
data_dir = "./chest_xray"

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Load the ResNet50 model (pre-trained on ImageNet) and set up for transfer learning
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last 20 layers of the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add custom layers on top of ResNet50 for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces the feature map to a vector
x = BatchNormalization()(x)       # Add batch normalization to stabilize learning
x = Dense(128, activation='relu')(x)  # Reduced number of neurons for simplicity
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)               # Adjusted dropout rate
output = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Define the complete model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with a reduced learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Reduced learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

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
# Verify the paths in the `flow_from_directory` method to match the download location
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
# Adjusted class weights for better balance between classes
class_weight = {0: 1.2, 1: 1.3}

# Define callbacks for early stopping, model checkpointing, and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)  # Adjusted patience
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

# Print training and validation metrics per epoch
print("\nEpoch\tTraining Loss\tValidation Loss\tTraining Accuracy\tValidation Accuracy")

# Iterate over each epoch's results and print them
for epoch in range(len(history.history['loss'])):
    train_loss = history.history['loss'][epoch]
    val_loss = history.history['val_loss'][epoch]
    train_accuracy = history.history['accuracy'][epoch]
    val_accuracy = history.history['val_accuracy'][epoch]

    # Print the data for the current epoch
    print(f"{epoch+1}\t{train_loss:.4f}\t\t{val_loss:.4f}\t\t{train_accuracy:.4f}\t\t{val_accuracy:.4f}")

# Save the trained model
#model.save('pneumonia_model.keras')
model.save('/home/ubuntu/models/pneumonia_model.keras')  # Using .keras format instead of .h5
# Load the trained model for inference
from tensorflow.keras.models import load_model

import os
# test the model all images in /content/chest_xray/test and get the predictions of each image
# Get the list of image files in the test directory
test_dir = '/chest_xray/test'
for root, dirs, files in os.walk(test_dir):
  for file in files:
    if file.endswith(('.jpg', '.jpeg', '.png')):
      test_image_path = os.path.join(root, file)
      # Load and preprocess the image
      img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
      img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
      img_array = np.expand_dims(img_array, axis=0)

      # Make the prediction
      prediction = model.predict(img_array)
      confidence = float(prediction[0][0])
      result = "Pneumonia" if confidence > 0.5 else "Normal"

      # Print the results
      print(f"Image: {test_image_path}")
      print(f"Prediction: {result} (confidence: {confidence:.2%})")
      print("---")

# Evaluate the model on the test set and generate classification metrics
Y_true = test_generator.classes  # True labels
Y_pred = (model.predict(test_generator) > 0.5).astype("int32")  # Binary predictions

# Confusion matrix and classification report
conf_matrix = confusion_matrix(Y_true, Y_pred)
class_report = classification_report(Y_true, Y_pred, target_names=test_generator.class_indices.keys())
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
