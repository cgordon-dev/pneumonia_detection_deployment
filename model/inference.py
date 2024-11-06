import redis
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# Connect to Redis
r = redis.Redis(host='10.0.2.111', port=6379, db=0)

# Load the model
model = load_model('/home/ubuntu/models/pneumonia_model.keras')

# Directory containing test images
test_dir = '/home/ubuntu/chest_xray/test/'

# Traverse through the 'PNEUMONIA' and 'NORMAL' directories
for root, dirs, files in os.walk(test_dir):
    for filename in files:
        if filename.endswith(('.jpeg', '.jpg', '.png')):
            try:
                # Full path to image
                image_path = os.path.join(root, filename)

                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
                img_array = np.expand_dims(img_array, axis=0)

                # Make prediction
                prediction = model.predict(img_array, verbose=0)
                confidence = float(prediction[0][0])
                result = "Pneumonia" if confidence > 0.5 else "Normal"
                confidence_percentage = confidence * 100  # Convert to percentage

                # Store result in Redis
                r.hset(filename, mapping={
                    'prediction': result,
                    'confidence': f"{confidence_percentage:.2f}%"  # Store as percentage string
                })

                # Print result
                print(f"File: {filename}")
                print(f"Prediction: {result} (confidence: {confidence_percentage:.2f}%)")
                print("-" * 50)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")