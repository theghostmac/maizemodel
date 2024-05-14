import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Load the model
model_path = '/Users/macbobbychibuzor/workspace/internship/maizemodel/models/Disease_Classifier'
model = tf.keras.models.load_model(model_path)

# Simulate image capture by loading an image from disk
image_path = '/Users/macbobbychibuzor/workspace/internship/maizemodel/src/bligggght.JPG'

# Load and inspect the image
image = Image.open(image_path)
print(f"the image is at: {image_path}")
print(f"Loaded image shape (before resizing): {image.size}")
image = image.resize((256, 256))  # Resize to match the model's expected input
image_array = np.array(image)
print(f"Image shape (after resizing): {image_array.shape}")
print(f"Image data sample (before normalization): {image_array[0, 0, :]}")

# Normalize the image
image = image_array / 255.0
image = np.expand_dims(image, axis=0)  # Add a batch dimension

# Predict and inspect model output
predictions = model.predict(image)
print(f"Raw model output probabilities: {predictions}")

predicted_class = np.argmax(predictions)
confidence = np.max(predictions) * 100
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
print(f"Predicted class: {class_names[predicted_class]}, Confidence: {confidence:.2f}%")
