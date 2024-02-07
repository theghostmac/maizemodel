import numpy as np
import tensorflow as tf
from picamera import PiCamera
from PIL import Image
import time

# Load the model.
model = tf.keras.models.load_model('/home/pi/models/Disease_Classifier')

camera = PiCamera()
camera.resolution = (256, 256) # Match the model's expected output.
camera.start_preview()

# Allow camera to warm up.
time.sleep(2)

# Capture an image.
camera.capture('/home/pi/image.jpg')
image = Image.open('/home/pi/image.jpg')
image = np.array(image) / 255.0 # Normalize the image.
image = np.expand_dims(image, axis=0) # Add a batch dimension.

# Predict.
predictions = model.predict(image)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")

camera.stop_preview()