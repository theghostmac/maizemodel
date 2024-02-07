from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import jsonify
import os

app = Flask(__name__)

# Load the trained model
# MODEL_PATH = '/home/pi/models/Disease_Classifier' # uncomment for raspberry pi
MODEL_PATH = '/Users/macbobbychibuzor/workspace/internship/maizemodel/models/Disease_Classifier' # uncomment for local machine.
model = tf.keras.models.load_model(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the POST request with image upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Convert the PIL image to the correct format and size
            image = Image.open(file.stream).resize((256, 256))
            image = np.array(image) / 255.0  # Normalize the image
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Predict
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1) * 100

            # Assuming `class_names` is defined to map predicted class indices to actual names
            class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
            result = {
                'class': class_names[predicted_class[0]],
                'confidence': f"{confidence[0]:.2f}%"
            }

            return render_template('index.html', result=result)
    # GET request returns the page with the upload form
    return render_template('index.html', result=None)


@app.route('/capture', methods=['GET'])
def capture():
    with PiCamera() as camera:
        camera.resolution = (256, 256)  # Match the model's expected input
        camera.start_preview()
        # Allow camera to warm up
        time.sleep(2)
        # Capture an image directly into the numpy array
        # image_path = '/home/pi/captured_image.jpg' # uncomment for raspberry pi.
        image_path = '/Users/macbobbychibuzor/workspace/internship/maizemodel/src/some_image.jpg'
        camera.capture(image_path)
        camera.stop_preview()

    image = Image.open(image_path)
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0] * 100

    class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    return jsonify({
        'class': class_names[predicted_class],
        'confidence': f"{confidence:.2f}%"
    })



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
