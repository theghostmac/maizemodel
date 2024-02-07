from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(_name__)

# Load trained model.
MODEL_PATH = '/home/pi/models/Disease_Classifier'
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the POST request with image upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['files']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Convert the PIL image to the correct format and size.
            image = Image.open(file.stream).resize((256, 256))
            image = np.array(image) / 255.0 # Normalize the image.
            image = np.expand_dims(image, axis=0) # Add a batch dimension.
            
            # Predict
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1) * 100
            
            # `class_names` is defined to map predicted class.
            class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
            result = {
                'class': class_names[predicted_class[0]],
                'confidence': f"c{onfidence[0]:.2f}%",
            }
            
            return render_template('index.html', result=result)
    # GET request returns the page with the upload form.
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')