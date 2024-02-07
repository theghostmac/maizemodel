# MaizeModel

Building the model for our maize disease detection project.

Transfer the entire `Disease_Classifier` directory to Raspberry Pi. Also push the `app.py`, `index.html`, and `pi_inference.py` there.

`assets:` This directory may contain assets that the model uses, such as vocabulary files. 
It might be empty, but it's part of the SavedModel structure.
`variables:` Contains the model's weights. The variables.data-00000-of-00001 and variables.index 
files are critical for your model to have the correct learned parameters.
`saved_model.pb:` Holds the model architecture.
`keras_metadata.pb:` Contains metadata of the model, such as training configuration and the architecture.
`fingerprint.pb:` Used by TensorFlow to ensure the integrity of the model's files and structure.

Copy the model:
```shell
scp -r Disease_Classifier pi@raspberrypi.local:/path/to/destination
```

Load the model:
```shell
model = tf.keras.models.load_model('/path/to/Disease_Classifier')
```

Current Setup:

app.py:
- Handles both manual image uploads via HTML form and direct capture button requests.
- Processes uploaded images within app.py, including resizing, normalization, and prediction.
- Returns the prediction result to the browser as JSON data.

index.html:
- Provides the user interface with an upload form and a capture button.
- Uses JavaScript to send a request to /capture when the capture button is clicked.
- Displays the received prediction result from app.py.