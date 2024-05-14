# MaizeModel

Building the model for our maize disease detection project.

Transfer the entire `Disease_Classifier` directory to Raspberry Pi. Also push the `app.py`, `index.html`, and `pi_inference.py` there.

## Explanation of the Makefile:

`setup`: Sets up the virtual environment and installs all dependencies listed in a requirements.txt file. You'll need to create this file listing all required packages.
`venv`: Creates the virtual environment.
`run`: Activates the virtual environment and runs the main Python script app.py.
`clean`: Cleans up the project by removing the virtual environment and any compiled Python files.
`help`: Provides a list of available commands in the Makefile.
Usage:

To setup the project, just run `make setup`.
To run the application, use `make run`.
To clean up the project, use `make clean`.

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