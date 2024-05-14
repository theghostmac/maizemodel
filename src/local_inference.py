import torch
import cv2
from PIL import Image
from torchvision import transforms
from time import sleep

# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.eval()

# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera could not be opened")
    exit()

# Define the transformation: Convert image to tensor
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # Resize the image
    transforms.ToTensor(),  # Converts image data to PyTorch tensor
])

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            sleep(0.5)
            continue

        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for compatibility with transforms
        pil_image = Image.fromarray(rgb_image)

        # Apply the transformation to the image
        input_tensor = transform(pil_image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            results = model(input_tensor)

        # Check the output format of the results
        print("Output from the model:", results)
        print("Type of results:", type(results))

        # Assuming the results are in the expected format, try accessing the first element
        if hasattr(results, 'xyxy'):
            for det in results.xyxy[0]:
                print("Detection:", det)
        else:
            print("The results object does not have 'xyxy' attribute.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(0.1)

except KeyboardInterrupt:
    print("Stopped by User")
finally:
    camera.release()
    cv2.destroyAllWindows()
