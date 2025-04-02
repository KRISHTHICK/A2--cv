import torch
from PIL import Image
import numpy as np

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Parse results
    detected_objects = results.pandas().xyxy[0].name.tolist()

    return detected_objects

if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'
    objects = detect_objects(image_path)
    print("Objects detected in the image:")
    print(objects)
