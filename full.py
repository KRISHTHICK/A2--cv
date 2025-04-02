import torch
from PIL import Image
import numpy as np
import openai

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load your OpenAI API key
openai.api_key = 'your-api-key'

def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Parse results
    detected_objects = results.pandas().xyxy[0].name.tolist()

    return detected_objects

def generate_descriptions(objects):
    descriptions = []
    for obj in objects:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Describe a {obj} in detail.",
            max_tokens=50
        )
        description = response.choices[0].text.strip()
        descriptions.append(description)
    return descriptions

if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'
    objects = detect_objects(image_path)
    print("Objects detected in the image:")
    print(objects)

    descriptions = generate_descriptions(objects)
    print("Generated descriptions:")
    for obj, desc in zip(objects, descriptions):
        print(f"{obj}: {desc}")
