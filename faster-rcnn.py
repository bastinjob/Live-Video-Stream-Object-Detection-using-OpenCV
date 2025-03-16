import cv2
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained Faster R-CNN model and move it to the selected device
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define a transform to convert the frame to a tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Open webcam (try changing index 0 to 1 or 2 if it fails)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could Not Open Webcam")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to PIL Image for the transformation
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the transform and move it to the device (GPU if available)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform face detection (prediction)
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(image_tensor)

    # Move predictions to CPU for processing
    boxes = prediction[0]['boxes'].cpu().numpy()  # Bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Object labels
    scores = prediction[0]['scores'].cpu().numpy()  # Confidence scores

    # Loop through all the predictions
    for box, score in zip(boxes, scores):
        if score > 0.5:  # Only consider predictions with confidence > 50%
            x_min, y_min, x_max, y_max = box

            # Draw bounding box around detected face
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Display confidence score
            confidence_text = f"Confidence: {score*100:.2f}%"
            cv2.putText(frame, confidence_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Object Detection with Confidence', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
