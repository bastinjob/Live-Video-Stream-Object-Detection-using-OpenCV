import cv2
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image

# Load COCO class labels
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained Faster R-CNN model and move to device
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Define a transform to convert frame to a tensor
transform = transforms.Compose([transforms.ToTensor()])

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could Not Open Webcam")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transform and move to device
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform object detection
    with torch.no_grad():
        prediction = model(image_tensor)

    # Move predictions to CPU for processing
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Loop through predictions
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Confidence threshold
            x_min, y_min, x_max, y_max = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Get label name
            label_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else "Unknown"

            # Display label & confidence score
            text = f"{label_name}: {score*100:.2f}%"
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
