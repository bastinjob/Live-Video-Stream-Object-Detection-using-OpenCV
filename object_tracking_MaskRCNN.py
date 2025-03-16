import cv2
import torch
import numpy as np
from torchvision import models, transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
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

# Load pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
model.to(device)
model.eval()

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Define image transformation
transform = transforms.Compose([transforms.ToTensor()])

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
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

    # Perform object detection & segmentation
    with torch.no_grad():
        prediction = model(image_tensor)

    # Move predictions to CPU for processing
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()

    detections = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score > 0.5:  # Confidence threshold
            x_min, y_min, x_max, y_max = map(int, box)

            # Extract mask and threshold it
            mask = mask[0]  # Extract the single channel
            mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

            # Resize mask to match bounding box size
            mask_resized = cv2.resize(mask, (x_max - x_min, y_max - y_min))
            mask_inv = cv2.bitwise_not(mask_resized)

            # Apply mask to frame
            roi = frame[y_min:y_max, x_min:x_max]
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask_resized)
            frame[y_min:y_max, x_min:x_max] = roi_masked

            # Prepare tracking input
            detections.append(([x_min, y_min, x_max, y_max], score, label))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks

        track_id = int(track.track_id)  # Ensure it's an integer
        x_min, y_min, x_max, y_max = map(int, track.to_ltrb())

        # Generate unique color for track ID
        color = (track_id * 53 % 255, track_id * 101 % 255, track_id * 173 % 255)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Get label name
        label_name = COCO_CLASSES[track.det_class] if track.det_class < len(COCO_CLASSES) else "Unknown"

        # Display label & confidence score
        text = f"ID {track_id}: {label_name}"
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow('Real-Time Object Segmentation & Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
