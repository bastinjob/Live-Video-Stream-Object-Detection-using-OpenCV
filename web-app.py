from flask import Flask, Response
import cv2
import torch
from torchvision import models, transforms

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (required for processing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(frame_rgb).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            prediction = model(input_tensor)

        # Draw bounding boxes and confidence scores on the frame
        for element in range(len(prediction[0]['boxes'])):
            box = prediction[0]['boxes'][element].cpu().numpy()
            score = prediction[0]['scores'][element].cpu().numpy()
            if score > 0.5:  # Filter out low-confidence predictions
                x1, y1, x2, y2 = box
                label = f"Face: {score*100:.2f}%"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
