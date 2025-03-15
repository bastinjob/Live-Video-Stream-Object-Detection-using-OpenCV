import cv2
from fer import FER

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load pre-trained gender classification model
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
gender_list = ['Male', 'Female']

# Initialize emotion detector
detector = FER()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Emotion Detection
        emotion, score = detector.top_emotion(face)
        cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gender Classification
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104, 177, 123), swapRB=False)
        gender_net.setInput(blob)
        gender_pred = gender_net.forward()
        gender = gender_list[gender_pred.argmax()]
        cv2.putText(frame, f"{gender}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Eye Detection
        face_region = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Face Detection, Emotion, Gender, and Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
