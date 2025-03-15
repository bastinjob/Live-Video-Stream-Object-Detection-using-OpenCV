import cv2
import cv2.data

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could Not Open webcam")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab Frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for(x,y,w,y) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (225,0,0), 2)

    cv.imshow('Real Time Face Detection: ', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()