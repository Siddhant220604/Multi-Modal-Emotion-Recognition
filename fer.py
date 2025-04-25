import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model_path = r"C:\Users\siddj\OneDrive\Desktop\Facial Emotion Recognition - Image Classification\fer.h5"
model = load_model(model_path)

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image for model input
def extract_features(image):
    feature = np.array(image, dtype="float32")
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Extract face region
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize to 48x48

        img = extract_features(face_roi)  # Preprocess
        pred = model.predict(img)  # Make prediction
        prediction_label = labels[pred.argmax()]  # Get label

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(27) & 0xFF == ord('q'):  # Press 'q' to quit
        break

webcam.release()
cv2.destroyAllWindows()
