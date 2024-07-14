import cv2
import numpy as np
import threading
import os
import json
from datetime import datetime
import logging
from deepface import DeepFace
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
people_data = {
    'count': 0,
    'genders': {'male': 0, 'female': 0},
    'ages': {},
    'emotions': {},
    'people': {}
}

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_list = ['male', 'female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def detect_people():
    global people_data
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)  # Limit to 1 FPS

    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        people_data['count'] = len(faces)
        people_data['genders'] = {'male': 0, 'female': 0}
        people_data['ages'] = {}
        people_data['emotions'] = {}
        people_data['people'] = {}

        for person_id, (x, y, w, h) in enumerate(faces):
            face_roi_color = frame[y:y+h, x:x+w]  # Extract color ROI for model input

            face_blob = cv2.dnn.blobFromImage(cv2.resize(face_roi_color, (227, 227)), 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

            # Gender detection
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Age detection
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Emotion detection
            result = DeepFace.analyze(face_roi_color, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']  # Access the first element of the result list

            people_data['genders'][gender] += 1
            people_data['ages'][age] = people_data['ages'].get(age, 0) + 1
            people_data['emotions'][emotion] = people_data['emotions'].get(emotion, 0) + 1
            people_data['people'][person_id] = {'gender': gender, 'age': age, 'emotion': emotion}

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"{gender}, {age}, {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if time.time() - start_time >= 1:
            print(json.dumps(people_data, indent=4))
            start_time = time.time()

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1)  # To maintain 1 FPS

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_people()
