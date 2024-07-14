import cv2
import numpy as np
import threading
import os
import json
from datetime import datetime
import logging
from deepface import DeepFace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
people_data = {
    'count': 0,
    'genders': {'male': 0, 'female': 0},
    'emotions': {},
    'people': {}
}

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
gender_list = ['male', 'female']

def detect_people():
    global people_data
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        people_data['count'] = len(faces)
        people_data['genders'] = {'male': 0, 'female': 0}
        people_data['emotions'] = {}
        people_data['people'] = {}

        for person_id, (x, y, w, h) in enumerate(faces):
            face_roi = rgb_frame[y:y+h, x:x+w]

            # Gender detection
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Emotion detection
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            people_data['genders'][gender] += 1
            people_data['emotions'][emotion] = people_data['emotions'].get(emotion, 0) + 1
            people_data['people'][person_id] = {'gender': gender, 'emotion': emotion}

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"{gender}, {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        print(json.dumps(people_data, indent=4))

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_people()
