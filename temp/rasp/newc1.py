import cv2
import numpy as np
import threading
import os
import json
from datetime import datetime
import logging
import time
from collections import deque
import dlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
people_data = {
    'count': 0,
    'genders': {'male': 0, 'female': 0},
    'ages': {},
    'people': {}
}

historical_data = deque(maxlen=3600)  # Store data for the last hour (3600 seconds)

# Load models
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

gender_list = ['male', 'female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Initialize face tracker
tracker = dlib.correlation_tracker()
tracking = False

def preprocess_frame(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    return blob

def detect_faces(frame, face_net):
    blob = preprocess_frame(frame)
    face_net.setInput(blob)
    detections = face_net.forward()
    return detections

def get_face_bbox(detection, frame_shape, confidence_threshold=0.5):
    confidence = detection[2]
    if confidence < confidence_threshold:
        return None
    
    box = detection[3:7] * np.array([frame_shape[1], frame_shape[0], frame_shape[1], frame_shape[0]])
    return box.astype("int")

def analyze_face(face_roi, gender_net, age_net):
    blob = cv2.dnn.blobFromImage(cv2.resize(face_roi, (227, 227)), 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return gender, age

def capture_video(frame_queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            break
        frame_queue.append(frame)
        time.sleep(1/30)  # Maintain 30 FPS

    cap.release()

def process_frame(frame, face_net, gender_net, age_net):
    global people_data, tracking, tracker

    if not tracking:
        detections = detect_faces(frame, face_net)
        if detections.shape[2] > 0:
            bbox = get_face_bbox(detections[0, 0, 0, :], frame.shape)
            if bbox is not None:
                tracker.start_track(frame, dlib.rectangle(*bbox))
                tracking = True
    else:
        tracking_score = tracker.update(frame)
        if tracking_score > 7.0:
            bbox = tracker.get_position()
            bbox = (int(bbox.left()), int(bbox.top()), int(bbox.right()), int(bbox.bottom()))
        else:
            tracking = False
            return

    x, y, x2, y2 = bbox
    face_roi = frame[y:y2, x:x2]

    gender, age = analyze_face(face_roi, gender_net, age_net)

    people_data['count'] = 1
    people_data['genders'][gender] += 1
    people_data['ages'][age] = people_data['ages'].get(age, 0) + 1
    people_data['people'][0] = {'gender': gender, 'age': age}

    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{gender}, {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def analyze_trends():
    while True:
        if len(historical_data) > 0:
            avg_count = sum(d['count'] for d in historical_data) / len(historical_data)
            gender_ratio = sum(d['genders']['male'] for d in historical_data) / sum(d['genders']['female'] for d in historical_data) if sum(d['genders']['female']) > 0 else 0
            most_common_age = max(set(a for d in historical_data for a in d['ages']), key=lambda x: sum(d['ages'].get(x, 0) for d in historical_data))

            logger.info(f"Trend Analysis:")
            logger.info(f"Average people count: {avg_count:.2f}")
            logger.info(f"Male to Female ratio: {gender_ratio:.2f}")
            logger.info(f"Most common age group: {most_common_age}")

        time.sleep(60)  # Analyze trends every minute

def main():
    frame_queue = deque(maxlen=5)
    
    video_thread = threading.Thread(target=capture_video, args=(frame_queue,))
    video_thread.start()

    trend_thread = threading.Thread(target=analyze_trends)
    trend_thread.start()

    start_time = time.time()
    frame_count = 0

    while True:
        if len(frame_queue) > 0:
            frame = frame_queue.pop()
            frame_count += 1

            if frame_count % 30 == 0:  # Process every 30th frame (1 fps)
                processed_frame = process_frame(frame, face_net, gender_net, age_net)
                
                if processed_frame is not None:
                    cv2.imshow('Frame', processed_frame)

                    if time.time() - start_time >= 1:
                        logger.info(json.dumps(people_data, indent=4))
                        historical_data.append(people_data.copy())
                        start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    video_thread.join()
    trend_thread.join()

if __name__ == '__main__':
    main()