import cv2
import numpy as np
import uuid
import os
from ultralytics import YOLO
import mediapipe as mp
import time

last_logged_time = 0
LOG_INTERVAL = 5  # seconds

# YOLO Models
object_model = YOLO("/Users/rahul/work/AntiCheatingInnoKreat/VERSION_1/models/weights/best.pt")#-------------------change this according to location
person_model = YOLO('yolov8n.pt')

# MediaPipe
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Constants
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CENTER = [362, 263]
RIGHT_EYE_CENTER = [33, 133]
EAR_THRESHOLD = 0.28
EAR_CONSEC_FRAMES = 2
EYE_MOVEMENT_THRESHOLD = 0.02
MAX_EYE_MOVEMENTS = 20

# Persistent session vars
blink_counter = 0
total_blinks = 0
last_eye_center = None
eye_movement_count = 0

# Object detection 
custom_thresholds = {
    'Book': 0.2,
    'Earphone': 0.4,
    'Mobile_phone': 0.5,
    'cap': 0.85,
    'headset': 0.8,
    'smart_watch': 0.45,
    'sunglasses': 1.0
}
IGNORE_CLASSES = {"sunglasses", "cap"}

CAPTURE_DIR = "captured_images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

def save_evidence(img, prefix="evidence", exam_id=None):
    if not exam_id:
        exam_id = "unknown_exam"  # fallback
    
    exam_dir = os.path.join(CAPTURE_DIR, exam_id)
    os.makedirs(exam_dir, exist_ok=True)

    fname = f"{prefix}_{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(exam_dir, fname)
    cv2.imwrite(fpath, img)
    return fpath

def face_crop_from_landmarks(lms, frame):
    xs = [int(p.x * frame.shape[1]) for p in lms]
    ys = [int(p.y * frame.shape[0]) for p in lms]
    x1, y1, x2, y2 = max(min(xs)-20,0), max(min(ys)-20,0), \
                     min(max(xs)+20, frame.shape[1]), \
                     min(max(ys)+20, frame.shape[0])
    return frame[y1:y2, x1:x2]

def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def get_eye_center(landmarks, indices):
    return np.mean([(landmarks[i].x, landmarks[i].y) for i in indices], axis=0)


def analyze_frame(frame, timestamp, exam_id=None):
    global blink_counter, total_blinks, last_eye_center, eye_movement_count,last_logged_time
    events = []
    current_time = time.time()
    if current_time - last_logged_time < LOG_INTERVAL:
        return []
    
    last_logged_time = current_time
    events = [] 

    #Person detection
    person_results = person_model.predict(source=frame, classes=[0], verbose=False)[0]
    num_people = len(person_results.boxes) if person_results.boxes else 0

    if num_people == 0:
     evidence = save_evidence(frame, "face_not_found", exam_id)
     events.append({
        "timestamp": timestamp,
        "type": "violation",
        "reason": "FACE NOT FOUND",
        "evidence_path": evidence         
    })
    elif num_people > 1:
     evidence = save_evidence(frame, "multi_person", exam_id)
     events.append({
        "timestamp": timestamp,
        "type": "violation",
        "reason": "MULTIPLE PERSONS IN FRAME",
        "evidence_path": evidence        
    })

    #Object detection
    object_results = object_model.predict(source=frame, conf=0.25, verbose=False)[0]
    for box in object_results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = object_model.names[class_id]

        if label in IGNORE_CLASSES:
            continue

        if conf < custom_thresholds.get(label, 0.3):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        filename = f"{label}_{uuid.uuid4().hex}.jpg"
        path = save_evidence(crop, prefix=label, exam_id=exam_id)
        cv2.imwrite(path, crop)

        events.append({
            "timestamp": timestamp,
            "type": "object_detection",
            "message": f"{label} detected with {conf:.2f} confidence",
            "label": label,
            "confidence": round(conf, 2),
            "evidence_path": path
        })

    #Face + Eye analysis
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = face_mesh.process(rgb)

    if face.multi_face_landmarks:
        landmarks = face.multi_face_landmarks[0].landmark
        face_crop   = face_crop_from_landmarks(landmarks, frame)        # Eye movement detection
        left_eye_center = get_eye_center(landmarks, LEFT_EYE_CENTER)
        right_eye_center = get_eye_center(landmarks, RIGHT_EYE_CENTER)
        center = np.mean([left_eye_center, right_eye_center], axis=0)

        if last_eye_center is not None:
            dist = np.linalg.norm(center - last_eye_center)
            if dist > EYE_MOVEMENT_THRESHOLD:
                eye_movement_count += 1
                if eye_movement_count > MAX_EYE_MOVEMENTS:
                    events.append({
                        "timestamp": timestamp,
                        "type": "violation",
                        "reason": "SUSPICIOUS_EYE_MOVEMENT",
                        "evidence_path": save_evidence(face_crop, "eye_move", exam_id)   # <<<
                     })
                    eye_movement_count = 0
            else:
                eye_movement_count = max(0, eye_movement_count - 1)

        last_eye_center = center
        # print(f"Eye center X: {center[0]:.2f}")
        if center[0] < 0.40 or center[0] > 0.60:
            events.append({
                "timestamp": timestamp,
                "type": "violation",
                "reason": "LOOKING_AWAY_FROM_SCREEN",
                "evidence_path": save_evidence(face_crop, "looking_away",exam_id)
            })

        # Blink detection
        left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_LANDMARKS])
        right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_LANDMARKS])
        ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                total_blinks += 1
                if total_blinks > 25:
                    events.append({
                        "timestamp": timestamp,
                        "type": "violation",
                        "reason": "SUSPICIOUS BLINKING DETECTED",
                        "evidence_path": save_evidence(face_crop, "blink", exam_id) 
                    })
                    total_blinks = 0
            blink_counter = 0
# gives continous ratio of eyes and ear ratio
        # events.append({
        #     "timestamp": timestamp,
        #     "type": "face_analysis",
        #     "eye_aspect_ratio": round(ear, 2),
        #     "blink_count": total_blinks,
        #     "eye_center": center.tolist()
        # })

    return events

