import cv2
import time
import datetime
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import platform
from collections import deque

# Constants
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CENTER = [362, 263]
RIGHT_EYE_CENTER = [33, 133]
EAR_THRESHOLD = 0.28
EAR_CONSEC_FRAMES = 2
SUSPICIOUS_BLINK_RATE = 25
EYE_MOVEMENT_THRESHOLD = 0.02
MAX_EYE_MOVEMENTS = 20
ALERT_DURATION = 3

class ExamProctor:
    def __init__(self, duration_hours=3, output_video="output.mp4"):
        self.model = YOLO('yolov8n.pt')
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.duration = duration_hours * 3600
        self.output_video = output_video
        self.start_time = None
        self.out = None
        self.last_eye_center = None
        self.eye_movement_count = 0
        self.violations = []
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_timestamps = deque()
        self.current_alert = None
        self.alert_time = 0
        self.alert_count = 0
        self.total_faces = 0
        self.total_hands = 0

    def compute_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, landmarks):
        left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_LANDMARKS])
        right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_LANDMARKS])
        ear = (self.compute_ear(left_eye) + self.compute_ear(right_eye)) / 2.0

        if ear < EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= EAR_CONSEC_FRAMES:
                self.total_blinks += 1
                self.blink_timestamps.append(time.time())
            self.blink_counter = 0

        now = time.time()
        while self.blink_timestamps and now - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()

        if len(self.blink_timestamps) > SUSPICIOUS_BLINK_RATE:
            self.log_violation("SUSPICIOUS_BLINKING")
            self.blink_timestamps.clear()

    def get_eye_center(self, landmarks, indices):
        return np.mean([(landmarks[i].x, landmarks[i].y) for i in indices], axis=0)

    def detect_eye_movement(self, left, right):
        center = np.mean([left, right], axis=0)
        if self.last_eye_center is not None:
            dist = np.linalg.norm(center - self.last_eye_center)
            if dist > EYE_MOVEMENT_THRESHOLD:
                self.eye_movement_count += 1
                if self.eye_movement_count > MAX_EYE_MOVEMENTS:
                    self.log_violation("SUSPICIOUS_EYE_MOVEMENT")
                    self.eye_movement_count = 0
            else:
                self.eye_movement_count = max(0, self.eye_movement_count - 1)
        self.last_eye_center = center

    def log_violation(self, violation_type):
        now = datetime.datetime.now().isoformat()
        self.violations.append({'time': now, 'type': violation_type})
        self.alert_count += 1
        print(f"[{violation_type}] Detected at {now}")
        self.beep()
        self.current_alert = f"⚠️ {violation_type.replace('_', ' ')}"
        self.alert_time = time.time()

    def beep(self):
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(1000, 200)
        else:
            print('\a')

    def overlay_alert(self, frame):
        if self.current_alert and (time.time() - self.alert_time < ALERT_DURATION):
            cv2.putText(frame, self.current_alert, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            self.current_alert = None

    def draw_dashboard(self, frame, ear):
        panel_x, panel_y = 20, 120
        spacing = 35
        cv2.rectangle(frame, (10, 100), (300, 300), (0, 0, 0), -1)
        cv2.putText(frame, f"Faces: {self.total_faces}", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Hands: {self.total_hands}", (panel_x, panel_y + spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (panel_x, panel_y + 2 * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: {self.alert_count}", (panel_x, panel_y + 3 * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (panel_x, panel_y + 4 * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def save_report(self, json_path="violations.json"):
        with open(json_path, 'w') as f:
            json.dump(self.violations, f, indent=4)
        print(f"\n--- Session Ended ---\nViolations saved to: {json_path}")
        print(f"Video saved to: {self.output_video}")
        print(f"Total Violations: {len(self.violations)}")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        self.out = cv2.VideoWriter(
            self.output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        self.start_time = time.time()
        print("🟢 Exam proctoring started...")

        while time.time() - self.start_time < self.duration:
            ret, frame = cap.read()
            if not ret:
                break

            self.out.write(frame)
            results = self.model(frame, classes=[0])
            boxes = results[0].boxes
            self.total_hands = 0

            if boxes is not None:
                for box in boxes:
                    if len(boxes) > 1:
                        self.log_violation("MULTIPLE_PERSONS")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (y2 - y1) > frame.shape[0] // 2:
                        self.total_hands += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.face_mesh.process(rgb)

            ear = 0
            if not face.multi_face_landmarks:
                self.total_faces = 0
                self.log_violation("FACE_NOT_FOUND")
            else:
                self.total_faces = len(face.multi_face_landmarks)
                landmarks = face.multi_face_landmarks[0].landmark
                left_eye_center = self.get_eye_center(landmarks, LEFT_EYE_CENTER)
                right_eye_center = self.get_eye_center(landmarks, RIGHT_EYE_CENTER)
                self.detect_eye_movement(np.array(left_eye_center), np.array(right_eye_center))
                self.detect_blink(landmarks)
                left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_LANDMARKS])
                right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_LANDMARKS])
                ear = (self.compute_ear(left_eye) + self.compute_ear(right_eye)) / 2.0

            self.overlay_alert(frame)
            self.draw_dashboard(frame, ear)

            cv2.imshow("Exam Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Terminated by user.")
                break

        cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        self.save_report()

if __name__ == "__main__":
    proctor = ExamProctor(duration_hours=3)
    proctor.run()
