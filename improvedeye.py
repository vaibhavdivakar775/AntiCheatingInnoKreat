import cv2
import time
import datetime
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import platform

class ExamProctor:
    def __init__(self, duration_hours=3, output_video="output.mp4"):
        self.model = YOLO('yolov8n.pt')
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.start_time = None
        self.duration = duration_hours * 3600
        self.last_eye_center = None
        self.eye_movement_count = 0
        self.violations = []

        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]

        self.out = None
        self.output_video = output_video

        self.current_alert = None
        self.alert_time = 0

    def get_eye_center(self, landmarks, eye_indices):
        coords = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
        return np.mean(coords, axis=0)

    def detect_eye_movement(self, left, right):
        center = np.mean([left, right], axis=0)
        if self.last_eye_center is not None:
            dist = np.linalg.norm(center - self.last_eye_center)
            if dist > 0.02:
                self.eye_movement_count += 1
                if self.eye_movement_count > 20:
                    self.log_violation("SUSPICIOUS_EYE_MOVEMENT")
            else:
                self.eye_movement_count = max(0, self.eye_movement_count - 1)
        self.last_eye_center = center

    def log_violation(self, violation_type):
        now = datetime.datetime.now().isoformat()
        self.violations.append({'time': now, 'type': violation_type})
        print(f"[{violation_type}] Detected at {now}")
        self.beep()

        # Set current alert message and timestamp
        self.current_alert = f"⚠️ {violation_type.replace('_', ' ')}"
        self.alert_time = time.time()

    def beep(self):
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(1000, 200)
        else:
            os.system('printf "\a"')

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not accessible")
            return

        self.start_time = time.time()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        self.out = cv2.VideoWriter(
            self.output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        while time.time() - self.start_time < self.duration:
            ret, frame = cap.read()
            if not ret:
                break

            self.out.write(frame)  # Save to video file

            results = self.model(frame, classes=[0])
            if len(results[0].boxes) > 1:
                self.log_violation("MULTIPLE_PERSONS")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.face_mesh.process(rgb)

            if not face.multi_face_landmarks:
                self.log_violation("FACE_NOT_FOUND")
            else:
                landmarks = face.multi_face_landmarks[0].landmark
                left_eye = self.get_eye_center(landmarks, self.LEFT_EYE)
                right_eye = self.get_eye_center(landmarks, self.RIGHT_EYE)
                self.detect_eye_movement(np.array(left_eye), np.array(right_eye))

            # Show alert overlay (if any)
            if self.current_alert and (time.time() - self.alert_time < 3):  # 3 seconds display
                cv2.putText(frame, self.current_alert, (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                self.current_alert = None  # Clear alert after 3 seconds

            cv2.imshow("Exam Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        self.save_report()

    def save_report(self, json_path="violations.json"):
        with open(json_path, 'w') as f:
            json.dump(self.violations, f, indent=4)
        print(f"\n--- Session Ended ---\nViolations saved to: {json_path}")
        print(f"Video saved to: {self.output_video}")
        print(f"Total Violations: {len(self.violations)}")

if __name__ == "__main__":
    print("🟢 Starting exam proctoring...")
    proctor = ExamProctor(duration_hours=3)
    proctor.run()
