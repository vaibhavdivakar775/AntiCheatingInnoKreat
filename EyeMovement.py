import cv2
import numpy as np
import time
import json
from datetime import datetime
import mediapipe as mp
import math
from collections import deque

class EyeMovementMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        self.LEFT_EYE_CENTER = [468, 469, 470, 471, 472]
        self.RIGHT_EYE_CENTER = [473, 474, 475, 476, 477]
        
        self.gaze_history = deque(maxlen=30)
        self.blink_history = deque(maxlen=100)
        self.suspicious_activities = []
        
        self.is_calibrated = False
        self.calibration_points = []
        self.center_gaze_baseline = None
        
        self.RAPID_MOVEMENT_THRESHOLD = 0.1
        self.PROLONGED_LOOK_AWAY_DURATION = 3
        self.EXCESSIVE_BLINK_RATE = 0.5
        self.SUSPICIOUS_GAZE_PATTERN_COUNT = 5
        
        self.last_gaze_time = time.time()
        self.look_away_start_time = None
        self.rapid_movement_count = 0
        self.last_rapid_movement_time = 0
        
    def get_eye_landmarks(self, landmarks, eye_indices):
        eye_points = []
        for idx in eye_indices:
            point = landmarks[idx]
            eye_points.append([point.x, point.y])
        return np.array(eye_points)
    
    def calculate_eye_center(self, eye_landmarks):
        return np.mean(eye_landmarks, axis=0)
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def estimate_gaze_direction(self, left_eye_center, right_eye_center, face_landmarks):
        eye_center = (left_eye_center + right_eye_center) / 2
        
        nose_tip = np.array([face_landmarks[1].x, face_landmarks[1].y])
        
        gaze_vector = eye_center - nose_tip
        
        gaze_magnitude = np.linalg.norm(gaze_vector)
        if gaze_magnitude > 0:
            gaze_direction = gaze_vector / gaze_magnitude
        else:
            gaze_direction = np.array([0, 0])
            
        return gaze_direction, eye_center
    
    def detect_rapid_eye_movement(self, current_gaze):
        if len(self.gaze_history) < 2:
            return False
            
        previous_gaze = self.gaze_history[-1]
        movement_magnitude = np.linalg.norm(current_gaze - previous_gaze)
        
        return movement_magnitude > self.RAPID_MOVEMENT_THRESHOLD
    
    def analyze_blink_patterns(self, left_ear, right_ear):
        avg_ear = (left_ear + right_ear) / 2
        is_blinking = avg_ear < 0.25
        
        self.blink_history.append({
            'timestamp': time.time(),
            'is_blinking': is_blinking,
            'ear': avg_ear
        })
        
        current_time = time.time()
        recent_blinks = [b for b in self.blink_history 
                        if current_time - b['timestamp'] < 10]
        
        if len(recent_blinks) > 0:
            blink_count = sum(1 for b in recent_blinks if b['is_blinking'])
            blink_rate = blink_count / 10
            
            return blink_rate > self.EXCESSIVE_BLINK_RATE
        
        return False
    
    def is_looking_away(self, gaze_direction):
        if not self.is_calibrated:
            return np.linalg.norm(gaze_direction) > 0.3
        
        if self.center_gaze_baseline is not None:
            deviation = np.linalg.norm(gaze_direction - self.center_gaze_baseline)
            return deviation > 0.2
        
        return False
    
    def log_suspicious_activity(self, activity_type, details=""):
        timestamp = datetime.now().isoformat()
        activity = {
            "timestamp": timestamp,
            "type": activity_type,
            "details": details,
            "gaze_data": list(self.gaze_history) if len(self.gaze_history) > 0 else []
        }
        self.suspicious_activities.append(activity)
        print(f"[{timestamp}] EYE ALERT: {activity_type} - {details}")
    
    def calibrate_gaze(self, gaze_direction):
        self.calibration_points.append(gaze_direction.copy())
        
        if len(self.calibration_points) >= 30:
            self.center_gaze_baseline = np.mean(self.calibration_points, axis=0)
            self.is_calibrated = True
            print("Gaze calibration completed!")
    
    def analyze_frame(self, frame):
        current_time = time.time()
        alerts = []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE_INDICES)
            right_eye = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE_INDICES)
            
            h, w = frame.shape[:2]
            left_eye_px = left_eye * [w, h]
            right_eye_px = right_eye * [w, h]
            
            left_eye_center = self.calculate_eye_center(left_eye)
            right_eye_center = self.calculate_eye_center(right_eye)
            
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            
            gaze_direction, eye_center = self.estimate_gaze_direction(
                left_eye_center, right_eye_center, face_landmarks
            )
            
            if not self.is_calibrated:
                self.calibrate_gaze(gaze_direction)
                cv2.putText(frame, f"Calibrating... Look at center ({len(self.calibration_points)}/30)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.gaze_history.append(gaze_direction)
                
                if self.detect_rapid_eye_movement(gaze_direction):
                    self.rapid_movement_count += 1
                    self.last_rapid_movement_time = current_time
                    
                    if self.rapid_movement_count >= self.SUSPICIOUS_GAZE_PATTERN_COUNT:
                        alerts.append("Rapid eye movements detected")
                        self.log_suspicious_activity("RAPID_EYE_MOVEMENT", 
                            f"Count: {self.rapid_movement_count}")
                        self.rapid_movement_count = 0
                
                if current_time - self.last_rapid_movement_time > 2:
                    self.rapid_movement_count = 0
                
                if self.is_looking_away(gaze_direction):
                    if self.look_away_start_time is None:
                        self.look_away_start_time = current_time
                    elif current_time - self.look_away_start_time > self.PROLONGED_LOOK_AWAY_DURATION:
                        alerts.append("Prolonged looking away detected")
                        self.log_suspicious_activity("LOOKING_AWAY", 
                            f"Duration: {current_time - self.look_away_start_time:.1f}s")
                else:
                    self.look_away_start_time = None
                
                if self.analyze_blink_patterns(left_ear, right_ear):
                    alerts.append("Excessive blinking detected")
                    self.log_suspicious_activity("EXCESSIVE_BLINKING", 
                        f"Rate: {len([b for b in self.blink_history if b['is_blinking']])/10:.2f} blinks/sec")
            
            for landmark in left_eye_px:
                cv2.circle(frame, tuple(landmark.astype(int)), 1, (0, 255, 0), -1)
            for landmark in right_eye_px:
                cv2.circle(frame, tuple(landmark.astype(int)), 1, (0, 255, 0), -1)
            
            if self.is_calibrated:
                center_x, center_y = w // 2, h // 2
                gaze_end_x = int(center_x + gaze_direction[0] * 100)
                gaze_end_y = int(center_y + gaze_direction[1] * 100)
                cv2.arrowedLine(frame, (center_x, center_y), (gaze_end_x, gaze_end_y), 
                               (255, 0, 0), 3)
            
            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Gaze: ({gaze_direction[0]:.2f}, {gaze_direction[1]:.2f})", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, alert in enumerate(alerts):
            cv2.putText(frame, alert, (10, 60 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Eye Alerts: {len(self.suspicious_activities)}", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, alerts
    
    def run_monitoring(self):
        print("Starting eye movement monitoring...")
        print("Look at the center of the screen for calibration")
        print("Press 'q' to quit, 's' to save report, 'r' to recalibrate")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            processed_frame, alerts = self.analyze_frame(frame)
            
            cv2.imshow('Eye Movement Monitor', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_report()
            elif key == ord('r'):
                self.recalibrate()
    
    def recalibrate(self):
        self.is_calibrated = False
        self.calibration_points = []
        self.center_gaze_baseline = None
        print("Recalibrating... Look at the center of the screen")
    
    def save_report(self):
        report = {
            "eye_movement_session": {
                "start_time": datetime.now().isoformat(),
                "total_alerts": len(self.suspicious_activities),
                "calibrated": self.is_calibrated,
                "suspicious_activities": self.suspicious_activities,
                "session_stats": {
                    "total_gaze_points": len(self.gaze_history),
                    "total_blinks": len([b for b in self.blink_history if b['is_blinking']]),
                    "average_blink_rate": len([b for b in self.blink_history if b['is_blinking']]) / max(1, len(self.blink_history)) * 30
                }
            }
        }
        
        filename = f"eye_movement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Eye movement report saved to {filename}")
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    eye_monitor = EyeMovementMonitor()
    
    try:
        eye_monitor.run_monitoring()
    except KeyboardInterrupt:
        print("\nEye movement monitoring stopped by user")
    finally:
        eye_monitor.save_report()
        print("Eye tracking session ended") max(1, len(self.blink_history)) * 30
                
            
    
        
        filename = f"eye_movement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Eye movement report saved to {filename}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Initialize eye movement monitor
    eye_monitor = EyeMovementMonitor()
    
    try:
        # Start monitoring
        eye_monitor.run_monitoring()
    except KeyboardInterrupt:
        print("\nEye movement monitoring stopped by user")
    finally:
        # Save final report
        eye_monitor.save_report()
        print("Eye tracking session ended")
