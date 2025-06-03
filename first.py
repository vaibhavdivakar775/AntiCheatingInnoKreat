import cv2
import numpy as np
import time
import json
from datetime import datetime
import mediapipe as mp

class ExamMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Tracking variables
        self.face_count_history = []
        self.no_face_start_time = None
        self.looking_away_start_time = None
        self.suspicious_activities = []
        
        # Thresholds
        self.MAX_NO_FACE_DURATION = 5  # seconds
        self.MAX_LOOKING_AWAY_DURATION = 3  # seconds
        self.MULTIPLE_FACE_THRESHOLD = 2
        
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        
        return faces
    
    def detect_hands(self, frame):
        """Detect hands in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
        return hand_count
    
    def is_looking_at_camera(self, face_bbox, frame_shape):
        """Simple heuristic to check if person is looking at camera"""
        x, y, w, h = face_bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Check if face is roughly centered
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Allow some deviation from center
        x_threshold = frame_w * 0.3
        y_threshold = frame_h * 0.3
        
        return (abs(face_center_x - frame_center_x) < x_threshold and 
                abs(face_center_y - frame_center_y) < y_threshold)
    
    def log_suspicious_activity(self, activity_type, details=""):
        """Log suspicious activities with timestamp"""
        timestamp = datetime.now().isoformat()
        activity = {
            "timestamp": timestamp,
            "type": activity_type,
            "details": details
        }
        self.suspicious_activities.append(activity)
        print(f"[{timestamp}] ALERT: {activity_type} - {details}")
    
    def analyze_frame(self, frame):
        """Analyze a single frame for suspicious activities"""
        current_time = time.time()
        alerts = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        face_count = len(faces)
        
        # Check for no face detected
        if face_count == 0:
            if self.no_face_start_time is None:
                self.no_face_start_time = current_time
            elif current_time - self.no_face_start_time > self.MAX_NO_FACE_DURATION:
                alerts.append("No face detected for extended period")
                self.log_suspicious_activity("NO_FACE_DETECTED", 
                    f"Duration: {current_time - self.no_face_start_time:.1f}s")
        else:
            self.no_face_start_time = None
        
        # Check for multiple faces
        if face_count > self.MULTIPLE_FACE_THRESHOLD:
            alerts.append(f"Multiple faces detected: {face_count}")
            self.log_suspicious_activity("MULTIPLE_FACES", f"Count: {face_count}")
        
        # Check if looking away (for single face)
        if face_count == 1:
            if not self.is_looking_at_camera(faces[0], frame.shape):
                if self.looking_away_start_time is None:
                    self.looking_away_start_time = current_time
                elif current_time - self.looking_away_start_time > self.MAX_LOOKING_AWAY_DURATION:
                    alerts.append("Looking away from camera")
                    self.log_suspicious_activity("LOOKING_AWAY", 
                        f"Duration: {current_time - self.looking_away_start_time:.1f}s")
            else:
                self.looking_away_start_time = None
        
        # Detect hands (unusual hand movements)
        hand_count = self.detect_hands(frame)
        if hand_count > 2:
            alerts.append(f"Unusual hand activity detected")
            self.log_suspicious_activity("UNUSUAL_HAND_ACTIVITY", f"Hands: {hand_count}")
        
        # Draw bounding boxes and alerts on frame
        for i, (x, y, w, h) in enumerate(faces):
            color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Face {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display alerts on frame
        for i, alert in enumerate(alerts):
            cv2.putText(frame, alert, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display statistics
        cv2.putText(frame, f"Faces: {face_count}", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Hands: {hand_count}", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: {len(self.suspicious_activities)}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, alerts
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("Starting exam monitoring... Press 'q' to quit, 's' to save report")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Analyze frame
            processed_frame, alerts = self.analyze_frame(frame)
            
            # Display frame
            cv2.imshow('Exam Monitor', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_report()
    
    def save_report(self):
        """Save monitoring report to file"""
        report = {
            "monitoring_session": {
                "start_time": datetime.now().isoformat(),
                "total_alerts": len(self.suspicious_activities),
                "suspicious_activities": self.suspicious_activities
            }
        }
        
        filename = f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Initialize monitor
    monitor = ExamMonitor()
    
    try:
        # Start monitoring
        monitor.run_monitoring()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        # Save final report
        monitor.save_report()
        print("Session ended")
