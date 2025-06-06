package main

import (
	"encoding/json"
	"fmt"
	"image"
	"log"
	"os"
	"time"

	"gocv.io/x/gocv"
)

type SuspiciousActivity struct {
	Timestamp string `json:"timestamp"`
	Type      string `json:"type"`
	Details   string `json:"details"`
}

type ExamMonitor struct {
	webcam            *gocv.VideoCapture
	window            *gocv.Window
	faceDetector      *gocv.CascadeClassifier
	faceDetection     *gocv.Net
	handDetector      *gocv.Net
	noFaceStartTime   *time.Time
	lookingAwayTime   *time.Time
	suspiciousActs    []SuspiciousActivity
	lastAlertTime     time.Time
	faceCountHistory  []int
}

const (
	maxNoFaceDuration      = 5 * time.Second
	maxLookingAwayDuration = 3 * time.Second
	multipleFaceThreshold  = 2
)

func NewExamMonitor() (*ExamMonitor, error) {
	// Open webcam
	webcam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		return nil, fmt.Errorf("error opening webcam: %v", err)
	}

	// Load face detection model (using Haar cascade as fallback)
	faceDetector := gocv.NewCascadeClassifier()
	if !faceDetector.Load("haarcascade_frontalface_default.xml") {
		log.Println("Warning: Could not load face detection model")
	}

	// Create window
	window := gocv.NewWindow("Exam Monitor")

	return &ExamMonitor{
		webcam:        webcam,
		window:        window,
		faceDetector:  &faceDetector,
		suspiciousActs: make([]SuspiciousActivity, 0),
	}, nil
}

func (em *ExamMonitor) DetectFaces(frame gocv.Mat) []image.Rectangle {
	// Convert to grayscale for Haar cascade
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(frame, &gray, gocv.ColorBGRToGray)

	// Detect faces
	faces := em.faceDetector.DetectMultiScale(gray)
	return faces
}

func (em *ExamMonitor) IsLookingAtCamera(face image.Rectangle, frame gocv.Mat) bool {
	frameWidth := frame.Cols()
	frameHeight := frame.Rows()

	faceCenterX := face.Min.X + face.Dx()/2
	faceCenterY := face.Min.Y + face.Dy()/2

	frameCenterX := frameWidth / 2
	frameCenterY := frameHeight / 2

	xThreshold := float64(frameWidth) * 0.3
	yThreshold := float64(frameHeight) * 0.3

	return (abs(faceCenterX-frameCenterX) < int(xThreshold) &&
		abs(faceCenterY-frameCenterY) < int(yThreshold))
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (em *ExamMonitor) LogSuspiciousActivity(activityType, details string) {
	activity := SuspiciousActivity{
		Timestamp: time.Now().Format(time.RFC3339),
		Type:      activityType,
		Details:   details,
	}
	em.suspiciousActs = append(em.suspiciousActs, activity)
	log.Printf("[%s] ALERT: %s - %s\n", activity.Timestamp, activityType, details)
}

func (em *ExamMonitor) AnalyzeFrame(frame gocv.Mat) (gocv.Mat, []string) {
	alerts := make([]string, 0)
	now := time.Now()

	// Detect faces
	faces := em.DetectFaces(frame)
	faceCount := len(faces)

	// Check for no face detected
	if faceCount == 0 {
		if em.noFaceStartTime == nil {
			em.noFaceStartTime = &now
		} else if now.Sub(*em.noFaceStartTime) > maxNoFaceDuration {
			alerts = append(alerts, "No face detected for extended period")
			em.LogSuspiciousActivity("NO_FACE_DETECTED", 
				fmt.Sprintf("Duration: %.1fs", now.Sub(*em.noFaceStartTime).Seconds()))
		}
	} else {
		em.noFaceStartTime = nil
	}

	// Check for multiple faces
	if faceCount > multipleFaceThreshold {
		alertMsg := fmt.Sprintf("Multiple faces detected: %d", faceCount)
		alerts = append(alerts, alertMsg)
		em.LogSuspiciousActivity("MULTIPLE_FACES", fmt.Sprintf("Count: %d", faceCount))
	}

	// Check if looking away (for single face)
	if faceCount == 1 {
		if !em.IsLookingAtCamera(faces[0], frame) {
			if em.lookingAwayTime == nil {
				em.lookingAwayTime = &now
			} else if now.Sub(*em.lookingAwayTime) > maxLookingAwayDuration {
				alerts = append(alerts, "Looking away from camera")
				em.LogSuspiciousActivity("LOOKING_AWAY", 
					fmt.Sprintf("Duration: %.1fs", now.Sub(*em.lookingAwayTime).Seconds()))
			}
		} else {
			em.lookingAwayTime = nil
		}
	}

	// Draw bounding boxes and alerts on frame
	for i, face := range faces {
		color := gocv.NewScalar(0, 255, 0, 0)
		if faceCount != 1 {
			color = gocv.NewScalar(0, 0, 255, 0)
		}
		gocv.Rectangle(&frame, face, color, 2)
		gocv.PutText(&frame, fmt.Sprintf("Face %d", i+1), image.Pt(face.Min.X, face.Min.Y-10), 
			gocv.FontHersheySimplex, 0.9, color, 2)
	}

	// Display alerts on frame
	for i, alert := range alerts {
		gocv.PutText(&frame, alert, image.Pt(10, 30+i*30), 
			gocv.FontHersheySimplex, 0.7, gocv.NewScalar(0, 0, 255, 0), 2)
	}

	// Display statistics
	gocv.PutText(&frame, fmt.Sprintf("Faces: %d", faceCount), image.Pt(10, frame.Rows()-60), 
		gocv.FontHersheySimplex, 0.6, gocv.NewScalar(255, 255, 255, 0), 2)
	gocv.PutText(&frame, fmt.Sprintf("Alerts: %d", len(em.suspiciousActs)), 
		image.Pt(10, frame.Rows()-20), gocv.FontHersheySimplex, 0.6, gocv.NewScalar(255, 255, 255, 0), 2)

	return frame, alerts
}

func (em *ExamMonitor) SaveReport() error {
	report := struct {
		MonitoringSession struct {
			StartTime         string              `json:"start_time"`
			TotalAlerts       int                 `json:"total_alerts"`
			SuspiciousActs    []SuspiciousActivity `json:"suspicious_activities"`
		} `json:"monitoring_session"`
	}{
		MonitoringSession: struct {
			StartTime         string              `json:"start_time"`
			TotalAlerts       int                 `json:"total_alerts"`
			SuspiciousActs    []SuspiciousActivity `json:"suspicious_activities"`
		}{
			StartTime:         time.Now().Format(time.RFC3339),
			TotalAlerts:       len(em.suspiciousActs),
			SuspiciousActs:    em.suspiciousActs,
		},
	}

	filename := fmt.Sprintf("exam_report_%s.json", time.Now().Format("20060102_150405"))
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating report file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		return fmt.Errorf("error encoding report: %v", err)
	}

	log.Printf("Report saved to %s\n", filename)
	return nil
}

func (em *ExamMonitor) RunMonitoring() error {
	log.Println("Starting exam monitoring... Press 'ESC' to quit, 's' to save report")

	for {
		frame := gocv.NewMat()
		if ok := em.webcam.Read(&frame); !ok {
			return fmt.Errorf("cannot read from webcam")
		}
		if frame.Empty() {
			continue
		}

		// Flip frame horizontally for mirror effect
		gocv.Flip(frame, &frame, 1)

		// Analyze frame
		processedFrame, _ := em.AnalyzeFrame(frame)

		// Display frame
		em.window.IMShow(processedFrame)

		// Handle key presses
		key := em.window.WaitKey(1)
		switch key {
		case 27: // ESC
			return nil
		case 's', 'S':
			if err := em.SaveReport(); err != nil {
				log.Printf("Error saving report: %v\n", err)
			}
		}

		frame.Close()
		processedFrame.Close()
	}
}

func (em *ExamMonitor) Close() {
	if em.webcam != nil {
		em.webcam.Close()
	}
	if em.window != nil {
		em.window.Close()
	}
	if em.faceDetector != nil {
		em.faceDetector.Close()
	}
}

func main() {
	// Initialize monitor
	monitor, err := NewExamMonitor()
	if err != nil {
		log.Fatalf("Error initializing monitor: %v", err)
	}
	defer monitor.Close()

	// Start monitoring
	if err := monitor.RunMonitoring(); err != nil {
		log.Printf("Monitoring error: %v", err)
	}

	// Save final report
	if err := monitor.SaveReport(); err != nil {
		log.Printf("Error saving final report: %v", err)
	}
	log.Println("Session ended")
}