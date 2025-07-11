from flask import Flask, request, jsonify, render_template
import os
import uuid
import datetime
import base64
import cv2
import numpy as np
import io
#
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition
# 
from ultralytics import YOLO
import mediapipe as mp
import time

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')


### configs

last_logged_time = 0
LOG_INTERVAL = 5  # seconds

object_model = YOLO(r"C:\Users\Ojas\Downloads\ITPL_API\models\weights\best.pt")#-------------------change this according to location
person_model = YOLO('yolov8n.pt')

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

# speaker verification model
verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

CAPTURE_DIR = "captured_images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

SESSIONS = {}
exam_running = False
recordings_dir = "recordings"
os.makedirs(recordings_dir, exist_ok=True)
candidate_voice_path = os.path.join(recordings_dir, "candidate.wav")

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
        "type": "warning",
        "reason": "FACE NOT FOUND",
        "evidence_path": evidence         
    })
    elif num_people > 1:
     evidence = save_evidence(frame, "multi_person", exam_id)
     events.append({
        "timestamp": timestamp,
        "type": "warning",
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
            "type": "object detection",
            "message": f"{label} detected with {conf:.2f} confidence",
            "label": label,
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
                        "type": "warning",
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
                "type": "warning",
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
                        "type": "warning",
                        "reason": "SUSPICIOUS BLINKING DETECTED",
                        "evidence_path": save_evidence(face_crop, "blink", exam_id) 
                    })
                    total_blinks = 0
            blink_counter = 0
    return events

# Function to clean and preprocess audio
def clean_and_filter_audio(raw_audio_data, fmt):
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(raw_audio_data), format=fmt)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment = audio_segment.set_sample_width(2)
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = audio_segment.frame_rate
        y = nr.reduce_noise(y=samples, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Audio cleaning error: {e}")
        return None, None

# Voice verification
def process_audio(audio_bytes, fmt, exam_id=None):
    y, sr = clean_and_filter_audio(audio_bytes, fmt)
    if y is None:
        print("Skipping due to audio cleaning failure.")
        return

    if len(y) < 2 * sr:
        print("Audio too short for verification. Skipping.")
        return

    # Generate timestamp for filename and logs
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_log = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create subfolder for each exam
    if exam_id:
        exam_audio_dir = os.path.join(recordings_dir, exam_id)
        os.makedirs(exam_audio_dir, exist_ok=True)
        temp_path = os.path.join(exam_audio_dir, f"audio_{timestamp_str}.wav")
    else:
        temp_path = os.path.join(recordings_dir, f"audio_{timestamp_str}.wav")

    sf.write(temp_path, y, sr, subtype='PCM_16')

    if not os.path.exists(candidate_voice_path):
        print("Candidate voice not registered.")
        if exam_id and exam_id in SESSIONS:
            SESSIONS[exam_id]["events"].append({
                "timestamp": timestamp_log,
                "type": "violation",
                "message": "Candidate voice not registered. Skipping audio verification."
            })
        return
    
    score, prediction = verifier.verify_files(candidate_voice_path, temp_path)
    score_value = float(score)

    if exam_id and exam_id in SESSIONS:
        if score_value < 0.2:
            category = "Noise"
            msg = f"[AUDIO] Silence or background noise detected (Score: {score_value:.2f})"
        elif score_value < 0.6:
            category = "unknown Speaker"
            msg = f"[AUDIO] Voice does NOT match candidate (Score: {score_value:.2f})"
        else:
            category = "Candidate speaking"
            msg = f"[AUDIO] Candidate is speaking (Score: {score_value:.2f})"

    SESSIONS[exam_id]["events"].append({
        "timestamp": timestamp_log,
        "type": "voice_event",
        "category": category,
        "message": msg
    })

# summary

def get_summary(exam_id):
    if exam_id not in SESSIONS:
        return {"error": "Invalid exam ID"}

    summary = {
        "total_events": 0,
        "object_detections": {},
        "voice_events": {
            "candidate": 0,
            "unknown": 0,
            "noise": 0
        },
        "face_warnings": {
            "face_not_found": 0,
            "multiple_faces": 0
        },
        "eye_warnings": {
            "looking_away": 0,
            "suspicious_movement": 0,
            "blinking_violations": 0
        },
        "frontend_violations": {
            "tab_switches": 0,
            "fullscreen_exits": 0,
            "inactivity": 0,
            "extended_monitor": 0,
            "geolocation_error": 0
        }
    }

    for event in SESSIONS[exam_id]["events"]:
        summary["total_events"] += 1

        msg = event.get("message", "").lower()
        reason = event.get("reason", "").lower()
        category = event.get("category", "").lower()
        event_type = event.get("type", "").lower()
        label = event.get("label", "").lower() if "label" in event else ""

        # Object detections
        if event_type == "object detection":
            summary["object_detections"][label] = summary["object_detections"].get(label, 0) + 1

        # Voice events
        if event_type == "voice_event":
            if "candidate" in category:
                summary["voice_events"]["candidate"] += 1
            elif "unknown" in category:
                summary["voice_events"]["unknown"] += 1
            elif "noise" in category:
                summary["voice_events"]["noise"] += 1

        # Face warnings
        if reason == "face not found":
            summary["face_warnings"]["face_not_found"] += 1
        elif reason == "multiple persons in frame":
            summary["face_warnings"]["multiple_faces"] += 1

        # Eye & blinking
        elif reason == "looking_away_from_screen":
            summary["eye_warnings"]["looking_away"] += 1
        elif reason == "suspicious_eye_movement":
            summary["eye_warnings"]["suspicious_movement"] += 1
        elif reason == "suspicious blinking detected":
            summary["eye_warnings"]["blinking_violations"] += 1

        # Frontend event keywords
        if "tab or window switch" in msg:
            summary["frontend_violations"]["tab_switches"] += 1
        elif "fullscreen exited" in msg:
            summary["frontend_violations"]["fullscreen_exits"] += 1
        elif "no activity detected" in msg:
            summary["frontend_violations"]["inactivity"] += 1
        elif "extended monitor" in msg:
            summary["frontend_violations"]["extended_monitor"] += 1
        elif "geolocation error" in msg:
            summary["frontend_violations"]["geolocation_error"] += 1

    return summary


## API ROUTES

@app.route('/log_event', methods=['POST'])
def log_event():
    data = request.get_json()
    exam_id = data.get("exam_id")
    timestamp = data.get("timestamp")
    message = data.get("message")

    if not all([exam_id, timestamp, message]):
        return jsonify({ "error": "Missing data" }), 400

    if exam_id not in SESSIONS:
        return jsonify({ "error": "Invalid exam ID" }), 404

    event = {
        "timestamp": timestamp,
        "type": "violation",
        "message": message
    }
    SESSIONS[exam_id]["events"].append(event)
    return jsonify({ "status": "event logged" })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    global exam_running
    exam_running = True
    exam_id = str(uuid.uuid4())
    SESSIONS[exam_id] = {
        "start_time": datetime.datetime.now(),
        "events": []
    }
    return jsonify({"exam_id": exam_id})

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    data = request.get_json()
    exam_id = data.get('exam_id')
    image_data = data.get('image')
    timestamp = data.get('timestamp')

    if not all([exam_id, image_data]):
        return jsonify({"error": "Missing data"}), 400
    if exam_id not in SESSIONS:
        return jsonify({"error": "Invalid exam ID"}), 404

    try:
        header,encoded = image_data.split(',', 1)
        jpg_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    events = analyze_frame(frame, timestamp, exam_id=exam_id)
    SESSIONS[exam_id]["events"].extend(events)

    return jsonify({"status": "frame received"})

@app.route('/register_voice', methods=['POST'])
def register_voice():
    audio = request.files.get("audio")
    exam_id = request.form.get("exam_id")

    if not audio or not exam_id:
        return jsonify({"error": "Missing audio or exam_id"}), 400

    candidate_path = os.path.join(recordings_dir, "candidate_raw.webm")
    audio.save(candidate_path)

    with open(candidate_path, 'rb') as f:
        raw = f.read()
        y, sr = clean_and_filter_audio(raw, "webm")
        if y is None:
            return jsonify({"error": "Failed to process audio"}), 500
        sf.write(candidate_voice_path, y, sr, subtype='PCM_16')

    print("Candidate reference voice registered.")
    return jsonify({"status": "Voice registered successfully"})

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if not exam_running:
        return "Exam not running", 403

    data = request.get_json()
    audio_b64 = data.get("audio")
    exam_id = data.get("exam_id")
    if not audio_b64 or "," not in audio_b64:
     return "Invalid base64 audio", 400

    header, b64data = audio_b64.split(",", 1)
    print(f"Decoding format: {header}")
    fmt = None
    if "webm" in header:
        fmt = "webm"
    elif "ogg" in header:
        fmt = "ogg"
    elif "wav" in header:
        fmt = "wav"
    else:
        return "Unsupported audio format", 415

    audio_bytes = base64.b64decode(b64data)
    process_audio(audio_bytes, fmt,exam_id=exam_id)

    return jsonify({"status": "received"})

@app.route('/report')
def report_page():
    return render_template('report.html')

@app.route('/end_exam', methods=['POST'])
def end_exam():
    global exam_running
    exam_running = False
    data = request.get_json()
    exam_id = data.get("exam_id")
    if not exam_id or exam_id not in SESSIONS:
        return jsonify({"error": "Invalid exam ID"}), 400

    report = {
        "exam_id": exam_id,
        "start_time": SESSIONS[exam_id]["start_time"].isoformat(),
        "end_time": datetime.datetime.now().isoformat(),
        "events": SESSIONS[exam_id]["events"]
    }

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/{exam_id}.json", "w") as f:
        import json
        json.dump(report, f, indent=2)

    return jsonify(report)

@app.route("/summary/<exam_id>")
def get_exam_summary(exam_id):
    return jsonify(get_summary(exam_id))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
