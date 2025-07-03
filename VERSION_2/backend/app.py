from flask import Flask, request, jsonify, render_template
import os
import uuid
import datetime
import base64
import cv2
import numpy as np
import io
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition
from detection.detection_pipeline import analyze_frame
from database import SessionLocal, init_db
from models import Report
import json

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')

SESSIONS = {}
exam_running = False
recordings_dir = "recordings"
os.makedirs(recordings_dir, exist_ok=True)
candidate_voice_path = os.path.join(recordings_dir, "candidate.wav")

# speaker verification model
verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

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
                "message": "Candidate voice not registered. Skipping audio verification."
            })
        return
    
    score, prediction = verifier.verify_files(candidate_voice_path, temp_path)
    score_value = float(score)

    # Log result 
    if exam_id and exam_id in SESSIONS:
        msg = f"Candidate is {'speaking' if prediction else 'not speaking'} (Score: {score_value:.2f})"
        SESSIONS[exam_id]["events"].append({
            "timestamp": timestamp_log,
            "message": msg
        })
        
    # if prediction:
    #     print(f"[{timestamp_log}] Candidate is speaking! Score: {score_value:.2f}")
    # else:
    #     print(f"[{timestamp_log}] Candidate is not speaking. Score: {score_value:.2f}")


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
        "type": "log_event",
        "message": message
    }
    SESSIONS[exam_id]["events"].append(event)
    return jsonify({ "status": "event logged" })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/report')
def report_page():
    return render_template('report.html')

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

    # Save report to JSON file
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/{exam_id}.json", "w") as f:
        json.dump(report, f, indent=2)

    # Store report in the database
    report_title = f"Exam Report - {report.get('student_name', 'Unknown')}"
    report_content = json.dumps(report)

    init_db()
    session = SessionLocal()
    new_report = Report(title=report_title, content=report_content)
    session.add(new_report)
    session.commit()
    session.close()

    print(f"✅ Report for exam {exam_id} saved to database.")

    return jsonify(report)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

