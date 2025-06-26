from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
import uuid
import datetime
from detection.detection_pipeline import analyze_frame
from detection.voice_verification import verify_audio
import base64
import cv2
import numpy as np

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')

SESSIONS = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/report')
def report_page():
    return render_template('report.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    exam_id = str(uuid.uuid4())
    SESSIONS[exam_id] = {
        "start_time": datetime.datetime.now(),
        "events": []
    }
    return jsonify({ "exam_id": exam_id })

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
        header, encoded = image_data.split(',', 1)
        jpg_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    events = analyze_frame(frame, timestamp)
    for event in events:
        SESSIONS[exam_id]["events"].append(event)

    return jsonify({"status": "frame received"})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    print("📡 Received audio upload")
    audio = request.files.get("audio")
    exam_id = request.form.get("exam_id")
    timestamp = request.form.get("timestamp") or datetime.datetime.now().isoformat()
    print("📌 Exam ID:", exam_id)

    if not audio or not exam_id:
        print("no audio detected")
        return jsonify({"error": "Missing audio or exam_id"}), 400

    if exam_id not in SESSIONS:
        return jsonify({"error": "Invalid exam ID"}), 404

    fmt = audio.filename.split('.')[-1]
    print("📦 Audio Format:", fmt)

    audio_bytes = audio.read()
    result = verify_audio(audio_bytes, fmt, timestamp, exam_id)

    if result:
        SESSIONS[exam_id]["events"].append(result)
        return jsonify(result), 200

    return jsonify({"status": "Voice verified successfully"})

@app.route('/register_voice', methods=['POST'])
def register_voice():
    audio = request.files.get("audio")
    exam_id = request.form.get("exam_id")

    print("📥 Received register_voice request")
    print("🎤 Audio present:", audio is not None)
    print("🆔 Exam ID:", exam_id)

    if not audio or not exam_id:
        return jsonify({ "error": "Missing audio or exam_id" }), 400

    folder = os.path.join("recordings", exam_id)
    os.makedirs(folder, exist_ok=True)

    audio_path = os.path.join(folder, "candidate.wav")
    print("📂 Saving to:", audio_path)
    audio.save(audio_path)

    return jsonify({ "status": "Voice registered successfully" })

@app.route('/log_event', methods=['POST'])
def log_event():
    data = request.get_json()
    exam_id = data.get("exam_id")
    timestamp = data.get("timestamp")
    message = data.get("message")

    if not all([exam_id, message, timestamp]):
        return jsonify({"error": "Missing data"}), 400

    if exam_id not in SESSIONS:
        return jsonify({"error": "Invalid exam ID"}), 404

    log_entry = {
        "timestamp": timestamp,
        "message": message
    }
    SESSIONS[exam_id]["events"].append(log_entry)
    return jsonify({"status": "event logged"})

@app.route('/log_ip', methods=['POST'])
def log_ip():
    data = request.get_json()
    exam_id = data.get("exam_id")
    ip = data.get("ip")

    if not all([exam_id, ip]):
        return jsonify({"error": "Missing data"}), 400

    if exam_id not in SESSIONS:
        return jsonify({"error": "Invalid exam ID"}), 404

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "message": f"Public IP: {ip}"
    }
    SESSIONS[exam_id]["events"].append(log_entry)
    return jsonify({"status": "IP logged"})

@app.route('/end_exam', methods=['POST'])
def end_exam():
    data = request.get_json()
    exam_id = data.get("exam_id")
    if not exam_id or exam_id not in SESSIONS:
        return jsonify({ "error": "Invalid exam ID" }), 400

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

if __name__ == "__main__":
    app.run(debug=True)
