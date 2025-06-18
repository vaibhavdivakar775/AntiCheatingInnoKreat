from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import base64
import io
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition

app = Flask(__name__)

exam_running = False
candidate_voice_path = "candidate.wav"  # Reference voice
recordings_dir = "recordings"
os.makedirs(recordings_dir, exist_ok=True)

# Load speaker verification model
verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def clean_and_filter_audio(raw_audio_data, fmt):
    audio_segment = AudioSegment.from_file(io.BytesIO(raw_audio_data), format=fmt)
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    audio_segment = audio_segment.set_sample_width(2)  # Force 16-bit PCM

    # Correct scaling for 16-bit PCM
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
    sr = audio_segment.frame_rate

    # Apply noise reduction (optional, but do not overdo)
    y = nr.reduce_noise(y=samples, sr=sr)
    return y, sr


def process_audio(audio_bytes, fmt):
    y, sr = clean_and_filter_audio(audio_bytes, fmt)

    min_samples = 2 * sr
    if len(y) < min_samples:
        print("Audio segment too short for reliable speaker verification. Skipping.")
        return

    temp_path = os.path.join(recordings_dir, f"live_{datetime.now().strftime('%H%M%S')}.wav")
    sf.write(temp_path, y, sr, subtype='PCM_16')

    score, prediction = verifier.verify_files(candidate_voice_path, temp_path)
    score_value = float(score)
    is_candidate = score_value > 0.16  # Updated threshold here

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_candidate:
        print(f"[{timestamp}] ❌ Candidate is speaking! Score: {score_value:.2f}")
        # Optional: Trigger alert/log here
    else:
        print(f"[{timestamp}] ✅ Not candidate. Score: {score_value:.2f}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", exam_running=exam_running)

@app.route("/start_exam", methods=["POST"])
def start_exam():
    global exam_running
    exam_running = True
    return render_template("index.html", exam_running=exam_running, duration=request.form.get("duration", 30))

@app.route("/end_exam", methods=["POST"])
def end_exam():
    global exam_running
    exam_running = False
    return render_template("index.html", exam_running=exam_running)

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if not exam_running:
        return "Exam not running", 403

    audio_b64 = request.json.get("audio")
    if not audio_b64:
        return "No audio", 400

    header, b64data = audio_b64.split(",", 1)
    if "webm" in header:
        fmt = "webm"
    elif "ogg" in header:
        fmt = "ogg"
    elif "wav" in header:
        fmt = "wav"
    else:
        return "Unsupported audio format", 415

    audio_bytes = base64.b64decode(b64data)
    process_audio(audio_bytes, fmt)

    return jsonify({"status": "received"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
