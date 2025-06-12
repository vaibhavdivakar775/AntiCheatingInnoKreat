from flask import Flask, render_template, request, session, redirect, url_for, Response
import subprocess
from objdetection import generate_frames  # Imported from objdet.py

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session tracking
process = None

# Path to your YOLO model
MODEL_PATH = r"C:/Users/Ojas/Downloads/DC++/college/2nd year/python/runs/detect/objdet_cpu/weights/best.pt"

@app.route('/')
def index():
    exam_running = session.get('exam_running', False)
    duration = session.get('duration', 5)  # Default to 5 minutes
    return render_template('index.html', exam_running=exam_running, duration=duration)

@app.route('/start_exam', methods=['POST'])
def start_exam():
    global process
    duration = int(request.form.get('duration', 5))  # Get duration from form
    session['duration'] = duration  # Store in session
    session['exam_running'] = True

    if process is None:
        print("Starting detection script...")
        process = subprocess.Popen([
            "python",
            r"C:/Users/Ojas/Downloads/DC++/college/2nd year/python/objdetection/objdetection.py"
        ])
        # this is the adress to the objet detection module
    return redirect(url_for('index'))

@app.route('/end_exam', methods=['POST'])
def end_exam():
    global process
    if process:
        print("Terminating detection script...")
        process.terminate()
        process = None

    session['exam_running'] = False
    session.pop('duration', None)
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(MODEL_PATH), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
