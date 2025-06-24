let video = document.getElementById("video");
let startBtn = document.getElementById("startExam");
let endBtn = document.getElementById("endExam");
let timerEl = document.getElementById("timer");
let reportEl = document.getElementById("report");

let mediaStream = null;
let mediaRecorder = null;
let examDuration = 300; // 5 minutes in seconds
let countdownInterval = null;
let examId = null;

async function startExam() {
  
  mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  video.srcObject = mediaStream;

  // 2. Notify backend to start exam
  const res = await fetch("/start_exam", { method: "POST" });
  const data = await res.json();
  examId = data.exam_id;

  // 3. Start sending video and audio
  sendVideoFrames();
  startAudioChunks();

  // 4. Start timer
  startTimer();

  startBtn.disabled = true;
  endBtn.disabled = false;
}

function startTimer() {
  let remainingTime = 600; // Reset every time you start

  countdownInterval = setInterval(() => {
    if (remainingTime <= 0) {
      clearInterval(countdownInterval);
      endExam();
    } else {
      remainingTime--;
      const mins = String(Math.floor(remainingTime / 60)).padStart(2, '0');
      const secs = String(remainingTime % 60).padStart(2, '0');
      timerEl.textContent = `Time Left: ${mins}:${secs}`;
    }
  }, 1000);
}

function sendVideoFrames() {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  setInterval(() => {
    if (!mediaStream) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const frameData = canvas.toDataURL("image/jpeg");

    fetch('/upload_frame', {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exam_id: examId,
        timestamp: Date.now(),
        image: frameData
      })
    });
  }, 5000); // Send frame every 5 seconds
}

function startAudioChunks() {
  let audioChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream);

  mediaRecorder.ondataavailable = async (e) => {
    audioChunks.push(e.data);
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    audioChunks = [];

    const formData = new FormData();
    formData.append("exam_id", examId);
    formData.append("timestamp", Date.now());
    formData.append("audio", blob);

    await fetch('/upload_audio', {
      method: 'POST',
      body: formData
    });
  };

  mediaRecorder.start(5000); // Send audio every 5 seconds
}

async function endExam() {
  const response = await fetch('/end_exam', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exam_id: examId })
  });

  const report = await response.json();
  localStorage.setItem("examReport", JSON.stringify(report));
  window.location.href = "/report";
}


function showReport(report) {
  let html = "<h2>Exam Report</h2><ul>";
  for (const event of report.events) {
    html += `<li>[${new Date(event.timestamp).toLocaleTimeString()}] ${event.message}</li>`;
  }
  html += "</ul>";
  reportEl.innerHTML = html;
}

// Event Listeners
startBtn.addEventListener("click", startExam);
endBtn.addEventListener("click", endExam);
