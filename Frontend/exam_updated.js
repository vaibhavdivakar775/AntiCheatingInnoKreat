let video = document.getElementById("video");
let startBtn = document.getElementById("startExam");
let endBtn = document.getElementById("endExam");
let timerEl = document.getElementById("timer");
let reportEl = document.getElementById("report");

let mediaStream = null;
let mediaRecorder = null;
let examId = null;
let countdownInterval = null;
let logEvents = [];
let lastActivity = Date.now();

// ------------------------- Logging -------------------------
function logEvent(message) {
  const entry = {
    timestamp: Date.now(),
    message: message
  };
  logEvents.push(entry);
  console.log(`[Log] ${message}`);

  // Live push to backend (optional)
  if (examId) {
    fetch('/log_event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ exam_id: examId, ...entry })
    });
  }
}

// ---------------------- Exam Start -------------------------
async function startExam() {
    checkExtendedDisplay();
  try {
    // Get webcam + mic
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    video.srcObject = mediaStream;

    // Notify backend
    const res = await fetch("/start_exam", { method: "POST" });
    const data = await res.json();
    examId = data.exam_id;

    logEvent("✅ Exam started");
    requestFullscreen();
    startTimer();
    sendVideoFrames();
    startAudioChunks();

    startBtn.disabled = true;
    endBtn.disabled = false;
  } catch (err) {
    console.error("Error starting exam:", err);
    logEvent("❌ Failed to start webcam or mic");
  }
  fetch('https://api.ipify.org?format=json')
  .then(res => res.json())
  .then(data => {
    const ip = data.ip;
    logEvent(`🌐 Public IP: ${ip}`);
    // Optionally log to server
    if (examId) {
      fetch('/log_ip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exam_id: examId, ip: ip })
      });
    }
  })
  .catch(() => logEvent("❌ Failed to fetch IP address"));

}

// ------------------------- Timer ---------------------------
function startTimer() {
  let remainingTime = 600; // 10 minutes

  countdownInterval = setInterval(() => {
    if (remainingTime <= 0) {
      clearInterval(countdownInterval);
      endExam();
    } else {
      remainingTime--;
      const mins = String(Math.floor(remainingTime / 60)).padStart(2, '0');
      const secs = String(remainingTime % 60).padStart(2, '0');
      timerEl.textContent = `⏱️ Time Left: ${mins}:${secs}`;
    }
  }, 1000);
}

// -------------------- Video Capture ------------------------
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
  }, 5000); // every 5 seconds
}

// --------------------- Audio Capture -----------------------
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

  mediaRecorder.start(5000); // 5 second chunks
}

// -------------------- End Exam ----------------------------
async function endExam() {
  clearInterval(countdownInterval);
  mediaRecorder?.stop();
  mediaStream?.getTracks().forEach(track => track.stop());

  // Send all logs
  await fetch('/log_event_batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exam_id: examId, events: logEvents })
  });

  // Notify backend
  const response = await fetch('/end_exam', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exam_id: examId })
  });

  const report = await response.json();
  localStorage.setItem("examReport", JSON.stringify(report));
  window.location.href = "/report";
}

// ------------------ Proctoring Events ----------------------

async function checkExtendedDisplay() {
  try {
    const screens = await window.getScreenDetails?.(); // Experimental API
    if (screens && screens.screens.length > 1) {
      logEvent("⚠️ Multiple monitors detected");
    } else {
      logEvent("✅ Only one display detected");
    }
  } catch (err) {
    // Fallback detection (less reliable)
    if (window.screenLeft !== 0 || window.screenTop !== 0) {
      logEvent("⚠️ Potential multi-monitor setup (screen offset detected)");
    } else {
      logEvent("✅ Display appears normal");
    }
  }
}

// ⏳ Inactivity detection
["mousemove", "keydown", "click"].forEach(evt =>
  document.addEventListener(evt, () => lastActivity = Date.now())
);

setInterval(() => {
  if (Date.now() - lastActivity > 30000) {
    logEvent("⚠️ No activity for 30 seconds");
    lastActivity = Date.now();
  }
}, 5000);

// 🪟 Tab switching
window.addEventListener("blur", () => logEvent("⚠️ Tab or window switch detected"));
window.addEventListener("focus", () => logEvent("✅ User returned to tab"));

// 📺 Fullscreen exit
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement) {
    logEvent("⚠️ Fullscreen exited");
  }
});

function requestFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen();
  }
}

// ----------------- Report Viewer (Optional) ----------------
function showReport(report) {
  let html = "<h2>Exam Report</h2><ul>";
  for (const event of report.events) {
    html += `<li>[${new Date(event.timestamp).toLocaleTimeString()}] ${event.message}</li>`;
  }
  html += "</ul>";
  reportEl.innerHTML = html;
}

// ------------------ Button Events -------------------------
startBtn.addEventListener("click", startExam);
endBtn.addEventListener("click", endExam);

// Optional: Load report if on report.html
if (window.location.pathname.includes("report.html")) {
  const saved = localStorage.getItem("examReport");
  if (saved) {
    const parsed = JSON.parse(saved);
    showReport(parsed);
  }
}
