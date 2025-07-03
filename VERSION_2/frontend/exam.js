
let video = document.getElementById("video");
let startBtn = document.getElementById("startExam");
let endBtn = document.getElementById("endExam");
let timerEl = document.getElementById("timer");
let reportEl = document.getElementById("report");
let registerBtn = document.getElementById("registerVoice");
let mediaStream = null;
let mediaRecorder = null;
let countdownInterval = null;
let examId = null;
let lastActivity = Date.now();

function logEvent(msg) {
    console.log(`[${new Date().toLocaleTimeString()}] ${msg}`);
    if (examId) {
        fetch('/log_event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exam_id: examId,
                timestamp: Date.now(),
                message: msg
            })
        });
    }
}
//inactivity
setInterval(() => {
    if (Date.now() - lastActivity > 30000) {
        logEvent("No activity detected for more than 30 seconds");
        lastActivity = Date.now();
    }
}, 5000);

// Activity tracker
["mousemove", "keydown", "click"].forEach(evt =>
    document.addEventListener(evt, () => lastActivity = Date.now())
);
// Tab/window switch detection
window.onblur = () => logEvent("Tab or window switch detected");
// window.onfocus = () => logEvent("User returned to tab");******************************************
//exit full screen
document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) {
        logEvent("Fullscreen exited during exam");
    }
});
//dual monitor
function checkExtendedMonitor() {
    if (window.screen && (window.screen.availWidth > window.screen.width || window.screen.availHeight > window.screen.height)) {
        logEvent("Possible extended monitor detected");
    }
}
//geolocation
function logGeolocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude, accuracy } = position.coords;
                logEvent(`🌍 Location: Lat ${latitude.toFixed(5)}, Lng ${longitude.toFixed(5)}, ±${accuracy}m`);
            },
            (error) => {
                logEvent(`Geolocation error: ${error.message}`);
            }
        );
    } else {
        logEvent("Geolocation not supported in this browser.");
    }
}

async function startExam() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  video.srcObject = mediaStream;

  const res = await fetch("/start_exam", { method: "POST" });
  const data = await res.json();
  examId = data.exam_id;

  sendVideoFrames();
  startAudioChunks();
  startTimer();
//screendet
  checkExtendedMonitor();
  document.documentElement.requestFullscreen();
  logGeolocation();
//
  startBtn.disabled = true;
  endBtn.disabled = false;
}

function startTimer() {
  let remainingTime = 600;
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
  }, 5000);
}
function startAudioChunks() {
  const audioStream = new MediaStream(mediaStream.getAudioTracks());
  mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });

  mediaRecorder.ondataavailable = async (e) => {
    if (e.data.size === 0) {
      console.warn("Empty audio chunk, skipping upload.");
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64Data = reader.result;

      if (!base64Data || !base64Data.includes("base64,")) {
        console.error("Invalid base64 audio format, not uploading.");
        return;
      }

      console.log("Uploading audio chunk to server...");

      fetch('/upload_audio', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exam_id: examId,
          timestamp: Date.now(),
          audio: base64Data
        })
      }).then(res => res.json()).then(data => {
        console.log("Audio uploaded:", data);
      }).catch(err => {
        console.error("Upload error:", err);
      });
    };

    reader.readAsDataURL(e.data);
  };

  const loopedAudioRecording = () => {
    if (mediaRecorder && mediaRecorder.state === "inactive") {
      mediaRecorder.start();
      console.log("Recording audio for 20s...");

      setTimeout(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
          console.log("Stopped. Waiting 10s...");
          setTimeout(loopedAudioRecording, 10000);
        }
      }, 20000);
    }
  };

  loopedAudioRecording();
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

registerBtn.addEventListener("click", async () => {
  if (!examId) {
    alert("Exam not started. Start the exam first.");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  let chunks = [];

  recorder.ondataavailable = e => chunks.push(e.data);

  recorder.onstop = async () => {
    const blob = new Blob(chunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append("audio", blob, "voice_sample.webm");
    formData.append("exam_id", examId);

    const res = await fetch('/register_voice', {
      method: "POST",
      body: formData
    });

    const result = await res.json();
    alert(result.status || result.error);
  };

  recorder.start();
  setTimeout(() => recorder.stop(), 5000);
});

function showReport(report) {
  let html = "<h2>Exam Report</h2><ul>";
  for (const event of report.events) {
    html += `<li>[${new Date(event.timestamp).toLocaleTimeString()}] ${event.message}</li>`;
  }
  html += "</ul>";
  reportEl.innerHTML = html;
}

startBtn.addEventListener("click", startExam);
endBtn.addEventListener("click", endExam);
