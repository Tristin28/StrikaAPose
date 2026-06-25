import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---------- element references ----------
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const canvasCtx = overlay.getContext("2d");
const placeholder = document.getElementById("placeholder");
const startCameraBtn = document.getElementById("startCameraBtn");
const metricSelect = document.getElementById("metricSelect");
const skeletonPill = document.getElementById("skeletonPill");

const predictionLabel = document.getElementById("predictionLabel");
const metricOut = document.getElementById("metricOut");
const distanceOut = document.getElementById("distanceOut");
const simPct = document.getElementById("simPct");
const simBar = document.getElementById("simBar");
const lastUpdated = document.getElementById("lastUpdated");
const resultCard = document.getElementById("resultCard");

const ringFg = document.getElementById("ringFg");
const ringText = document.getElementById("ringText");

const qualityDot = document.getElementById("qualityDot");
const qualityText = document.getElementById("qualityText");
const landmarkCount = document.getElementById("landmarkCount");
const qualityWarning = document.getElementById("qualityWarning");

const userPoseImage = document.getElementById("userPoseImage");
const userPosePlaceholder = document.getElementById("userPosePlaceholder");
const matchedImage = document.getElementById("matchedImage");
const matchPlaceholder = document.getElementById("matchPlaceholder");
const matchCompareCard = document.getElementById("matchCompareCard");
const historyGrid = document.getElementById("historyGrid");

const dbgDistance = document.getElementById("dbgDistance");
const dbgMetric = document.getElementById("dbgMetric");
const dbgLatency = document.getElementById("dbgLatency");
const dbgPrediction = document.getElementById("dbgPrediction");

const snapshotCanvas = document.createElement("canvas"); // offscreen, used to snapshot the webcam frame

// ---------- state ----------
const UPDATE_MS = 5000;             // backend match interval
const RING_CIRC = 2 * Math.PI * 24; // circumference of the countdown ring (r = 24)

let poseLandmarker = null;
let drawingUtils = null;
let stream = null;
let cameraRunning = false;
let latestLandmarks = null;   // most recent detected pose (sent to backend every 5s)
let lastVideoTime = -1;       // so we only run detection on a new frame
let nextUpdateTime = 0;       // timestamp of the next backend call (drives the countdown ring)
let matchHistory = [];        // the last 3 matches, most recent first

// ---------- pipeline indicator ----------
function setPipeline(step) {
  document.querySelectorAll("#pipeline li").forEach(li => {
    li.classList.toggle("active", li.dataset.step === step);
  });
}

// 1. Load the MediaPipe pose model in the browser (same model the dataset was built with).
async function setupModel() {
  setPipeline("mediapipe");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/Model/pose_landmarker_full.task",
      delegate: "GPU"   // run on the GPU (WebGL) so the live loop stays smooth
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  drawingUtils = new DrawingUtils(canvasCtx);
  startCameraBtn.textContent = "Start Camera";
  startCameraBtn.disabled = false;
}

// 2. Start / pause / resume the webcam.
async function toggleCamera() {
  if (!poseLandmarker) return;

  if (!stream) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      placeholder.style.display = "none";
      cameraRunning = true;
      nextUpdateTime = Date.now() + UPDATE_MS;
      setCameraButton();
      setSkeletonPill();
      setPipeline("webcam");
      video.addEventListener("loadeddata", predictWebcam);
    } catch (error) {
      console.error("Camera access error:", error);
      qualityText.textContent = "Could not access the camera. Please allow camera permission.";
    }
    return;
  }

  // already started -> toggle pause / resume
  cameraRunning = !cameraRunning;
  if (cameraRunning) {
    video.play();
    nextUpdateTime = Date.now() + UPDATE_MS;
  } else {
    video.pause();
  }
  setCameraButton();
  setSkeletonPill();
}

function setCameraButton() {
  if (!stream) { startCameraBtn.textContent = "Start Camera"; return; }
  startCameraBtn.textContent = cameraRunning ? "Pause Camera" : "Resume Camera";
  startCameraBtn.classList.toggle("running", cameraRunning);
  startCameraBtn.classList.toggle("paused", !cameraRunning);
}

function setSkeletonPill() {
  skeletonPill.textContent = cameraRunning ? "Skeleton: live" : "Skeleton: paused";
  skeletonPill.classList.toggle("off", !cameraRunning);
}

// 3. Run pose detection on every animation frame so the skeleton moves in real time.
function predictWebcam() {
  if (cameraRunning && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // Resizing a canvas clears it, so only do it when the size actually changes.
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }

    const result = poseLandmarker.detectForVideo(video, performance.now());

    canvasCtx.clearRect(0, 0, overlay.width, overlay.height);
    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#5b7cfa", lineWidth: 4 });
      drawingUtils.drawLandmarks(landmarks, { color: "#ff3b6b", radius: 5 });
      latestLandmarks = landmarks;
      updateQuality(landmarks);
      setPipeline("landmarks");
    } else {
      latestLandmarks = null;
      updateQuality(null);
    }
  }
  window.requestAnimationFrame(predictWebcam);
}

// 7. Pose quality / visibility indicator.
function updateQuality(landmarks) {
  if (!landmarks) {
    qualityDot.className = "quality-dot bad";
    qualityText.textContent = "No pose detected";
    landmarkCount.textContent = "0/33 landmarks visible";
    qualityWarning.textContent = "Make sure you are in frame.";
    return;
  }

  const visible = landmarks.filter(lm => (lm.visibility ?? 0) >= 0.5).length;
  landmarkCount.textContent = `${visible}/33 landmarks visible`;

  if (visible >= 28) {
    qualityDot.className = "quality-dot good";
    qualityText.textContent = "Good full-body visibility";
    qualityWarning.textContent = "";
  } else if (visible >= 16) {
    qualityDot.className = "quality-dot warn";
    qualityText.textContent = "Partial body visible";
    qualityWarning.textContent = "Move further back for better full-body matching.";
  } else {
    qualityDot.className = "quality-dot bad";
    qualityText.textContent = "Poor pose detection";
    qualityWarning.textContent = "Move further back for better full-body matching.";
  }
}

// 4. Every 5 seconds send the latest pose to Flask and show the closest match.
async function sendLandmarksToBackend() {
  nextUpdateTime = Date.now() + UPDATE_MS; // reset the countdown each tick
  if (!cameraRunning || !latestLandmarks) return;

  // Only the 33 [x, y, z] landmarks are sent — never the image. (Privacy + matches the brief.)
  const landmarkArray = latestLandmarks.map(lm => [lm.x, lm.y, lm.z]);
  const userPoseDataUrl = captureUserPose(); // snapshot now so "Your Pose" matches what was sent

  setPipeline("flask");
  const t0 = performance.now();
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks: landmarkArray, metric: metricSelect.value })
    });
    const latency = Math.round(performance.now() - t0);
    const result = await response.json();
    updateResultPanel(result, userPoseDataUrl, latency);
    setPipeline("match");
  } catch (error) {
    console.error("Prediction request failed:", error);
    qualityWarning.textContent = "Could not reach the matching server.";
  }
}

// Draw the current webcam frame onto an offscreen canvas and return it as an image.
function captureUserPose() {
  snapshotCanvas.width = video.videoWidth;
  snapshotCanvas.height = video.videoHeight;
  snapshotCanvas.getContext("2d").drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
  return snapshotCanvas.toDataURL("image/png");
}

// Turn a raw distance into a 0–100% "similarity" feel (smaller distance = higher similarity).
function distanceToSimilarity(distance) {
  return Math.round((1 / (1 + distance)) * 100);
}

function updateResultPanel(result, userPoseDataUrl, latency) {
  if (result.error) {
    qualityWarning.textContent = result.error;
    return;
  }

  const best = result.best_match;
  const sim = distanceToSimilarity(best.distance);

  // Result card
  predictionLabel.textContent = result.prediction;
  metricOut.textContent = result.metric;
  distanceOut.textContent = best.distance.toFixed(2);
  simPct.textContent = sim + "%";
  simBar.style.width = sim + "%";
  lastUpdated.textContent = new Date().toLocaleTimeString();

  // Side-by-side comparison (stays until the next match)
  userPoseImage.src = userPoseDataUrl;
  userPoseImage.style.display = "block";
  userPosePlaceholder.style.display = "none";

  matchedImage.src = "/" + best.image; // best.image is e.g. "Images/Heart/Heart_001.jpg"
  matchedImage.style.display = "block";
  matchPlaceholder.style.display = "none";

  // Add this match to the front of the history and keep only the last 3.
  matchHistory.unshift({
    image: best.image,
    label: best.label,
    distance: best.distance,
    time: new Date().toLocaleTimeString()
  });
  matchHistory = matchHistory.slice(0, 3);
  renderHistory();

  // Debug panel
  dbgDistance.textContent = best.distance.toFixed(4);
  dbgMetric.textContent = result.metric;
  dbgLatency.textContent = latency + " ms";
  dbgPrediction.textContent = result.prediction;

  flashMatch();
}

// History of the last 3 matches, most recent first.
function renderHistory() {
  if (matchHistory.length === 0) return;
  historyGrid.innerHTML = matchHistory.map((m, i) => `
    <div class="history-card ${i === 0 ? "latest" : ""}">
      <span class="rank">${i === 0 ? "Latest" : i + 1}</span>
      <img src="/${m.image}" alt="${m.label}" />
      <div class="history-label">${m.label}</div>
      <div class="history-score">distance ${m.distance.toFixed(2)} · ${m.time}</div>
    </div>`).join("");
}

// 8 (demo polish). Short highlight flash when a new match arrives.
function flashMatch() {
  matchCompareCard.classList.remove("flash");
  void matchCompareCard.offsetWidth; // force reflow so the animation can replay
  matchCompareCard.classList.add("flash");
  resultCard.classList.remove("flash");
  void resultCard.offsetWidth;
  resultCard.classList.add("flash");
}

// Circular countdown ring, updated ~10x/sec.
function updateCountdown() {
  const remaining = Math.max(0, nextUpdateTime - Date.now());
  const fraction = remaining / UPDATE_MS; // 1 just after a call, 0 right before the next
  ringFg.style.strokeDashoffset = RING_CIRC * (1 - fraction);
  ringText.textContent = Math.ceil(remaining / 1000);
}

// ---------- wiring ----------
resultCard.classList.add("highlight-target");
startCameraBtn.addEventListener("click", toggleCamera);
setInterval(sendLandmarksToBackend, UPDATE_MS);
setInterval(updateCountdown, 100);
setupModel();
