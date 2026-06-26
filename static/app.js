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
const LANDMARK_VISIBILITY_THRESHOLD = 0.5;
const REQUIRED_LANDMARKS_FOR_MATCH = [
  { index: 11, name: "left shoulder" },
  { index: 12, name: "right shoulder" },
  { index: 13, name: "left elbow" },
  { index: 14, name: "right elbow" },
  { index: 15, name: "left wrist" },
  { index: 16, name: "right wrist" },
  { index: 23, name: "left hip" },
  { index: 24, name: "right hip" }
];
const SMOOTHING_WINDOW = 3;
const MIN_STABLE_MATCHES = 2;
const SIMILARITY_DISTANCE_LIMITS = {
  euclidean: 9.5,
  cosine: 0.07,
  manhattan: 75.0
};

let poseLandmarker = null;
let drawingUtils = null;
let stream = null;
let cameraRunning = false;
let latestLandmarks = null;   // most recent detected pose (sent to backend every 5s)
let lastVideoTime = -1;       // so we only run detection on a new frame
let nextUpdateTime = 0;       // timestamp of the next backend call (drives the countdown ring)
let matchHistory = [];        // the last 3 matches, most recent first
let predictionBuffer = [];    // recent labels used to avoid displaying one-frame matches

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
      resetPredictionSmoothing();
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
  resetPredictionSmoothing();
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

function countVisibleLandmarks(landmarks) {
  if (!landmarks) return 0;
  return landmarks.filter(lm => landmarkVisibility(lm) >= LANDMARK_VISIBILITY_THRESHOLD).length;
}

function landmarkVisibility(landmark) {
  return landmark?.visibility ?? 0;
}

function getMissingKeyLandmarks(landmarks) {
  if (!landmarks) return REQUIRED_LANDMARKS_FOR_MATCH.map(lm => lm.name);

  return REQUIRED_LANDMARKS_FOR_MATCH
    .filter(lm => landmarkVisibility(landmarks[lm.index]) < LANDMARK_VISIBILITY_THRESHOLD)
    .map(lm => lm.name);
}

function resetPredictionSmoothing() {
  predictionBuffer = [];
}

function countStablePredictions(result) {
  const matchKey = `${result.metric}:${result.prediction}`;
  predictionBuffer.push(matchKey);
  predictionBuffer = predictionBuffer.slice(-SMOOTHING_WINDOW);
  return predictionBuffer.filter(key => key === matchKey).length;
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

  const visible = countVisibleLandmarks(landmarks);
  const missingKeyLandmarks = getMissingKeyLandmarks(landmarks);
  landmarkCount.textContent = `${visible}/33 landmarks visible`;

  if (missingKeyLandmarks.length === 0) {
    qualityDot.className = "quality-dot good";
    qualityText.textContent = "Key pose points visible";
    qualityWarning.textContent = "";
  } else if (visible >= 1) {
    qualityDot.className = "quality-dot warn";
    qualityText.textContent = "Important pose points hidden";
    qualityWarning.textContent = "Keep shoulders, elbows, wrists, and hips visible.";
  } else {
    qualityDot.className = "quality-dot bad";
    qualityText.textContent = "No reliable pose points";
    qualityWarning.textContent = "Make sure you are in frame.";
  }
}

// 4. Every 5 seconds send the latest pose to Flask and show the closest match.
async function sendLandmarksToBackend() {
  nextUpdateTime = Date.now() + UPDATE_MS; // reset the countdown each tick
  if (!cameraRunning || !latestLandmarks) return;

  const missingKeyLandmarks = getMissingKeyLandmarks(latestLandmarks);
  if (missingKeyLandmarks.length > 0) {
    resetPredictionSmoothing();
    showNoMatchState("Keep shoulders, elbows, wrists, and hips visible.", {
      metric: metricSelect.value,
      prediction: "not sent"
    });
    qualityWarning.textContent = "Keep shoulders, elbows, wrists, and hips visible.";
    setPipeline("landmarks");
    return;
  }

  // Only landmark coordinates and visibility values are sent — never the image.
  const landmarkArray = latestLandmarks.map(lm => [lm.x, lm.y, lm.z]);
  const visibilityArray = latestLandmarks.map(lm => landmarkVisibility(lm));
  const userPoseDataUrl = captureUserPose(); // snapshot now so "Your Pose" matches what was sent

  setPipeline("flask");
  const t0 = performance.now();
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        landmarks: landmarkArray,
        visibility: visibilityArray,
        metric: metricSelect.value
      })
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

// Map each metric's distance scale to a readable confidence-style percentage.
// This is not model accuracy; it is a UI score where 100 means very close and
// 0 means the distance is at or beyond the no-match threshold for that metric.
function distanceToSimilarity(distance, metric) {
  const limit = SIMILARITY_DISTANCE_LIMITS[metric];
  if (!limit) return "—";

  const score = (1 - distance / limit) * 100;
  return Math.max(0, Math.min(100, Math.round(score)));
}

function showNoMatchState(message, debug = {}) {
  predictionLabel.textContent = "No confident match";
  metricOut.textContent = debug.metric || "—";
  distanceOut.textContent = debug.distance ?? "—";
  simPct.textContent = debug.similarity ?? "—";
  simBar.style.width = "0%";
  lastUpdated.textContent = "not sent";

  matchedImage.removeAttribute("src");
  matchedImage.style.display = "none";
  matchPlaceholder.textContent = message;
  matchPlaceholder.style.display = "block";

  userPoseImage.removeAttribute("src");
  userPoseImage.style.display = "none";
  userPosePlaceholder.textContent = "No pose sent yet.";
  userPosePlaceholder.style.display = "block";

  dbgDistance.textContent = debug.rawDistance || "—";
  dbgMetric.textContent = debug.metric || "—";
  dbgLatency.textContent = debug.latency || "—";
  dbgPrediction.textContent = debug.prediction || "no match";
}

function updateDisplayedMetric() {
  metricOut.textContent = metricSelect.value;
  dbgMetric.textContent = metricSelect.value;
}

function showPendingMatchState(result, userPoseDataUrl, latency, stableCount) {
  const best = result.best_match;
  const sim = distanceToSimilarity(best.distance, result.metric);

  predictionLabel.textContent = `Confirming ${result.prediction}`;
  metricOut.textContent = result.metric;
  distanceOut.textContent = best.distance.toFixed(2);
  simPct.textContent = sim === "—" ? "—" : sim + "%";
  simBar.style.width = sim === "—" ? "0%" : sim + "%";
  lastUpdated.textContent = new Date().toLocaleTimeString();

  userPoseImage.src = userPoseDataUrl;
  userPoseImage.style.display = "block";
  userPosePlaceholder.style.display = "none";

  matchedImage.removeAttribute("src");
  matchedImage.style.display = "none";
  matchPlaceholder.textContent = `Hold the pose (${stableCount}/${MIN_STABLE_MATCHES}).`;
  matchPlaceholder.style.display = "block";

  dbgDistance.textContent = best.distance.toFixed(4);
  dbgMetric.textContent = result.metric;
  dbgLatency.textContent = latency + " ms";
  dbgPrediction.textContent = `pending ${result.prediction}`;
  qualityWarning.textContent = "Hold the pose for one more check.";
}

function updateResultPanel(result, userPoseDataUrl, latency) {
  if (result.error) {
    resetPredictionSmoothing();
    showNoMatchState(result.error, {
      metric: result.metric || metricSelect.value,
      latency: latency + " ms",
      prediction: "error"
    });
    return;
  }

  const best = result.best_match;

  if (result.match_found === false) {
    resetPredictionSmoothing();
    const sim = best ? distanceToSimilarity(best.distance, result.metric) : "—";

    predictionLabel.textContent = "No confident match";
    metricOut.textContent = result.metric || metricSelect.value;
    distanceOut.textContent = best ? best.distance.toFixed(2) : "—";
    simPct.textContent = sim === "—" ? "—" : sim + "%";
    simBar.style.width = sim === "—" ? "0%" : sim + "%";
    lastUpdated.textContent = new Date().toLocaleTimeString();

    userPoseImage.src = userPoseDataUrl;
    userPoseImage.style.display = "block";
    userPosePlaceholder.style.display = "none";

    matchedImage.removeAttribute("src");
    matchedImage.style.display = "none";
    matchPlaceholder.textContent = result.message || "Make a clearer dataset pose.";
    matchPlaceholder.style.display = "block";

    dbgDistance.textContent = best ? best.distance.toFixed(4) : "—";
    dbgMetric.textContent = result.metric || metricSelect.value;
    dbgLatency.textContent = latency + " ms";
    dbgPrediction.textContent = result.distance_margin !== undefined
      ? `no match; margin ${result.distance_margin.toFixed(4)}`
      : result.message || "no match";
    qualityWarning.textContent = result.message || "Make a clearer dataset pose.";
    return;
  }

  if (!best) {
    resetPredictionSmoothing();
    showNoMatchState("No match returned by the backend.", {
      metric: result.metric || metricSelect.value,
      latency: latency + " ms",
      prediction: "no match"
    });
    return;
  }

  const stableCount = countStablePredictions(result);
  if (stableCount < MIN_STABLE_MATCHES) {
    showPendingMatchState(result, userPoseDataUrl, latency, stableCount);
    return;
  }

  const sim = distanceToSimilarity(best.distance, result.metric);

  // Result card
  predictionLabel.textContent = result.prediction;
  metricOut.textContent = result.metric;
  distanceOut.textContent = best.distance.toFixed(2);
  simPct.textContent = sim === "—" ? "—" : sim + "%";
  simBar.style.width = sim === "—" ? "0%" : sim + "%";
  lastUpdated.textContent = new Date().toLocaleTimeString();

  // Side-by-side comparison (stays until the next match)
  userPoseImage.src = userPoseDataUrl;
  userPoseImage.style.display = "block";
  userPosePlaceholder.style.display = "none";

  matchedImage.src = "/" + best.image;
  matchedImage.style.display = "block";
  matchPlaceholder.style.display = "none";

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
metricSelect.addEventListener("change", () => {
  resetPredictionSmoothing();
  updateDisplayedMetric();
});
updateDisplayedMetric();
setInterval(sendLandmarksToBackend, UPDATE_MS);
setInterval(updateCountdown, 100);
setupModel();
