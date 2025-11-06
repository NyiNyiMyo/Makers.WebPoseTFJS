import React from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as posedetection from "@tensorflow-models/pose-detection";
import "./styles.css";

// ✅ Initialize TF backend before model loads
(async () => {
  await tf.setBackend("webgl");
  await tf.ready();
  console.log("✅ TF.js backend initialized:", tf.getBackend());
})();

class App extends React.Component {
  videoRef = React.createRef();
  imageRef = React.createRef();
  canvasRef = React.createRef();
  fpsRef = React.createRef();
  fileInputRef = React.createRef();
  animationId = null;
  stream = null;

  state = {
    modelType: "movenet_lightning",
    detector: null,
    running: false,
    loadingModel: false,
    confThreshold: 0.2,
    imageMode: false,
  };

  componentWillUnmount() {
    this.stopDetection();
  }

  startWebcam = async () => {
    try {
      const getMedia =
        (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) ||
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia;

      if (!getMedia) {
        alert("❌ Your browser does not support camera access. Try Chrome or Safari.");
        return;
      }

      const constraints = { audio: false, video: { facingMode: "user" } };

      const stream = await new Promise((resolve, reject) => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          navigator.mediaDevices.getUserMedia(constraints).then(resolve).catch(reject);
        } else {
          getMedia.call(navigator, constraints, resolve, reject);
        }
      });

      this.stream = stream;
      const videoEl = this.videoRef.current;
      videoEl.srcObject = stream;

      await new Promise((resolve) => {
        videoEl.onloadedmetadata = () => resolve();
      });

      await videoEl.play().catch(() => {});
      console.log("✅ Webcam ready");
      // ✅ Ensure video has correct intrinsic dimensions
      videoEl.width = videoEl.videoWidth;
      videoEl.height = videoEl.videoHeight;
    } catch (err) {
      console.error("❌ Webcam error:", err);
      alert("Camera access failed. Please allow permissions or use Safari/Chrome.");
    }
  };

  stopWebcam = () => {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      console.log("🛑 Webcam stopped");
    }
  };

  loadModel = async (modelType) => {
  this.setState({ loadingModel: true });
  let detector;

  try {
    if (modelType.startsWith("movenet")) {
      const model =
        modelType === "movenet_thunder"
          ? posedetection.movenet.modelType.SINGLEPOSE_THUNDER
          : posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;

      detector = await posedetection.createDetector(posedetection.SupportedModels.MoveNet, {
        modelType: model,
      });
    } 
    else if (modelType === "posenet") {
      // ✅ Force PoseNet TF.js runtime with correct config
      detector = await posedetection.createDetector(posedetection.SupportedModels.PoseNet, {
        runtime: "tfjs",
      architecture: "MobileNetV1",
      outputStride: 16,
      inputResolution: { width: 257, height: 257 },
      multiplier: 0.75,
      quantBytes: 2,
      modelUrl: "https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/075/model-stride16.json"
      });
    }

    this.setState({ detector, loadingModel: false });
    console.log("✅ Pose model loaded:", modelType);
    return detector;
  } catch (err) {
    console.error("❌ Model load failed:", err);
    alert("Failed to load pose model. Check console for details.");
    this.setState({ loadingModel: false });
  }
};

  startDetection = async () => {
  if (this.state.running) return;

  // 1️⃣ Reset image mode
  this.setState({ imageMode: false });

  await this.startWebcam();
  const detector =
    this.state.detector || (await this.loadModel(this.state.modelType));

  const video = this.videoRef.current;
  // 2️⃣ Ensure video is ready
  await new Promise((resolve) => {
    const check = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) resolve();
      else requestAnimationFrame(check);
    };
    check();
  });

  // 3️⃣ Reset canvas internal size
  const canvas = this.canvasRef.current;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // 4️⃣ Reset canvas CSS to match container (important!)
  canvas.style.width = "100%";
  canvas.style.height = "100%";

  // 5️⃣ Reset video CSS if needed
  video.style.width = "100%";
  video.style.height = "100%";
  video.style.objectFit = "contain";

  this.setState({ running: true });
  this.lastTime = performance.now();
  this.detectFrame(video, detector);
};

  stopDetection = () => {
  if (!this.state.running) return;

  // Stop the loop
  cancelAnimationFrame(this.animationId);
  this.animationId = null;

  // Stop webcam stream (only applies to live video)
  this.stopWebcam();

  // Update state
  this.setState({ running: false });

  // ✅ Clear overlay only if we're in video mode
  if (!this.state.imageMode) {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (this.fpsRef.current) this.fpsRef.current.innerText = "FPS: --";
  }
};

  toggleDetection = async () => {
    if (this.state.running) this.stopDetection();
    else await this.startDetection();
  };

    // --- detection loop (video) ---
  detectFrame = async (video, detector) => {
    if (!this.state.running || !detector) return;

    // Prevent PoseNet “roi width cannot be 0”
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      this.animationId = requestAnimationFrame(() => this.detectFrame(video, detector));
      return;
    }

    let poses = [];
    try {
      // Choose estimate options depending on model
      const isPoseNet = this.state.modelType === "posenet";
      const estimateOpts = isPoseNet ? { maxPoses: 5, flipHorizontal: false } : { maxPoses: 1, flipHorizontal: false };

      // pass options to estimatePoses
      poses = await detector.estimatePoses(video, estimateOpts);
    } catch (err) {
      console.warn("⚠️ Pose estimation skipped:", err.message);
    }

    const now = performance.now();
    const fps = (1000 / (now - this.lastTime)).toFixed(1);
    this.lastTime = now;
    if (this.fpsRef.current) this.fpsRef.current.innerText = `FPS: ${fps}`;

    this.renderPredictions(poses, video);
    this.animationId = requestAnimationFrame(() => this.detectFrame(video, detector));
  };

    // --- image input handler ---
  handleImageSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // stop any live detection
    this.stopDetection();

    const detector =
      this.state.detector || (await this.loadModel(this.state.modelType));

    const img = this.imageRef.current;
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
  this.setState({ imageMode: true });

  const canvas = this.canvasRef.current;
  const ctx = canvas.getContext("2d");

  // ✅ match canvas exactly to image natural size
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = `${img.naturalWidth}px`;
  canvas.style.height = `${img.naturalHeight}px`;

  const poses = await detector.estimatePoses(img, {
    maxPoses: this.state.modelType === "posenet" ? 12 : 1,
    flipHorizontal: false,
  });

  // ✅ render 1:1
  this.renderPredictions(poses, img);

  if (this.fileInputRef.current) this.fileInputRef.current.value = "";
};
  };

  renderPredictions = (poses, sourceEl) => {
  const canvas = this.canvasRef.current;
  const ctx = canvas.getContext("2d");

  const isImage = sourceEl.tagName.toLowerCase() === "img";
  const isVideo = sourceEl.tagName.toLowerCase() === "video";

  let srcW, srcH;
  if (isImage) {
    srcW = sourceEl.naturalWidth;
    srcH = sourceEl.naturalHeight;

    canvas.width = srcW;
    canvas.height = srcH;
  } else if (isVideo) {
    srcW = sourceEl.videoWidth;
    srcH = sourceEl.videoHeight;

    canvas.width = srcW;
    canvas.height = srcH;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;

  poses.forEach((pose) => {
    pose.keypoints.forEach((kp) => {
      if (kp.score > this.state.confThreshold) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle =
          this.state.modelType === "posenet" ? "#FFCC00" : "#00FFFF";
        ctx.fill();
      }
    });

    const adjacentPairs = posedetection.util.getAdjacentPairs(
      this.state.modelType.startsWith("move")
        ? posedetection.SupportedModels.MoveNet
        : posedetection.SupportedModels.PoseNet
    );

    adjacentPairs.forEach(([i, j]) => {
      const kp1 = pose.keypoints[i];
      const kp2 = pose.keypoints[j];
      if (kp1.score > this.state.confThreshold && kp2.score > this.state.confThreshold) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.strokeStyle =
          this.state.modelType === "posenet" ? "#FF8800" : "#FF00FF";
        ctx.stroke();
      }
    });
  });
};


  handleModelChange = async (e) => {
    const newModelType = e.target.value;
    this.setState({ modelType: newModelType, detector: null });
    if (this.state.running) {
      this.stopDetection();
      await this.loadModel(newModelType);
      this.startDetection();
    }
  };

  handleConfChange = (e) => {
    this.setState({ confThreshold: parseFloat(e.target.value) });
  };

  render() {
    const { modelType, loadingModel, confThreshold } = this.state;

    return (
      <div style={{ textAlign: "center" }}>
        <h2 style={{fontSize: "22px", color: "#19caa4ff"}}>Makers - Real-Time Pose Estimation</h2>
        <h3>TensorFlow.js — (MoveNet / PoseNet)</h3>

        {/* Model selector */}
        <div style={{ marginBottom: "10px" }}>
          <label style={{ fontWeight: "bold", marginRight: "8px" }}>Select Model:</label>
          <select
            value={modelType}
            onChange={this.handleModelChange}
            style={{
              padding: "6px 10px",
              borderRadius: "8px",
              fontSize: "14px",
            }}
          >
            <option value="movenet_lightning">MoveNet Lightning (Fast)</option>
            <option value="movenet_thunder">MoveNet Thunder (Accurate)</option>
            <option value="posenet">PoseNet (Legacy)</option>
          </select>
        </div>

        {loadingModel && (
          <div style={{ color: "#FF4444", marginBottom: "8px" }}>
            Loading model, please wait...
          </div>
        )}

        {/* --- Confidence Threshold Slider --- */}
        <div style={{ margin: "12px 0" }}>
          <label style={{ fontSize: "14px", fontWeight: "bold", color: "#ff9900ff", marginRight: "8px" }}>
            Conf: {confThreshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={confThreshold}
            onChange={this.handleConfChange}
            style={{ width: "200px" }}
          />
        </div>

        <div
          ref={this.fpsRef}
          style={{
            fontSize: "18px",
            fontWeight: "bold",
            color: "#00FFFF",
            marginBottom: "10px",
          }}
        >
          FPS: --
        </div>

        {/* --- Main video/image/canvas container --- */}
<div
  id="viewContainer"
  style={{
    position: "relative",
    display: "inline-block",
    backgroundColor: "black",
    borderRadius: "10px",
    boxShadow: "0 0 10px #ccc",
    overflow: "hidden",
    width: this.state.imageMode ? "auto" : "90vw",
    maxWidth: this.state.imageMode ? "100vw" : "640px",
    height: this.state.imageMode ? "auto" : "480px",
    maxHeight: this.state.imageMode ? "100vh" : "480px",
  }}
>
  <video
    ref={this.videoRef}
    autoPlay
    playsInline
    muted
    style={{
      display: this.state.imageMode ? "none" : "block",
      width: "100%",
      height: "100%",
      objectFit: "contain",
      borderRadius: "10px",
    }}
  />

  <img
    ref={this.imageRef}
    alt=""
    style={{
      display: this.state.imageMode ? "block" : "none",
      width: "100%",
      height: "auto",
      maxWidth: "100vw",
      maxHeight: "100vh",
      borderRadius: "10px",
      backgroundColor: "black",
    }}
  />

  <canvas
    ref={this.canvasRef}
    style={{
      position: "absolute",
      top: 0,
      left: 0,
      zIndex: 1,
      pointerEvents: "none",
      width: this.state.imageMode ? "100%" : "100%",
      height: this.state.imageMode ? "100%" : "100%",
    }}
  />
          {/* --- Buttons --- */}
          <div
            style={{
              position: "absolute",
              bottom: window.innerWidth < 600 ? "18px" : "12px",
              left: "50%",
              transform: "translateX(-50%)",
              display: "flex",
              gap: "10px",
              zIndex: 2,
              flexWrap: "nowrap",
            }}
          >
            <button
              onClick={this.toggleDetection}
              disabled={this.state.loadingModel}
              style={{
                backgroundColor: this.state.running ? "#FF5555" : "#00CC88",
                color: "#fff",
                border: "none",
                borderRadius: "25px",
                padding: window.innerWidth < 600 ? "6px 12px" : "8px 16px",
                fontSize: window.innerWidth < 600 ? "10px" : "14px",
                fontWeight: "bold",
                cursor: "pointer",
                boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
              }}
            >
              {this.state.running ? "⏹ Stop" : "▶ Start"}
            </button>

            <label
              style={{
                backgroundColor: "#007BFF",
                color: "#fff",
                border: "none",
                borderRadius: "25px",
                padding: window.innerWidth < 600 ? "6px 12px" : "8px 16px",
                fontSize: window.innerWidth < 600 ? "10px" : "14px",
                fontWeight: "bold",
                cursor: "pointer",
                boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
              }}
            >
              📷 Select Image
              <input
                ref={this.fileInputRef}
                type="file"
                accept="image/*"
                onChange={this.handleImageSelect}
                style={{ display: "none" }}
              />
            </label>
          </div>
        </div>
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
