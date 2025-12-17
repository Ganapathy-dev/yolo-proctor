import {initCocoModel, runCocoVideo, runCocoImage} from "./detections/coco.js"
import { initYoloModel, runYoloVideo, runYoloImage } from "./detections/tfjs.js"
import { initYoloOnnx, detectImage, runOnnxWebcam } from "./detections/onnx.js"

const inputImage = document.getElementById("inputImage");
const canvas = document.getElementById("canvas");
const imageInput = document.getElementById("imageInput");

const openImageBtn = document.getElementById("openImageBtn");
const closeImageBtn = document.getElementById("closeImageBtn");

const video = document.getElementById("webcam");
const openWebcamBtn = document.getElementById("openWebcamBtn");
const stopWebcamBtn = document.getElementById("stopWebcamBtn");

const loader = document.getElementById("loader");
const loaderText = document.getElementById("loaderText");

const modeButtons = document.querySelectorAll("#buttons button");

let webcamStream = null;
let webcamRunning = false;


function showLoader(text) {
  loaderText.textContent = text || "Loading...";
  loader.style.display = "flex";
}

function hideLoader() {
  loader.style.display = "none";
}

function getModeFromURL() {
  const mode = new URLSearchParams(window.location.search).get("mode");
  if (mode === "coco" || mode === "tfjs") return mode;
  return "onnx";
}

function ensureModeInURL() {
  let mode = getModeFromURL();

  if (!mode) {
    mode = "onnx";
    const params = new URLSearchParams(window.location.search);
    params.set("mode", mode);

    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, "", newUrl);
  }

  return mode;
}

modeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    if (mode === getModeFromURL()) return;
    window.location.href = `${window.location.pathname}?mode=${mode}`;
  });
});


async function startWebcam(model, runVideo) {
  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
  });

  video.srcObject = webcamStream;
  video.style.display = "block";

  inputImage.style.display = "none";
  openWebcamBtn.style.display = "none";
  openImageBtn.style.display = "none";
  stopWebcamBtn.style.display = "inline-block";

  webcamRunning = true;
  runVideo(model, video, canvas, () => !webcamRunning);
}

function stopWebcam() {
  webcamRunning = false;

  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }

  video.style.display = "none";
  stopWebcamBtn.style.display = "none";
  openWebcamBtn.style.display = "inline-block";
  openImageBtn.style.display = "inline-block";

  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

stopWebcamBtn.addEventListener("click", stopWebcam);


openImageBtn.onclick = () => imageInput.click();

closeImageBtn.onclick = () => {
  inputImage.src = "#";
  inputImage.style.display = "none";
  closeImageBtn.style.display = "none";
  openWebcamBtn.style.display = "inline-block";
  openImageBtn.style.display = "inline-block";
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
};





// coco ssd 

async function initCoco() {
  showLoader("Loading COCO-SSD...");

  const model = await initCocoModel();

  hideLoader();

  // webcam
  openWebcamBtn.onclick = () => {
    startWebcam(model, runCocoVideo);
  };

  // image
  imageInput.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;

    inputImage.src = URL.createObjectURL(file);
    inputImage.onload = () => {
      runCocoImage(model, inputImage, canvas);
    };

    inputImage.style.display = "block";
    closeImageBtn.style.display = "inline-block";
    openWebcamBtn.style.display = "none"
    openImageBtn.style.display = "none";
  };
}



// yolov8n tfjs

async function initTfjsYolo() {
  showLoader("Loading YOLOv8 TFJS...");

  const model = await initYoloModel();

  hideLoader();

  // Webcam detection
  openWebcamBtn.onclick = () => {
    startWebcam(model, runYoloVideo);
  };

  // Image detection
  imageInput.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;

    inputImage.src = URL.createObjectURL(file);
    inputImage.onload = () => {
      runYoloImage(model, inputImage, canvas);
    };

    inputImage.style.display = "block";
    closeImageBtn.style.display = "inline-block";
    openWebcamBtn.style.display = "none"
    openImageBtn.style.display = "none";
  };
}



// yolov8n onnx

async function initOnnx() {
  showLoader("Loading YOLOv8 ONNX...");

  const sessions = await initYoloOnnx();

  hideLoader();

  // webcam
  openWebcamBtn.onclick = () => {
    startWebcam(sessions, (sessions, video, canvas) => {
      runOnnxWebcam(sessions, video, canvas);
    });
  };

  // image detection
  imageInput.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;
    inputImage.src = URL.createObjectURL(file);
    inputImage.onload = async () => {
      canvas.width = 640;
      canvas.height = 640;
      await detectImage(
        inputImage,
        canvas,
        sessions,
        100,
        0.45,
        0.25,
        [1, 3, 640, 640]
      );
    };
    inputImage.style.display = "block";
    closeImageBtn.style.display = "inline-block";
    openWebcamBtn.style.display = "none";
    openImageBtn.style.display = "none";
  };
}

function updateHeaderAndDescription(model) {
  const headingEl = document.getElementById("heading");
  const descriptionEl = document.getElementById("description");

  switch (model) {
    case "onnx":
      headingEl.textContent = "YOLOv8n ONNX Object Detection App";
      descriptionEl.innerHTML = "YOLOv8n object detection running in-browser powered by <code>onnxruntime-web</code>";
      break;

    case "tfjs":
      headingEl.textContent = "YOLOv8n TFJS Object Detection App";
      descriptionEl.innerHTML = "YOLOv8n object detection running in-browser powered by <code>TensorFlow.js</code>";
      break;

    case "coco":
      headingEl.textContent = "COCO-SSD Object Detection App";
      descriptionEl.innerHTML = "COCO-SSD object detection running in-browser powered by <code>TensorFlow.js</code>";
      break;

    default:
      headingEl.textContent = "Object Detection App";
      descriptionEl.textContent = "Generic object detection application";
      break;
  }
}

async function bootstrap() {
  const mode = ensureModeInURL();
  updateHeaderAndDescription(mode);
  modeButtons.forEach(btn => {
    if (btn.dataset.mode === mode) btn.classList.add("active");
  });

  if (mode === "coco") return initCoco();
  if (mode === "tfjs") return initTfjsYolo();
  return initOnnx();
}

window.addEventListener("load", bootstrap);
