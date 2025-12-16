import cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import { detectImage } from "./detect.js";

const inputImage = document.getElementById("inputImage");
const canvas = document.getElementById("canvas");
const imageInput = document.getElementById("imageInput");
const openBtn = document.getElementById("openImageBtn");
const closeBtn = document.getElementById("closeImageBtn");
const loader = document.getElementById("loader");
const loaderText = document.getElementById("loaderText");

const video = document.getElementById("webcam");
const openWebcamBtn = document.getElementById("openWebcamBtn");
const stopWebcamBtn = document.getElementById("stopWebcamBtn");

let webcamStream = null;
let webcamRunning = false;

function showLoader(text = "Loading...") {
  loaderText.textContent = text;
  loader.style.display = "flex";
}

function hideLoader() {
  loader.style.display = "none";
}

openBtn.addEventListener("click", () => imageInput.click());

closeBtn.addEventListener("click", () => {
  imageInput.value = "";
  URL.revokeObjectURL(inputImage.src);
  inputImage.src = "#";
  inputImage.style.display = "none";
  closeBtn.style.display = "none";

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

imageInput.addEventListener("change", async (e) => {
  if (typeof cv === "undefined") {
    alert("OpenCV is not ready yet. Please ensure it is imported.");
    return;
  }

  const file = e.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  inputImage.src = url;
  inputImage.style.display = "block";
  closeBtn.style.display = "inline-block";

  inputImage.onload = async () => {
    // Set canvas size to match image
    canvas.width = 640
    canvas.height = 640

    const ctx = canvas.getContext("2d");
    ctx.drawImage(inputImage, 0, 0);

    try {
      await detectImage(
        inputImage,
        canvas,
        {
          net: window.yolov8Session,
          nms: window.nmsSession
        },
        100,      // topk
        0.45,     // iouThreshold
        0.25,     // scoreThreshold
        [1, 3, 640, 640] // inputShape
      );

    } catch (err) {
      console.error("Detection error:", err);
    }

    hideLoader();
  };
});



openWebcamBtn.addEventListener("click", async () => {
  if (!window.yolov8Session) {
    alert("Model not loaded yet");
    return;
  }

  showLoader("Starting webcam...");

  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
  });

  video.srcObject = webcamStream;
  video.style.display = "block";
  inputImage.style.display = "none";

  openWebcamBtn.style.display = "none";
  stopWebcamBtn.style.display = "inline-block";

  webcamRunning = true;
  hideLoader();

    requestAnimationFrame(runWebcamDetection);
});

async function runWebcamDetection() {
    
  if (!webcamRunning) return;
    canvas.width = 640;
    canvas.height = 640;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      await detectImage(
        canvas,
        canvas,
        {
          net: window.yolov8Session,
          nms: window.nmsSession
        },
        100,
        0.45,
        0.25,
        [1, 3, 640, 640]
      );
    } catch (e) {
      console.error(e);
    }
  requestAnimationFrame(runWebcamDetection);
}



stopWebcamBtn.addEventListener("click", () => {
  webcamRunning = false;

  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }

  video.style.display = "none";
  stopWebcamBtn.style.display = "none";
  openWebcamBtn.style.display = "inline-block";

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});


async function init() {
  showLoader("Loading OpenCV.js...");

  if (typeof cv === "undefined") {
    console.log("OpenCV is not ready. Please reload the page.");
    return;
  }

  cv["onRuntimeInitialized"] = async () => {
    showLoader("Loading YOLOv8n model...");

    try {
      if (!window.yolov8Session)
        window.yolov8Session = await ort.InferenceSession.create("./yolov8n.onnx",{ executionProviders: ["wasm"] });
      if (!window.nmsSession)
        window.nmsSession = await ort.InferenceSession.create("./nms-yolov8.onnx", { executionProviders: ["wasm"] });

      const inputShape = [1, 3, 640, 640];
      const warmupTensor = new ort.Tensor(
        "float32",
        new Float32Array(inputShape.reduce((a, b) => a * b)),
        inputShape
      );
      await window.yolov8Session.run({ images: warmupTensor });

      hideLoader();
      console.log("Models loaded! You can now upload an image.");

    } catch (err) {
      console.error("Model loading error:", err);
    }
  };
}


window.addEventListener("load", async () => {
  await init();
});