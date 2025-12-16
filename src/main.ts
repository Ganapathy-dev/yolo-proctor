import { runCoco, runCocoImage } from "./detections/coco";
import { runYolo8n, runYolo8nImage } from "./detections/tfjs";
// import { runOnnxImage, runOnnxWebcam, stopOnnxWebcam } from "./detections/onnx";
import { gatherSystemInfo, renderSystemInfo, initCamera } from "./utils";

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const sysToggle = document.getElementById("sysToggle")!;
const sysInfo = document.getElementById("sysInfo")!;
const sysContent = document.getElementById("sysContent")!;
const buttons = document.querySelectorAll<HTMLButtonElement>("#buttons button");
const modeButtons = document.querySelectorAll<HTMLButtonElement>("#modeButtons button");
const currentModeTitle = document.getElementById("currentModeTitle")!;
const imageContainer = document.getElementById("imageContainer")!;
const imageInput = document.getElementById("imageInput") as HTMLInputElement;
const imageCanvas = document.getElementById("imageCanvas") as HTMLCanvasElement;
const videoWrapper = document.getElementById("video-wrapper")!;

let currentInputMode: "webcam" | "image" = "image";
let cameraStarted = false;

sysToggle.addEventListener("click", async () => {
  sysInfo.classList.toggle("open");
  if (sysInfo.classList.contains("open")) {
    sysContent.textContent = "Loading...";
    const info = await gatherSystemInfo(video);
    renderSystemInfo(info, sysContent);
  }
});

buttons.forEach(btn => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode!;
    if (mode === getModeFromURL()) return;
    window.location.href = `${window.location.pathname}?mode=${mode}`;
  });
});

modeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    currentInputMode = btn.dataset.mode as "webcam" | "image";
    updateInputModeUI();
    if (currentInputMode === "webcam" && !cameraStarted) {
      startWebcamDetection();
    }else{
      stopWebcam();
    }
  });
});

function updateInputModeUI() {
  if (currentInputMode === "webcam") {
    videoWrapper.style.display = "block";
    imageContainer.style.display = "none";
  } else {
    videoWrapper.style.display = "none";
    imageContainer.style.display = "flex";
  }
}

function getModeFromURL(): "coco" | "yolo8n" | "onnx" {
  const mode = new URLSearchParams(window.location.search).get("mode");
  if (mode === "yolo8n" || mode === "onnx") return mode;
  return "coco";
}

async function startWebcamDetection() {
  cameraStarted = true;
  const modelMode = getModeFromURL();

  await initCamera(video);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (modelMode === "coco") await runCoco(video, canvas, () => false);
  if (modelMode === "yolo8n") await runYolo8n(video, canvas, () => false);
  // if (modelMode === "onnx") await runOnnxWebcam(video, canvas, "/yolo-proctor/yolov8n.onnx");
}

function stopWebcam() {
  if (!video.srcObject) return;

  const stream = video.srcObject as MediaStream;
  stream.getTracks().forEach(track => track.stop());
  video.srcObject = null;
  cameraStarted = false;

  // stopOnnxWebcam();
}

function setupImageDetection() {
  const modelMode = getModeFromURL();

  imageInput.addEventListener("change", async () => {
    const file = imageInput.files?.[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
      imageCanvas.width = img.width;
      imageCanvas.height = img.height;

      if (modelMode === "coco") await runCocoImage(img, imageCanvas);
      if (modelMode === "yolo8n") await runYolo8nImage(img, imageCanvas);
      // if (modelMode === "onnx") await runOnnxImage(img, imageCanvas, "/yolo-proctor/yolov8n.onnx");
    };
  });
}

function init() {
  currentModeTitle.textContent =
    getModeFromURL() === "coco" ? "Coco SSD" :
    getModeFromURL() === "yolo8n" ? "YOLO8n-TFJS" :
    null;

  updateInputModeUI();
  setupImageDetection();
}

init();
