import { runCoco } from "./detections/coco";
import { runYolo8n } from "./detections/tfjs";
import { runOnnx } from "./detections/onnx";
import { gatherSystemInfo, renderSystemInfo, initCamera } from "./utils";

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const sysToggle = document.getElementById("sysToggle")!;
const sysInfo = document.getElementById("sysInfo")!;
const sysContent = document.getElementById("sysContent")!;
const buttons = document.querySelectorAll<HTMLButtonElement>("#buttons button");
const currentModeTitle = document.getElementById("currentModeTitle")!;
const modelSelector = document.getElementById("modelSelector") as HTMLSelectElement;
const logDiv = document.getElementById("log")!;

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
    // Reload page with selected mode in URL
    window.location.href = `${window.location.pathname}?mode=${mode}`;
  });
});

function getModeFromURL(): "coco" | "yolo8n" | "onnx" {
  const mode = new URLSearchParams(window.location.search).get("mode");
  if (mode === "yolo8n" || mode === "onnx") return mode;
  return "coco";
}

async function startDetection() {
  const mode = getModeFromURL();
  currentModeTitle.textContent =
    mode === "coco" ? "Coco SSD" :
    mode === "yolo8n" ? "YOLO8n-TFJS" :
    "YOLO ONNX";

  (document.getElementById("sidebar")!).style.display = mode === "onnx" ? "block" : "none";

  await initCamera(video);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (mode === "coco") await runCoco(video, canvas, () => false);
  if (mode === "yolo8n") await runYolo8n(video, canvas, () => false);
  if (mode === "onnx") await runOnnx(video, canvas, modelSelector, logDiv);
}

startDetection();
