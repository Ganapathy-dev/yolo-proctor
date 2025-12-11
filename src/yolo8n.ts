import YOLO from "yolo-ts";

const yolo = new YOLO();

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;

yolo.setup({
  modelUrl: "./public/yolov8n_web_model/model.json",
  scoreThreshold: 0.3,
  boxLineWidth: 4,
  boxLabels: true,
});

async function initWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise<void>((resolve) => (video.onloadedmetadata = () => resolve()));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function startDetection() {
  await initWebcam();
  const model = await yolo.loadModel();
  console.log("YOLO model loaded!", model);

  yolo.detectVideo(video, model!, canvas);
}

startDetection().catch(console.error);
