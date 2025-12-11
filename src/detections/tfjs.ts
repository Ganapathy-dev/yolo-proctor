import YOLO from "yolo-ts";

const yolo = new YOLO();
yolo.setup({
  modelUrl: "./public/yolov8n_web_model/model.json",
  scoreThreshold: 0.3,
  boxLineWidth: 4,
  boxLabels: true
});

export async function runYolo8n(video: HTMLVideoElement, canvas: HTMLCanvasElement, stopFlag: () => boolean) {
  const model = await yolo.loadModel();
  const detect = () => {
    if(stopFlag()) return;
    yolo.detectVideo(video, model!, canvas);
  };
  detect();
}
