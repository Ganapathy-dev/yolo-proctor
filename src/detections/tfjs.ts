import YOLO from "yolo-ts";

const yolo = new YOLO();
yolo.setup({
  modelUrl: "/yolo-proctor/yolov8n_web_model/model.json",
  scoreThreshold: 0.3,
  boxLineWidth: 4,
  boxLabels: true
});

let yoloModel: any = null;

export async function runYolo8n(video: HTMLVideoElement, canvas: HTMLCanvasElement, stopFlag: () => boolean) {
   if (!yoloModel) yoloModel = await yolo.loadModel();
  const detect = () => {
    if(stopFlag()) return;
    yolo.detectVideo(video, yoloModel!, canvas);
  };
  detect();
}

export async function runYolo8nImage(
  img: HTMLImageElement,
  canvas: HTMLCanvasElement
) {
  const ctx = canvas.getContext("2d")!;

  canvas.width = img.width;
  canvas.height = img.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "white";
  ctx.font = "28px Arial";
  ctx.fillText("Processing...", 20, 50);

  if (!yoloModel) yoloModel = await yolo.loadModel();

  yolo.detect(img, yoloModel, canvas, (det) => {
    const { boxes, scores, labels } = det;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    for (let i = 0; i < boxes.length; i += 4) {
      const y1 = boxes[i];
      const x1 = boxes[i + 1];
      const y2 = boxes[i + 2];
      const x2 = boxes[i + 3];

      const w = x2 - x1;
      const h = y2 - y1;

      const score = scores[i / 4];
      const label = labels[i / 4];

      ctx.strokeStyle = "lime";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      const text = `${label} (${score.toFixed(2)})`;
      ctx.font = "18px Arial";
      const tw = ctx.measureText(text).width;

      ctx.fillStyle = "lime";
      ctx.fillRect(x1, y1 - 22, tw + 8, 22);

      ctx.fillStyle = "black";
      ctx.fillText(text, x1 + 4, y1 - 5);
    }
  });
}