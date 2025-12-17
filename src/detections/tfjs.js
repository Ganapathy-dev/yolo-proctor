import YOLO from "yolo-ts";

const yolo = new YOLO();
let yoloModel = null;

export async function initYoloModel() {
  yolo.setup({
    modelUrl: "/yolo-proctor/yolov8n_web_model/model.json",
    scoreThreshold: 0.3,
    boxLineWidth: 4,
    boxLabels: true
  });

  if (!yoloModel) {
    yoloModel = await yolo.loadModel();
  }

  return yoloModel;
}

export async function runYoloVideo(model, video, canvas) {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      await new Promise(resolve => {
        video.onloadedmetadata = resolve;
      });
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    yolo.detectVideo(video, model, canvas);
}

export async function runYoloImage(model, img, canvas) {
  const ctx = canvas.getContext("2d");

  canvas.width = img.width;
  canvas.height = img.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "white";
  ctx.font = "28px Arial";
  ctx.fillText("Processing...", 20, 50);

  yolo.detect(img, model, canvas, det => {
    const boxes = det.boxes;
    const scores = det.scores;
    const labels = det.labels;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

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
      ctx.fillText(text, x1 + 4, y1 - 20);
    }
  });
}
