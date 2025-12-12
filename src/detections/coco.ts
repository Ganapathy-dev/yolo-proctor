// @ts-ignore
import * as cocoSsd from "@tensorflow-models/coco-ssd";

let currentCocoModel: cocoSsd.ObjectDetection | null = null;
let animationId: number | null = null;

export async function runCoco(video: HTMLVideoElement, canvas: HTMLCanvasElement, stopFlag: () => boolean) {
  const ctx = canvas.getContext("2d")!;

  if (!currentCocoModel) {
    currentCocoModel = await cocoSsd.load();
  }

  if (animationId) cancelAnimationFrame(animationId);

  const detectFrame = async () => {
    if (stopFlag()) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const predictions = await currentCocoModel!.detect(video);
    drawPredictions(predictions, ctx);

    animationId = requestAnimationFrame(detectFrame);
  };

  detectFrame();
}

export async function runCocoImage(
  img: HTMLImageElement,
  canvas: HTMLCanvasElement
) {
  const ctx = canvas.getContext("2d")!;
  canvas.width = img.width;
  canvas.height = img.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "white";
  ctx.font = "32px Arial";
  ctx.fillText("Processing...", 20, 50);

  if (!currentCocoModel) {
    currentCocoModel = await cocoSsd.load();
  }

  const predictions = await currentCocoModel.detect(img);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  drawPredictions(predictions, ctx);
}


function drawPredictions(predictions: cocoSsd.DetectedObject[], ctx: CanvasRenderingContext2D) {
  predictions.forEach(pred => {
    const [x, y, w, h] = pred.bbox;
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.font = "16px Arial";
    ctx.fillStyle = "lime";
    ctx.fillText(`${pred.class} (${pred.score.toFixed(2)})`, x, y - 5);
  });
}
