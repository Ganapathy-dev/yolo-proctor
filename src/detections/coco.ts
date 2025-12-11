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
    predictions.forEach(pred => {
      const [x, y, w, h] = pred.bbox;
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.font = "16px Arial";
      ctx.fillStyle = "lime";
      ctx.fillText(`${pred.class} (${pred.score.toFixed(2)})`, x, y - 5);
    });

    animationId = requestAnimationFrame(detectFrame);
  };

  detectFrame();
}
