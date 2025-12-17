import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";

let cocoModel = null;
let animationId = null;

export async function initCocoModel() {
  if (!cocoModel) {
    cocoModel = await cocoSsd.load();
  }
  return cocoModel;
}

export async function runCocoVideo(model, video, canvas) {
  const ctx = canvas.getContext("2d");

  if (video.videoWidth === 0) {
    await new Promise(resolve => {
      video.onloadedmetadata = resolve;
    });
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  const detectFrame = async () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const predictions = await model.detect(video);
    drawPredictions(predictions, ctx);
    animationId = requestAnimationFrame(detectFrame);
  };

  detectFrame();
}

export async function runCocoImage(model, img, canvas) {
  const ctx = canvas.getContext("2d");

  canvas.width = img.width;
  canvas.height = img.height;


  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "white";
  ctx.font = "28px Arial";
  ctx.fillText("Processing...", 20, 40);

  const predictions = await model.detect(img);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  drawPredictions(predictions, ctx);
}


function drawPredictions(predictions, ctx) {
  predictions.forEach(pred => {
    const [x, y, w, h] = pred.bbox;

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    const label = `${pred.class} (${pred.score.toFixed(2)})`;
    ctx.font = "16px Arial";

    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "lime";
    ctx.fillRect(x, y - 20, textWidth + 6, 20);

    ctx.fillStyle = "black";
    ctx.fillText(label, x + 3, y - 5);
  });
}
