import labels from "./labels.json";

export function renderBoxes(canvas, boxes) {
  console.log("Entering renderBoxes");

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const boxColor = "lime";
  const textColor = "red";
  const fontSize = Math.max(
    Math.round(Math.max(canvas.width, canvas.height) / 45),
    12
  );

  ctx.font = `${fontSize}px Arial`;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const label = labels[box.label];
    const score = (box.probability * 100).toFixed(1);
    const [x, y, w, h] = box.bounding;
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, w, h);

    const text = `${label} ${score}%`;
    const textY = y - fontSize - 2 < 0 ? y + 2 : y - fontSize - 2;
    ctx.fillStyle = textColor;
    ctx.fillText(text, x + 2, textY);

  });
}
