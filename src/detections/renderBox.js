const labels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
]
 

export function renderBoxes(canvas, boxes) {
  console.log("Entering renderBoxes");

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const boxColor = "lime";
  const textColor = "black"; // change text to black
  const bgColor = "lime"; // background for text
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
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const text = `${label} ${score}%`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = fontSize + 6;

    const textY = y - textHeight - 2 < 0 ? y + 2 : y - textHeight - 2;

    ctx.fillStyle = bgColor;
    ctx.fillRect(x, textY, textWidth + 4, textHeight);

    ctx.fillStyle = textColor;
    ctx.fillText(text, x + 2, textY + 2);
  });
}

