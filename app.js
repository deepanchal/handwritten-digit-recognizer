var model = undefined;
var canvas = document.querySelector("#drawing-canvas");
var ctx = canvas.getContext("2d");
var strokeStyle = "white";
var strokeLineJoin = "round";
var lineWidth = 10;
var drawing = false;
var resultBox = document.querySelector(".result");

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", (e) => {
  stopDrawing(e);
  predict();
});
canvas.addEventListener("mouseout", stopDrawing);

canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", (e) => {
  stopDrawing(e);
  predict();
});

function getPosition(event) {
  const clientX = event.clientX || event.touches[0].pageX;
  const clientY = event.clientY || event.touches[0].pageY;
  const canvasX = clientX - event.target.offsetLeft;
  const canvasY = clientY - event.target.offsetTop;

  return { x: canvasX, y: canvasY };
}

function startDrawing(ev) {
  ev.preventDefault();

  const coord = getPosition(ev);
  ctx.beginPath();
  ctx.moveTo(coord.x, coord.y);
  ctx.lineWidth = lineWidth;
  ctx.lineCap = strokeLineJoin;
  ctx.strokeStyle = strokeStyle;
  ctx.fill();
  drawing = true;
}

function draw(ev) {
  ev.preventDefault();

  if (drawing) {
    const coord = getPosition(ev);
    ctx.lineTo(coord.x, coord.y);
    ctx.stroke();
  }
}

function stopDrawing(ev) {
  ev.preventDefault();
  drawing = false;
}

document.querySelector(".clear-btn").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.querySelector(".result").innerText = null;
  resultBox.style.visibility = "hidden";
});

document.querySelector(".predict-btn").addEventListener("click", predict);

async function loadModel() {
  resultBox.style.visibility = "visible";
  resultBox.innerText = "Loading Model...";
  model = await tf.loadLayersModel("model/model.json");
  resultBox.innerHTML = "Model Loaded &#10003;";
}

async function predict() {
  let tensor = processImage(canvas);
  let predictions = await model.predict(tensor).data();
  let results = Array.from(predictions);
  resultBox.style.visibility = "visible";
  resultBox.innerText = "Model's Prediction: " + results.indexOf(Math.max(...results));
}

function processImage(canvas) {
  let tensor = tf.browser
    .fromPixels(canvas)
    .resizeNearestNeighbor([28, 28])
    .mean(2)
    .expandDims(2)
    .expandDims()
    .toFloat();
  return tensor.div(255.0);
}

loadModel();
