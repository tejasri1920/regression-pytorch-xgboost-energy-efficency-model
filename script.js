// Feature names in the exact order your model expects:
const featureNames = [
  "X1: Relative Compactness",
  "X2: Surface Area",
  "X3: Wall Area",
  "X4: Roof Area",
  "X5: Overall Height",
  "X6: Orientation",
  "X7: Glazing Area",
  "X8: Glazing Area Distribution"
];

// Create input fields dynamically
const inputsDiv = document.getElementById("inputs");
featureNames.forEach((name, i) => {
  const id = "x" + (i + 1);
  const row = document.createElement("div");
  row.innerHTML = `
    <label>${name}
      <input type="number" id="${id}" step="0.01">
    </label>
  `;
  inputsDiv.appendChild(row);
});

// Load ONNX model (must be in same folder as this file)
let sessionPromise = ort.InferenceSession.create("energy_model.onnx");

document.getElementById("predictBtn").onclick = async () => {
  const session = await sessionPromise;

  // Read inputs
  let values = [];
  for (let i = 1; i <= 8; i++) {
    const v = parseFloat(document.getElementById("x" + i).value);
    values.push(isNaN(v) ? 0 : v);
  }

  // Build tensor [1, 8]
  const inputTensor = new ort.Tensor("float32", Float32Array.from(values), [1, 8]);

  // Prepare feeds
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  // Run inference
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];

  // Display prediction
  document.getElementById("prediction").innerText = output.data[0].toFixed(3);
};
