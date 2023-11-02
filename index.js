const express = require('express');
const multer = require('multer');
const ort = require('onnxruntime-node');
const Jimp = require('jimp');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });
const port = 8080;

const classes = ['cardboard', 'food', 'glass', 'hazardous', 'metal', 'paper', 'plastic', 'trash'];

// Load ONNX model
async function loadModel(modelPath) {
    const session = await ort.InferenceSession.create(modelPath);
    return session;
}

// Run model on input data
async function runModel(session, inputData) {
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    return results;
}

// Softmax function to convert logits to probabilities
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// Main endpoint for image processing and inference
app.post('/image', upload.single('file'), async (req, res) => {
    const { file, body: { category } } = req;
    if (!file) {
        return res.status(400).send("No file part in the request");
    }

    const image = await Jimp.read(file.path);
    const resizedImage = image.resize(224, 224);
    const buffer = resizedImage.bitmap.data;
    const tensorData = new Float32Array(3 * 224 * 224);

    for (let y = 0; y < 224; ++y) {
        for (let x = 0; x < 224; ++x) {
            for (let c = 0; c < 3; ++c) {
                tensorData[c * 224 * 224 + y * 224 + x] = buffer[(y * 224 + x) * 4 + c] / 255;
            }
        }
    }

    try {
        const modelPath = 'model.onnx'; // Replace with the path to your ONNX model
        const session = await loadModel(modelPath);
        const results = await runModel(session, tensorData);
        const outputData = softmax(results.output.data); // Apply softmax to logits
        const maxIndex = outputData.indexOf(Math.max(...outputData));

        console.log(outputData);

        const folderPath = Math.random() < 0.8 ? `photos/train/${category}` : `photos/test/${category}`;
        fs.mkdirSync(folderPath, { recursive: true });
        fs.renameSync(file.path, path.join(folderPath, `${Math.random()}.jpg`));

        res.send([classes[maxIndex], outputData[maxIndex]]);
    } catch (error) {
        console.error(error);
        res.status(500).send('Error during model inference');
    }
});

app.listen(port, () => console.log(`Server running on port ${port}`));
