const express = require('express');
const multer = require('multer');
const { Tensor, InferenceSession } = require('onnxjs');
const Jimp = require('jimp');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });
const port = 8080;

const classes = ['cardboard', 'food', 'glass', 'hazardous', 'metal', 'paper', 'plastic', 'trash'];

app.post('/image', upload.single('file'), async (req, res) => {
    const { file, body: { category } } = req;
    if (!file) {
        return res.status(400).send("No file part in the request");
    }

    const image = await Jimp.read(file.path);
    const resizedImage = image.resize(224, 224);
    const imageBuffer = await resizedImage.getBufferAsync(Jimp.MIME_JPEG);
    const imageArray = new Float32Array(imageBuffer);
    const tensor = new Tensor(imageArray, 'float32', [1, 3, 224, 224]);

    const session = new InferenceSession();
    await session.loadModel('model.onnx');
    const outputMap = await session.run([tensor]);
    const outputData = outputMap.values().next().value.data;
    const maxIndex = outputData.indexOf(Math.max(...outputData));

    const folderPath = Math.random() < 0.8 ? `photos/train/${category}` : `photos/test/${category}`;
    fs.mkdirSync(folderPath, { recursive: true });
    fs.renameSync(file.path, path.join(folderPath, `${Math.random()}.jpg`));

    res.send([classes[maxIndex], outputData[maxIndex]]);
});

app.listen(port, () => console.log(`Server running on port ${port}`));