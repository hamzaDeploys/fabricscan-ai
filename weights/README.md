# weights/

Trained FabricScan AI models. Classes, in output order: `good`, `hole`, `stain`, `broken_thread`.

| File | What it is |
|---|---|
| `fabricscan.onnx` | The best model (highest macro F1 on the test split); the website runs this file |
| `mobilenetv2.keras`, `mobilenetv2.onnx` | MobileNetV2, ImageNet weights, frozen then top 40 layers fine-tuned |
| `scratch_cnn.keras`, `scratch_cnn.onnx` | Custom 4-stage CNN trained from random weights |
| `model_meta.json` | Class order, input format, chosen model and its test accuracy |
| `metrics.json` | Test metrics (accuracy, precision, recall, F1, confusion matrix), size, speed and training history of both models |

**Input:** one RGB image, 224 x 224, raw pixel values 0-255 as float32, layout NHWC (`[batch, 224, 224, 3]`).
Normalisation is inside the model. Prepare a photo like the training images: EXIF rotation, centre square crop, resize
to 256 then 224 px.

**Output:** `probs`, softmax probabilities `[batch, 4]`. The ONNX files also return `cam` (`[batch, h, w, 4]`), the class
activation maps the website uses to locate the defect.

## Test it

```bash
pip install numpy pillow onnxruntime
python predict.py test_samples --eval        # labelled test images in test_samples/<class>/
python predict.py my_photo.jpg               # any photo
```

## Load it yourself

```python
import numpy as np, onnxruntime as ort
session = ort.InferenceSession("weights/fabricscan.onnx")
probs = session.run(["probs"], {"image": batch_float32_nhwc})[0]

import keras  # TensorFlow 2.20
model = keras.models.load_model("weights/mobilenetv2.keras")
probs = model.predict(batch_float32_nhwc)
```
