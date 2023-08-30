# Hivemapper Edge Privacy

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

### Get required dependencies
```bash
pip install -r requirements.txt
```

### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
You can convert the model using the following code after installing ultralitics (`pip install ultralytics`):
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
model.export(format="onnx", imgsz=[960, 960])
```

# Run the ONNX model
```bash
python main.py --model-path /home/ed/git/hivemapper-edge-privacy/devel/models/pvc.onnx --show-detection True --unprocessed-framekm-path /home/ed/git/hivemapper-edge-privacy/devel/unprocessed_framekm --unprocessed-metadata-path /home/ed/git/hivemapper-edge-privacy/devel/unprocessed_metadata --framekm-path /home/ed/git/hivemapper-edge-privacy/devel/framekm --metadata-path /home/ed/git/hivemapper-edge-privacy/devel/metadata --ml-metadata-path /home/ed/git/hivemapper-edge-privacy/devel/ml_metadata --model-hash-path /home/ed/git/hivemapper-edge-privacy/devel/models/pvc.onnx.hash
```
