# Hivemapper Edge Privacy

![! ONNX YOLOv8 Object Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/raw/main/doc/img/detected_objects.jpg)
*Original image: [https://www.flickr.com/photos/nicolelee/19041780](https://www.flickr.com/photos/nicolelee/19041780)*

# Important
- The input images are directly resized to match the input size of the model. I skipped adding the pad to the input image, it might affect the accuracy of the model if the input image has a different aspect ratio compared to the input size of the model. Always try to get an input size with a ratio close to the input images you will use.

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```shell
git clone https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection.git
cd ONNX-YOLOv8-Object-Detection
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
python main.py --model-path /home/ed/git/ONNX-YOLOv8-Object-Detection/devel/models/pvc.onnx --show-detection True --unprocessed-framekm-path /home/ed/git/ONNX-YOLOv8-Object-Detection/devel/unprocessed_framekm --framekm-path /home/ed/git/ONNX-YOLOv8-Object-Detection/devel/framekm --metadata-path /home/ed/git/ONNX-YOLOv8-Object-Detection/devel/metadata --ml-metadata-path /home/ed/git/ONNX-YOLOv8-Object-Detection/devel/ml_metadata --model-hash-path /home/ed/git/ONNX-YOLOv8-Object-Detection/devel/models/pvc.onnx.hash
```
