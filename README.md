# Hivemapper Edge Privacy

## Requirements

* Runs only on linux kernel
* Check the **requirements.txt** file.
* For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

## Get required dependencies
```bash
pip install -r requirements.txt
```

## ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

## ONNX model
You can convert the model using the following code after installing ultralitics (`pip install ultralytics`):
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
model.export(format="onnx", imgsz=[960, 960])
```

## Run the ONNX model
```bash
python main.py --model-path /home/ed/git/hivemapper-edge-privacy/devel/models/pvc.onnx --show-detection True --unprocessed-framekm-path /home/ed/git/hivemapper-edge-privacy/devel/unprocessed_framekm --unprocessed-metadata-path /home/ed/git/hivemapper-edge-privacy/devel/unprocessed_metadata --framekm-path /home/ed/git/hivemapper-edge-privacy/devel/framekm --metadata-path /home/ed/git/hivemapper-edge-privacy/devel/metadata --ml-metadata-path /home/ed/git/hivemapper-edge-privacy/devel/ml_metadata --model-hash-path /home/ed/git/hivemapper-edge-privacy/devel/models/pvc.onnx.hash
```

## Mimic added folders in the `unprocessed_framekm`
To mimic what the odc-api is doing, we need 2 things:
- a relevant packed framekm metadata json
- equivalent packed framekm folder under unprocessed_framekm

The python program expects that the packed metadata be present at the same time (or before) the packed framekm unprocessed_framekm is produced.
Make sure that you have nothing in `unprocessed_framekm` and that `unprocessed_metadata` has the framekm that you are about to add.
For reproduction purposes, I would suggest having a `unprocessed_framekm_orig` folder containing all the framekms that you will test the model on.

```bash
cp -r /mnt/data/unprocessed_framekm_orig/km_20230830_230525_10_0 /mnt/data/unprocessed_framekm_orig/km_20230830_230525_10_0.clone && mv /mnt/data/unprocessed_framekm_orig/km_20230830_230525_10_0.clone /mnt/data/unprocessed_framekm/km_20230830_230525_10_0  
```

> It's important to do a `mv` and not a `cp` on the folder as a `cp` is not atomic

This will trigger the process of the python to start processing the framekm.

Open in another terminal, to check the logs of the service:
```bash
journalctl -u hivemapper_edge_privacy -f
```
