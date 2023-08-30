from yolov8 import YOLOv8
from PIL import Image

import yolov8.constant as constant
import cv2
import io
import inotify.adapters
import inotify.constants
import json
import logging
import yolov8.ml_metadata as ml_metadata
import yolov8.utils
import os
import shutil


class Watcher:
    notifier: inotify.adapters.Inotify
    onnx_detector: YOLOv8
    framekm_path: str
    metadata_path: str
    ml_metadata_path: str
    logger: logging.Logger

    def __init__(self, detector: YOLOv8, framekm_path: str, metadata_path: str, ml_metadata_path: str, logger: logging.Logger):
        self.notifier = inotify.adapters.Inotify()
        self.onnx_detector = detector
        self.framekm_path = framekm_path
        self.metadata_path = metadata_path
        self.ml_metadata_path = ml_metadata_path
        self.logger = logger

    def add_watch(self, path: str):
        self.notifier.add_watch(path)
    
    def run(self): 
        for event in self.notifier.event_gen(yield_nones=False):
            (_, type_names, path, name) = event
            self.logger.debug(f'[Event] type_names: {type_names}, path: {path}, name: {name}')
            
            if type_names[0] == 'IN_CREATE' or type_names[0] == "IN_MOVED_TO":
                new_path = os.path.join(path, name)
                if os.path.isdir(new_path) and name[0] != r'_':
                    self.logger.info(f"processing new folder: {new_path}")
                    self._process_folder(name, path, new_path)
                    self.logger.debug(f"done processing folder: {new_path}")
                
                if os.path.isfile(new_path) and 'km_completed_' in name:
                    renamed_path = os.path.join(self.framekm_path, name.replace('km_completed_', ''))
                    self.logger.debug(f'path file {path}')
                    self.logger.debug(f'new_path {new_path}')
                    self.logger.debug(f'renamed_path {renamed_path}')
                    os.rename(new_path, renamed_path)
                    self.logger.info(f'{new_path} moved to {renamed_path}')
                
                if os.path.isfile(new_path) and 'metadata_ml_completed_' in name:
                    self.logger.debug(f'completed ml {name}')
                    renamed_path = os.path.join(self.ml_metadata_path, name.replace('metadata_ml_completed_', '') + '.json')
                    self.logger.debug(f'renamed_path {renamed_path}')
                    os.rename(new_path, renamed_path)
                    self.logger.info(f'{new_path} moved to {renamed_path}')

    def _process_folder(self, name: str, orig_path: str, new_folder_path: str):
        framekm_name = os.path.join(new_folder_path, f'bin_{name}') 
        frames = []
        frames_sizes = []
        total_processed_frame_size = 0.0
        for f in os.listdir(new_folder_path):
            p = os.path.join(new_folder_path, f)
            self.logger.debug(f'file name {p}')
            img = cv2.imread(p, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append((p, img))
        
        with open(f'{framekm_name}', 'ab') as f:
            privacy_ml_metadata = ml_metadata.GenericMLMetadata()  # for all the frames in a packed framekm
            for j, val in enumerate(frames):
                img_id = val[0]
                img = val[1]
                img_ml_data = ml_metadata.MLFrameData()  # for all the boxes of an image
                
                boxes, scores, classe_ids = self.onnx_detector(img, img_ml_data)
                converted_box = yolov8.utils.xyxyxywh2(boxes)
                for i, class_id in enumerate(classe_ids):
                    bounding_box = ml_metadata.BoundingBox()
                    bounding_box.set_class_id(constant.CLASS_NAMES[class_id])
                    bounding_box.set_confidence(scores[i])
                    bounding_box.set_cxcywh(converted_box[i])
                    img_ml_data.detections.append(bounding_box)

                combined_img = self.onnx_detector.draw_detections(img)
                combined_img = self.onnx_detector.blur_boxes(combined_img, img_ml_data)
                
                im = Image.fromarray(combined_img)
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                size_of_processed_image = len(img_byte_arr)
                frames_sizes.append(size_of_processed_image)  # need to store this on the metadata json file
                total_processed_frame_size += size_of_processed_image

                f.write(img_byte_arr)

                img_ml_data.img_id = img_id
                img_ml_data.name = f'{name}.json'

                privacy_ml_metadata.frame_data.append(img_ml_data)

            with open(os.path.join(self.metadata_path, name), 'w+') as f:
                original_content = json.loads(f.read())
                orignal_content.bundle.size = total_processed_frame_size
                for i, size in enumerate(frames_sizes):
                    pass
                

            privacy_ml_metadata.model_hash = self.onnx_detector.model_hash
            metadata = ml_metadata.MLMetadata()
            metadata.privacy = privacy_ml_metadata
            metadata_path = os.path.join(orig_path, f'metadata_ml_completed_{name}')

            with open(metadata_path, 'w') as f:
                f.write(metadata.toJson())

        self.logger.info(f'moving and cleaning up directories and files for {name}')
        framkm_name_orig_path = os.path.join(orig_path, f'km_completed_{name}')
        os.rename(framekm_name, framkm_name_orig_path)
        shutil.rmtree(new_folder_path)
