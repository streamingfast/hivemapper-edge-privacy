from yolov8 import YOLOv8
from PIL import Image

import yolov8.constant as constant
import cv2
import io
import json
import logging
import yolov8.ml_metadata as ml_metadata
import yolov8.utils
import os
import glob
import shutil


class Watcher:
    onnx_detector: YOLOv8
    unprocessed_framekm_path: str
    unprocessed_metadata_path: str
    framekm_path: str
    metadata_path: str
    ml_metadata_path: str
    logger: logging.Logger

    def __init__(self, detector: YOLOv8, unprocessed_framekm_path: str, unprocessed_metadata_path: str, framekm_path: str, metadata_path: str, ml_metadata_path: str, logger: logging.Logger):
        self.onnx_detector = detector
        self.unprocessed_framekm_path = unprocessed_framekm_path
        self.unprocessed_metadata_path = unprocessed_metadata_path
        self.framekm_path = framekm_path
        self.metadata_path = metadata_path
        self.ml_metadata_path = ml_metadata_path
        self.logger = logger

    def scan_and_process_framekm(self):
        not_ready_unprocessed_frame_debt = glob.glob(os.path.join(self.unprocessed_framekm_path, '_km*'))
        self.logger.debug(f'Debt of not ready unprocessed framekm: {len(not_ready_unprocessed_frame_debt)}')
        
        unprocessed_framekm_debt = glob.glob(os.path.join(self.unprocessed_framekm_path, 'km_*'))
        if len(unprocessed_framekm_debt) == 0:
            self.logger.info(f'No unprocessed framekm debt')
            return

        self.logger.debug(f'Debt of unprocessed framekm: {len(unprocessed_framekm_debt)}')
        for unprocessed_framekm_path in unprocessed_framekm_debt:
            self.logger.info(f"Processing folder: {unprocessed_framekm_path}")
            framekm = unprocessed_framekm_path.split('/')[-1]  # glob stores full path, so fetch the name of the framekm
            self._process_folder(framekm, os.path.join(self.unprocessed_framekm_path, unprocessed_framekm_path))
            self.logger.debug(f"Done processing folder: {unprocessed_framekm_path}")

    def _process_folder(self, framekm: str, unprocessed_framekm_path: str):
        bin_full_framekm = os.path.join(unprocessed_framekm_path, f'bin_{framekm}') 
        frames = []
        frames_sizes = []
        total_processed_frame_size = 0
        for f in os.listdir(unprocessed_framekm_path):
            p = os.path.join(unprocessed_framekm_path, f)
            self.logger.debug(f'File name {p}')
            img = cv2.imread(p, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append((p, img))
        
        with open(f'{bin_full_framekm}', 'ab') as f:
            privacy_ml_metadata = ml_metadata.GenericMLMetadata()  # for all the frames in a packed framekm
            for j, val in enumerate(frames):
                img_id = val[0]
                img = val[1]
                img_ml_data = ml_metadata.MLFrameData()  # for all the boxes of an image
                
                boxes, scores, class_ids = self.onnx_detector(img, img_ml_data)
                
                if len(class_ids) > 0:
                    converted_box = yolov8.utils.xyxyxywh2(boxes)
                    for i, class_id in enumerate(class_ids):
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
                img_ml_data.name = f'{framekm}.json'

                privacy_ml_metadata.frame_data.append(img_ml_data)

        privacy_ml_metadata.model_hash = self.onnx_detector.model_hash
        self.write_ml_metadata(privacy_ml_metadata, framekm)
        self.move_and_cleanup_framekm(framekm, bin_full_framekm, unprocessed_framekm_path)
        self.move_and_cleanup_framekm_metadata(frames_sizes, framekm, total_processed_frame_size)
        

    def write_ml_metadata(self, privacy_ml_metadata: ml_metadata.GenericMLMetadata, framekm: str):
        metadata = ml_metadata.MLMetadata()
        metadata.privacy = privacy_ml_metadata
        with open(os.path.join(self.ml_metadata_path, framekm + '.json'), 'w') as f:
            f.write(metadata.toJson())

    def move_and_cleanup_framekm(self, framekm: str, bin_full_framekm: str, unprocessed_framekm_path: str):
        self.logger.info(f'Moving and cleaning up directories and files for {framekm}')
        framkm_name_orig_path = os.path.join(self.framekm_path, framekm)
        os.rename(bin_full_framekm, framkm_name_orig_path)
        self.logger.debug(f'Moved {bin_full_framekm} to {framkm_name_orig_path}')
        shutil.rmtree(unprocessed_framekm_path)
        self.logger.debug(f'Cleaned up {unprocessed_framekm_path}')
    
    def move_and_cleanup_framekm_metadata(self, frames_sizes: [], framekm: str, total_processed_frame_size: int):
        self.logger.debug(f'Frames size {len(frames_sizes)} for framekm_metadata')
        original_content = None
        unprocessed_metadata_framekm_path = os.path.join(self.unprocessed_metadata_path, framekm + '.json')
        processed_metadata_framekm_path = os.path.join(self.metadata_path, framekm + '.json')

        with open(unprocessed_metadata_framekm_path, 'r') as f:
            original_content = json.loads(f.read())
            original_content['bundle']['size'] = total_processed_frame_size
            for i, size in enumerate(frames_sizes):
                original_content['frames'][i]['bytes'] = size
        
        with open(processed_metadata_framekm_path, 'w') as f:
            f.write(str(original_content))
        os.remove(unprocessed_metadata_framekm_path)
