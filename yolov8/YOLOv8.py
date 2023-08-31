import time
import cv2
import numpy as np
import onnxruntime
import yolov8.constant as constant
import yolov8.ml_metadata as ml_metadata
import logging
import os

from yolov8.utils import xywh2xyxy, xyxyxywh2, nms
from yolov8.draw import draw_detections


class YOLOv8:
    input_height = int
    input_width = int
    logger: logging.Logger
    show_detections: bool
    model_hash: str

    def __init__(self, path, logger: logging.Logger, input_height: int, input_width: int, show_detections: bool, model_hash_path: str, conf_thres=0.7, iou_thres=0.5):
        self.logger = logger
        self.input_height = input_height
        self.input_width = input_width
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.show_detections = show_detections

        if not os.path.exists(model_hash_path):
            raise FileNotFoundError

        with open(model_hash_path) as f:
            self.model_hash = f.readlines()[0]

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image, img_ml_data):
        return self.detect_objects(image, img_ml_data)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # sess_options = onnxruntime.SessionOptions()
        # print("number of threads: " + str(sess_options.intra_op_num_threads))
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image, ml_frame_data: ml_metadata.MLFrameData):
        input_tensor = self.prepare_input(image, ml_frame_data)

        # Perform inference on the image
        outputs = self.inference(input_tensor, ml_frame_data)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs, ml_frame_data)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image, ml_frame_data: ml_metadata.MLFrameData):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        start = time.perf_counter()
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        resize_time = (time.perf_counter() - start) * 1000
        ml_frame_data.set_resizing_time(resize_time)
        self.logger.debug(f'Resizing time: {resize_time}ms')

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor, ml_frame_data: ml_metadata.MLFrameData):
        start = time.perf_counter()

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        inf_time = (time.perf_counter() - start) * 1000
        ml_frame_data.set_inference_time(inf_time)
        self.logger.info(f'Inference time: {inf_time}ms')

        return outputs

    def process_output(self, output, ml_frame_data: ml_metadata.MLFrameData):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        if self.show_detections:
            return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

        return image

    def blur_boxes(self, img, ml_frame_data: ml_metadata.MLFrameData):
        start = time.perf_counter()
        for i, class_id in enumerate(self.class_ids):
            detected_label = constant.CLASS_NAMES[class_id]
            
            if detected_label == 'license-plate' or detected_label == 'face':
                h, w = img.shape[:2]
                kernel_width = (w // 7) | 1
                kernel_height = (h // 7) | 1

                box = self.boxes[i]
                start_x, start_y, end_x, end_y = box.astype(np.int64)
                detected_label = img[start_y: end_y, start_x: end_x]
                try: 
                    blurred_detected_label = cv2.GaussianBlur(detected_label, (kernel_width, kernel_height), 0)
                    img[start_y: end_y, start_x: end_x] = blurred_detected_label
                except:
                    self.logger.info("exception occurred when blurring boxes, ignoring...")
            

        blurring_time = (time.perf_counter() - start) * 1000
        ml_frame_data.set_blurring_time(blurring_time)
        self.logger.debug(f'Blurring time: {blurring_time}ms')

        return img

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        # self.input_height = self.input_shape[2]
        # self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
