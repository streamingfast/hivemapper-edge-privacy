import json


class BoundingBox:
    class_id: str
    confidence: float
    cx: float
    cy: float
    width: float
    height: float

    def __init__(self):
        self.class_id = ''
        self.confidence = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.width = 0.0
        self.height = 0.0
    
    def set_class_id(self, class_id):
        self.class_id = str(class_id)
    
    def set_confidence(self, confidence):
        self.confidence = float(confidence)
    
    def set_cxcywh(self, box):
        self.cx = float(box[0])
        self.cy = float(box[1])
        self.width = float(box[2])
        self.height = float(box[3])


class MLFrameData:
    resizing_time: float
    inference_time: float
    blurring_time: float
    detections: [BoundingBox]
    img_id: str
    name: str  # the name of the json that this framekm is pointing to

    def __init__(self):
        self.resizing_time = 0.0
        self.inference_time = 0.0
        self.blurring_time = 0.0
        self.detections = []
        self.img_id = ''
        self.name = ''

    def set_resizing_time(self, resizing_time):
        self.resizing_time = float(resizing_time)
    
    def set_inference_time(self, inference_time):
        self.inference_time = float(inference_time)
    
    def set_blurring_time(self, blurring_time):
        self.blurring_time = float(blurring_time)


class GenericMLMetadata:
    model_hash: str
    frame_data: [MLFrameData]

    def __init__(self):
        self.model_hash = ''
        self.frame_data = []

    def set_model_hash(self, model_hash):
        self.model_hash = str(model_hash)


class MLMetadata:
    privacy: GenericMLMetadata

    def __init__(self):
        self.privacy = None

    def set_privacy(self, privacy):
        self.privacy = privacy

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)
