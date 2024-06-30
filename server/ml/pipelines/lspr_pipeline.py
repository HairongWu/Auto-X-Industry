import os
from PIL import Image
from sklearn.model_selection import train_test_split
from pymilvus import MilvusClient
import functools
from collections import OrderedDict

from .pipeline import *
from .ultralytics import YOLOv10

class LSPRPipeline(Pipeline): 
    def __init__(self):
        super().__init__()

        self.model = YOLOv10.from_pretrained('jameslahm/yolov10x',)

        args = AttrDict()

    def predict(self, image_paths):
        # Fix me: Change to batch inferences
        all_boxes = []
        all_labels = []
        all_logits = []
        all_lengths = []

        for img in image_paths:
            boxes_filt = self.model.predict(source=img, save=False)[0]
            boxes = boxes_filt.boxes
            labels = boxes_filt.names
            cls = boxes.cls.cpu().numpy()
            cls = [labels[c] for c in cls]

            boxes, cls = self.get_from_vdb(img, boxes, cls)

            all_boxes.append(boxes.xyxy.cpu().numpy())
            all_lengths.append(boxes.orig_shape)
            all_logits.append(boxes.conf.cpu().numpy())
            all_labels.append(cls)
            
        return all_boxes, all_labels, all_logits, all_lengths

    def train(self, annotations, names):
        self.save_to_vdb(annotations)

        train, val = train_test_split(annotations, test_size=0.4)
        val, test = train_test_split(val, test_size=0.5)

        data = {"train":train,"val":val, "nc":len(names), "names":names}
        self.model.train(data=data, epochs=500, batch=8, imgsz=320, save_dir=os.environ.get("MODEL_POOL_DIR") + "detection")
        # data = {"test":test}
        # self.model.val(data=data, batch=256)

    def finetune(self, annotations):
        pass

    def get_from_vdb(self, image, boxes, cls):
        return boxes, cls

    def save_to_vdb(self, annotations):
        pass