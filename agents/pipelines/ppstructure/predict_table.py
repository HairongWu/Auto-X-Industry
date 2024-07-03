import copy
import numpy as np
from .predict_rec import *
from .predict_det import *
from .utility import *
from .predict_system import sorted_boxes
from .matcher import TableMatch
from .table_master_match import TableMasterMatcher
from .predict_structure import *


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class TableSystem(object):
    def __init__(self, args, text_detector=None, text_recognizer=None):
        self.args = args

        self.text_detector = (
            TextDetector(copy.deepcopy(args))
            if text_detector is None
            else text_detector
        )
        self.text_recognizer = (
            TextRecognizer(copy.deepcopy(args))
            if text_recognizer is None
            else text_recognizer
        )

        self.table_structurer = TableStructurer(args)
        if args.table_algorithm in ["TableMaster"]:
            self.match = TableMasterMatcher()
        else:
            self.match = TableMatch(filter_ocr_result=True)

        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(args, "table")

    def __call__(self, img, return_ocr_result_in_table=False):
        result = dict()

        structure_res = self._structure(copy.deepcopy(img))
        print(structure_res)
        result["cell_bbox"] = structure_res[1].tolist()

        dt_boxes, rec_res = self._ocr(copy.deepcopy(img))

        if return_ocr_result_in_table:
            result["boxes"] = [x.tolist() for x in dt_boxes]
            result["rec_res"] = rec_res

        pred_html = self.match(structure_res, dt_boxes, rec_res)

        result["html"] = pred_html

        return result

    def _structure(self, img):
        structure_res = self.table_structurer(copy.deepcopy(img))
        return structure_res

    def _ocr(self, img):
        h, w = img.shape[:2]
        dt_boxes = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)

        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0) : int(y1), int(x0) : int(x1), :]
            img_crop_list.append(text_rect)
        rec_res = self.text_recognizer(img_crop_list)
        return dt_boxes, rec_res

