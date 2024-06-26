
import copy
import numpy as np

from .predict_rec import *
from .predict_det import *
from .predict_cls import *
from .utility import (
    get_rotate_crop_image,
    get_minarea_rect_crop,
    slice_generator,
    merge_fragmented,
)
def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

class TextSystem(object):
    def __init__(self, args):

        self.text_detector = TextDetector(args)
        self.text_recognizer = TextRecognizer(args)
        self.drop_score = args.drop_score
        self.text_classifier = TextClassifier(args)

        self.args = args

    def __call__(self, img, cls=True, slice={}):

        if img is None:
            return None, None

        ori_im = img.copy()
        if slice:
            slice_gen = slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )

            dt_slice_boxes = []
            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)

            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )

        else:
            dt_boxes = self.text_detector(img)

        if dt_boxes is None:
            return None, None

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)

        rec_res = self.text_recognizer(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res

