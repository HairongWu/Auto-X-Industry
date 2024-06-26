import cv2
import copy
import numpy as np
import math

from .utility import *
from .ppocr.postprocess import build_post_process


class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        postprocess_params = {
            "name": "ClsPostProcess",
            "label_list": args.label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            _,
        ) = create_predictor(args, "cls")
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [["", 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0

            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            if self.use_onnx:
                input_dict = {}
                input_dict[self.input_tensor.name] = norm_img_batch
                outputs = self.predictor.run(self.output_tensors, input_dict)
                prob_out = outputs[0]
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()
                prob_out = self.output_tensors[0].copy_to_cpu()
                self.predictor.try_shrink_memory()
            cls_result = self.postprocess_op(prob_out)

            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return img_list, cls_res, elapse
