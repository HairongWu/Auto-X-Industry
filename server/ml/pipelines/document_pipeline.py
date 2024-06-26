import os
import cv2

from .ppstructure.predict_system import TextSystem
from .ppstructure.predict_layout import LayoutPredictor
from .ppstructure.predict_table import TableSystem
from .ppstructure.utility import cal_ocr_word_box
import paddleclas

class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

class OCRPipeline(): 
    def __init__(self):
        super().__init__()
        args = AttrDict()
        
        args['det_model_dir'] = os.environ.get('det_model_dir')
        args['cls_model_dir'] = os.environ.get('cls_model_dir')
        args['rec_model_dir'] = os.environ.get('rec_model_dir')
        args['table_model_dir'] = os.environ.get('table_model_dir')
        args['layout_model_dir'] = os.environ.get('layout_model_dir')

        args['layout_dict_path'] = "./ml/pipelines/utils/dict/layout_dict/layout_publaynet_dict.txt"
        args['rec_char_dict_path'] = "./ml/pipelines/utils/ppocr_keys_v1.txt"
        args['table_char_dict_path'] = "./ml/pipelines/utils/dict/table_structure_dict_ch.txt"

        args['layout_score_threshold'] = 0.5
        args['layout_nms_threshold'] = 0.5
        args['use_onnx'] = False
        args['drop_score'] = 0.5
        args['use_gpu'] = True
        args['gpu_mem'] = 500
        args['gpu_id'] = 0
        args['return_word_box'] = False

        args['det_box_type'] = 'quad'
        args['det_algorithm'] = "DB"
        args['det_limit_side_len'] = 960
        args['det_limit_type'] = 'max'

        args['det_db_thresh'] = 0.3
        args['det_db_box_thresh'] = 0.6
        args['det_db_unclip_ratio'] = 1.5
        args['use_dilation'] = False
        args['det_db_score_mode'] = 'fast'

        args['det_east_score_thresh'] = 0.8
        args['det_east_cover_thresh'] = 0.1
        args['det_east_nms_thresh'] = 0.2

        args['det_sast_score_thresh'] = 0.5
        args['det_sast_nms_thresh'] = 0.2

        args['det_pse_thresh'] = 0
        args['det_pse_box_thresh'] = 0.85
        args['det_pse_min_area'] = 16
        args['det_pse_scale'] = 1

        args['scales'] = [8, 16, 32]
        args['alpha'] = 1.0
        args['beta'] = 1.0
        args['fourier_degree'] = 5

        args['rec_algorithm'] = "SVTR_LCNet"
        args['rec_image_shape'] = "3, 48, 320"
        args['rec_batch_num'] = 6
        args['use_space_char'] = True
        args['max_text_length'] = 25
        args['rec_image_inverse'] = True

        args['cls_image_shape'] = "3, 48, 192"
        args['cls_batch_num'] = 6
        args['cls_thresh'] = 0.9
        args['label_list'] = ["0", "180"]

        args['table_algorithm'] = "TableAttn"
        args['merge_no_span_structure'] = True
        args['table_max_len'] = 488

        self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation"
            )
        self.layout_predictor = LayoutPredictor(args)
        self.text_system = TextSystem(args)

        self.table_system = TableSystem(
                        args,
                        self.text_system.text_detector,
                        self.text_system.text_recognizer,
                    )
        self.return_word_box = args.return_word_box

    def run_ocr(self, image_paths):
        # Fix me: Change to batch inferences
        results = []
        for img in image_paths:
            img = cv2.imread(img)
            cls_result = self.image_orientation_predictor.predict(input_data=img)
            cls_res = next(cls_result)
            angle = cls_res[0]["label_names"][0]

            cv_rotate_code = {
                "90": cv2.ROTATE_90_COUNTERCLOCKWISE,
                "180": cv2.ROTATE_180,
                "270": cv2.ROTATE_90_CLOCKWISE,
            }
            if angle in cv_rotate_code:
                img = cv2.rotate(img, cv_rotate_code[angle])

            ori_im = img.copy()
            h, w = ori_im.shape[:2]
            layout_res = self.layout_predictor(img)
            text_res = self._predict_text(img)
            res_list = []
            for region in layout_res:
                res = ""
                if region["bbox"] is not None:
                    x1, y1, x2, y2 = region["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                bbox = [x1, y1, x2, y2]

                if region["label"] == "table":
                    if self.table_system is not None:
                        res = self.table_system(
                            roi_img, False
                        )
                else:
                    if text_res is not None:
                        # Filter the text results whose regions intersect with the current layout bbox.
                        res = self._filter_text_res(text_res, bbox)

                res_list.append(
                    {
                        "type": region["label"].lower(),
                        "bbox": bbox,
                        "res": res,
                    }
                )
            results.append([res_list,(h,w), angle])
        return results

    def _predict_text(self, img):
        filter_boxes, filter_rec_res = self.text_system(img)

        # remove style char,
        # when using the recognition model trained on the PubtabNet dataset,
        # it will recognize the text format in the table, such as <b>
        style_token = [
            "<strike>",
            "<strike>",
            "<sup>",
            "</sub>",
            "<b>",
            "</b>",
            "<sub>",
            "</sup>",
            "<overline>",
            "</overline>",
            "<underline>",
            "</underline>",
            "<i>",
            "</i>",
        ]
        res = []
        for box, rec_res in zip(filter_boxes, filter_rec_res):
            rec_str, rec_conf = rec_res[0], rec_res[1]
            for token in style_token:
                if token in rec_str:
                    rec_str = rec_str.replace(token, "")
            if self.return_word_box:
                word_box_content_list, word_box_list = cal_ocr_word_box(
                    rec_str, box, rec_res[2]
                )
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                        "text_word": word_box_content_list,
                        "text_word_region": word_box_list,
                    }
                )
            else:
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                    }
                )
        return res
    
    def _filter_text_res(self, text_res, bbox):
        res = []
        for r in text_res:
            box = r["text_region"]
            rect = box[0][0], box[0][1], box[2][0], box[2][1]
            if self._has_intersection(bbox, rect):
                res.append(r)
        return res
               
    def _has_intersection(self, rect1, rect2):
        x_min1, y_min1, x_max1, y_max1 = rect1
        x_min2, y_min2, x_max2, y_max2 = rect2
        if x_min1 > x_max2 or x_max1 < x_min2:
            return False
        if y_min1 > y_max2 or y_max1 < y_min2:
            return False
        return True    

    def trainer(self, data):
        pass

    def eval(self, data):
        pass

    def export(self, data):
        pass

    def train_ocr(self, data):
        self.trainer(data, "ppstructure/configs/det/det_mv3_db.yml")
        self.eval(data, "ppstructure/configs/det/det_mv3_db.yml")
        self.export(data, "ppstructure/configs/det/det_mv3_db.yml")

        self.trainer(data, "ppstructure/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml")
        self.eval(data, "ppstructure/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml")
        self.export(data, "ppstructure/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml")

        self.trainer(data, "ppstructure/configs/cls/cls_mv3.yml")
        self.eval(data, "ppstructure/configs/cls/cls_mv3.yml")
        self.export(data, "ppstructure/configs/cls/cls_mv3.yml")

        self.trainer(data, "ppstructure/configs/layout_analysis/picodet_lcnet_x1_0_layout.yml")
        self.eval(data, "ppstructure/configs/layout_analysis/picodet_lcnet_x1_0_layout.yml")
        self.export(data, "ppstructure/configs/layout_analysis/picodet_lcnet_x1_0_layout.yml")

        self.trainer(data, "ppstructure/configs/table/SLANet.yml")
        self.eval(data, "ppstructure/configs/table/SLANet.yml")
        self.export(data, "ppstructure/configs/table/SLANet.yml")

        self.trainer(data, "ppstructure/configs/cls/cls_mv3.yml")
        self.eval(data, "ppstructure/configs/cls/cls_mv3.yml")
        self.export(data, "ppstructure/configs/cls/cls_mv3.yml")

        self.trainer(data, "ppstructure/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml")
        self.eval(data, "ppstructure/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml")
        self.export(data, "ppstructure/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml")

        self.trainer(data, "ppstructure/configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml")
        self.eval(data, "ppstructure/configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml")
        self.export(data, "ppstructure/configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml")