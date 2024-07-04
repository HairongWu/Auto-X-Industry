import os
import cv2

from .ppstructure.predict_system import TextSystem
import paddleclas

from cnstd import LayoutAnalyzer

from .pipeline import *

class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

class DocumentPipeline(Pipeline): 
    def __init__(self, lang):
        super().__init__()
        args = AttrDict()
        
        args['det_model_dir'] = os.environ.get('det_model_dir')
        args['cls_model_dir'] = os.environ.get('cls_model_dir')
        args['rec_model_dir'] = os.environ.get(lang+'_rec_model_dir')

        lang_dict = {"zh":"./pipelines/utils/ppocr_keys_v1.txt","jp":"./pipelines/utils/dict/japan_dict.txt"}
        args['rec_char_dict_path'] = lang_dict[lang]

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

        self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation"
            )

        self.text_system = TextSystem(args)

        self.analyzer = LayoutAnalyzer('layout')

    def predict(self, image_paths):
        results = []

        for image_file in image_paths:
            img = cv2.imread(image_file)

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

            h, w = img.shape[:2]
            filter_boxes, filter_rec_res = self.text_system(img)
            res = []
            for box, rec_res in zip(filter_boxes, filter_rec_res):
                rec_str, rec_conf = rec_res[0], rec_res[1]
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": [box[0][0], box[0][1],box[2][0], box[2][1]],
                    }
                )

            # Predict on image
            layout_res = self.analyzer.analyze(image_file, resized_shape=704)

            results.append([res, layout_res, (h,w), angle])
        return results
