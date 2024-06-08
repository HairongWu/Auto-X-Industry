import logging
import magic 
import os
from uuid import uuid4
from typing import List, Dict, Optional
from auto_x_ml.model import AutoXMLBase

from auto_x_ml.modules.visual_pipeline import *

logger = logging.getLogger(__name__)

# LABEL_STUDIO_ACCESS_TOKEN = (
#         ''
# )
# LABEL_STUDIO_HOST = (
#         'http://127.0.0.1:8080/'
# )

vis_sol = VisualSolution()

class AutoSolution(AutoXMLBase):
    """
    """

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
                        
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        final_predictions = []
        image_paths = []
        video_paths = []
        for task in tasks:
              
            raw_img_path = task['data']['image']

            try:
                img_path = self.get_local_path(
                    raw_img_path,
                    # ls_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    #ls_host=LABEL_STUDIO_HOST,
                    task_id=task.get('id')
                )
                # printing the mime type of the file 
                mime = magic.from_file(img_path, mime = True)
                print(mime)

                if 'image/' in mime:
                    image_paths.append(img_path)

                elif 'video/' in mime:
                    video_paths.append(img_path)
                elif 'audio/' in mime:
                    pass
                elif mime == 'application/pdf':
                    pass
            except Exception as e:
                logger.error(f"Error getting local path: {e}")
                img_path = raw_img_path
        if len(image_paths) > 0:
            from_name_r, to_name_r, from_name_k, to_name_k = '','','',''
            for k, v in self.parsed_label_config.items():
                if v['type'] == 'KeyPointLabels':
                    from_name_k = k
                    to_name_k = v['to_name'][0]
                    labels_k = v['labels']
                if v['type'] == 'RectangleLabels':
                    from_name_r = k
                    to_name_r = v['to_name'][0]
                    labels_r = v['labels']
                    
            final_predictions.append(self.multiple_detection_tasks(image_paths, from_name_r, to_name_r, labels_r, from_name_k, to_name_k))

        return final_predictions
    
    def multiple_detection_tasks(self, image_paths, from_name_r, to_name_r, labels_r, from_name_k, to_name_k):
        
        predictions = []

        all_keypoints, all_boxes, all_labels, all_logits, all_lengths = vis_sol.run_keypoints(image_paths, labels_r)
        
        for points, (H, W) in zip(all_keypoints, all_lengths):            
            predictions.extend(self.get_keypoint_results(points, (H, W), from_name_k, to_name_k))

        for boxes_xyxy, label, logits, (H, W) in zip(all_boxes, all_labels, all_logits, all_lengths):                 
            predictions.extend(self.get_detection_results(boxes_xyxy, label, logits, (H, W), from_name_r, to_name_r))

        all_boxes, all_labels, all_logits, all_lengths = vis_sol.run_detection(image_paths)

        for boxes_xyxy, label, logits, (H, W) in zip(all_boxes, all_labels, all_logits, all_lengths):                 
            predictions.extend(self.get_detection_results(boxes_xyxy, label, logits, (H, W), from_name_r, to_name_r))

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }

    
    def get_detection_results(self, all_points, all_labels, all_scores, all_lengths, from_name_r, to_name_r):
        

        results = []
        height, width = all_lengths
        for points, score, label in zip(all_points, all_scores, all_labels):
            # random ID
            label_id = str(uuid4())[:9]
            results.append({
                'id': label_id,
                'from_name': from_name_r,
                'to_name': to_name_r,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    "rectanglelabels": [label],
                    'rotation': 0,
                    'width': (points[2] - points[0]) / width * 100,
                    'height': (points[3] - points[1]) / height * 100,
                    'x': points[0] / width * 100,
                    'y': points[1] / height * 100
                },
                'score': float(score),
                'type': 'rectanglelabels',
                'readonly': False
            })

        return results
    
    def get_keypoint_results(self, points, lengths, from_name_k, to_name_k):
        
        results = []
        height, width = lengths

        for point in points:
            
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:9]

            results.append({
                'id': label_id,
                'from_name': from_name_k,
                'to_name': to_name_k,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'x': point[0] / width * 100,
                    'y': point[1]/ height * 100,
                    'width': 0.1,
                    'labels': [point[2]],
                    'keypointlabels': [point[2]]
                },
                'score': 1.0,
                'readonly': False,
                'type': 'keypointlabels'
            })
        
        return results