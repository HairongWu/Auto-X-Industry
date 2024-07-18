import magic 
from uuid import uuid4
from PIL import Image
from typing import Union, List, Dict, Optional, Any, Tuple
import difflib

from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag

from server.model import *

from pipelines.lspr_pipeline import *
from pipelines.document_pipeline import *
from pipelines.video_pipeline import *
from pipelines.llm_pipeline import *

lspr_pipeline = LSPRPipeline()
#video_pipeline = VideoPipeline()
#llm_pipeline = LLMPipeline()

class AutoSolution(AutoXMLBase):

    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs")

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def _find_textarea_tag(self, prompt_tag, object_tag):
        """Free-form text predictor
        """
        li = self.label_interface

        try:
            textarea_from_name, _, _ = li.get_first_tag_occurence(
                'TextArea',
                self.SUPPORTED_INPUTS,
                name_filter=lambda s: s != prompt_tag.name,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(textarea_from_name)
        except:
            return None

    def _find_prompt_tags(self) -> Tuple[ControlTag, ObjectTag]:
        """Find prompting tags in the config
        """
        li = self.label_interface
        prompt_from_name, prompt_to_name, value = li.get_first_tag_occurence(
            # prompt tag
            self.PROMPT_TAG,
            # supported input types
            self.SUPPORTED_INPUTS,
            # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))

        return li.get_control(prompt_from_name), li.get_object(prompt_to_name)
                                        
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        print("tasks", tasks)
        print("context", context)
        final_predictions = []
        lspr_tasks = []
        doc_tasks = []
        video_tasks = []

        for task in tasks:
            try:
                if 'lspr' in task['data']:
                    raw_img_path = task['data']['lspr']
                    img_path = self.get_local_path(
                        raw_img_path,
                        task_id=task.get('id')
                    )
                    # printing the mime type of the file 
                    mime = magic.from_file(img_path, mime = True)
                    print(mime)

                    if 'image/' in mime:
                        lspr_tasks.append(img_path)

                elif 'doc' in task['data']:
                    raw_img_path = task['data']['doc']
                    img_path = self.get_local_path(
                        raw_img_path,
                        task_id=task.get('id')
                    )
                    # printing the mime type of the file 
                    mime = magic.from_file(img_path, mime = True)
                    print(mime)

                    if 'image/' in mime:
                        doc_tasks.append(img_path)
                    elif mime == 'application/pdf':
                        doc_tasks.append(img_path)
                elif 'video_url' in task['data']:
                    raw_img_path = task['data']['video_url']
                    img_path = self.get_local_path(
                        raw_img_path,
                        task_id=task.get('id')
                    )
                    # printing the mime type of the file 
                    mime = magic.from_file(img_path, mime = True)
                    print(mime)

                    if 'video/' in mime:
                        video_tasks.append(img_path)
                elif 'dialogue' in task['data']:
                    # prompt tag contains the prompt in the config
                    # object tag contains what we plan to label
                    prompt_tag, object_tag = self._find_prompt_tags()
                    textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)
                    history = task['data']['dialogue']

                    prompt = []
                    if 'annotations' in task:
                        for item in task['annotations']:
                            result = item['result'][0]
                            if result['from_name'] == prompt_tag.name:
                                prompt.extend(result['value']['text'])

                    pred = self.llm_task(prompt, prompt_tag, history,
                                                                textarea_tag)
                    final_predictions.append(pred)               

            except Exception as e:
                print(f"Error getting local path: {e}")
        
        if len(lspr_tasks) > 0:
            final_predictions.append(self.multiple_lspr_tasks(lspr_tasks))
        if len(doc_tasks) > 0:
            final_predictions.append(self.multiple_doc_tasks(doc_tasks))
        if len(video_tasks) > 0:
            final_predictions.append(self.multiple_video_tasks(video_tasks))
        return final_predictions
    
    def llm_task(self, prompt: List, prompt_tag: Any, history: str,
                             textarea_tag: ControlTag) -> Dict:
        """
        """
        # run inference
        # this are params provided through the web interface
        response = llm_pipeline.predict(prompt, history)
         # random ID
        regions = []
        if textarea_tag:
            regions.append(textarea_tag.label(text=response))
        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompt))
        
        return PredictionValue(result=regions, score=0, model_version=str(self.model_version)).model_dump()
    
    def multiple_lspr_tasks(self, image_paths):

        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self.get(prompt_tag.name)
        print('prmpts',prompts)
        li = self.label_interface
        from_name_r, to_name_r, value = li.get_first_tag_occurence('RectangleLabels', 'Image')
        
        predictions = []
        all_boxes, all_labels, all_logits, all_lengths = lspr_pipeline.predict(image_paths, prompts)
        update_labels = {}

        for boxes_xyxy, label, logits, (H, W) in zip(all_boxes, all_labels, all_logits, all_lengths):                 
            height, width = (H, W)
            for box, label, score in zip(boxes_xyxy, label, logits):
                if label not in update_labels:
                    update_labels[label] = 1
                # random ID
                label_id = str(uuid4())[:9]
                predictions.append({
                        'id': label_id,
                        'from_name': from_name_r,
                        'to_name': to_name_r,
                        'original_width': width,
                        'original_height': height,
                        'image_rotation': 0,
                        'value': {
                            'rotation': 0,
                            'width': (box[2] - box[0]) / width * 100,
                            'height': (box[3] - box[1]) / height * 100,
                            'x': box[0] / width * 100,
                            'y': box[1] / height * 100,
                            "rectanglelabels": [label],
                        },
                        'score': float(score),
                        'type': 'rectanglelabels',
                        'readonly': False
                    })
        
        self.add_new_labels(from_name_r, update_labels, tag_type='RectangleLabels')

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }    

    
    def multiple_doc_tasks(self, image_paths):
        li = self.label_interface
        from_name_r, to_name, value = li.get_first_tag_occurence('RectangleLabels', 'Image')
        from_name_t, to_name, value = li.get_first_tag_occurence('TextArea', 'Image')
        from_name_rot, to_name, value = li.get_first_tag_occurence('Choices', 'Image')
            
        predictions = []
        doc_pipeline = DocumentPipeline(from_name_t)
        res_list = doc_pipeline.predict(image_paths)
        
        for res in res_list:                 
            predictions.extend(self.get_ocr_results(res, from_name_r, from_name_t, from_name_rot, to_name))

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }    

    def get_ocr_results(self, res, from_name_r, from_name_t, from_name_rot, to_name):
        results = []
        height, width = res[2]

        results.append({
          "from_name": from_name_rot,
          "to_name": to_name,
          "type": "choices",
          "value": { "choices": [res[3]] }
        })
        for rs in res[0]:
            if len(rs['text'].strip()) > 0:
                label_id = str(uuid4())[:9]
                points = rs['text_region']
                results.append({
                    "original_width": width,
                    "original_height": height,
                    "value": {
                        "rotation": 0,
                        'width': (points[2] - points[0]) / width * 100,
                        'height': (points[3] - points[1]) / height * 100,
                        'x': points[0] / width * 100,
                        'y': points[1] / height * 100,
                        "text": [
                        rs['text'].strip()
                        ]
                    },
                    "id": label_id,
                    "from_name": from_name_t,
                    "to_name": to_name,
                    "type": "textarea",
                    'score': float(rs['confidence']),
                    "origin": "manual",
                })
                results.append({
                    'id': label_id,
                    'from_name': from_name_r,
                    'to_name': to_name,
                    'original_width': width,
                    'original_height': height,
                    'value': {
                        "rectanglelabels": ['Text'],
                        'width': (points[2] - points[0]) / width * 100,
                        'height': (points[3] - points[1]) / height * 100,
                        'x': points[0] / width * 100,
                        'y': points[1] / height * 100
                    },
                    'type': 'rectanglelabels',
                })
        update_labels = {}
        for rs in res[1]:
            rect_type = rs['type'].strip().lower()
            if rect_type not in update_labels:
                    update_labels[rect_type] = 1
            # random ID
            label_id = str(uuid4())[:9]
            points = [rs['box'][0][0], rs['box'][0][1], rs['box'][2][0], rs['box'][2][1]]
            results.append({
                'id': label_id,
                'from_name': from_name_r,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'value': {
                    "rectanglelabels": [rect_type],
                    'width': (points[2] - points[0]) / width * 100,
                    'height': (points[3] - points[1]) / height * 100,
                    'x': points[0] / width * 100,
                    'y': points[1] / height * 100
                },
                'type': 'rectanglelabels',
            })
        self.add_new_labels(from_name_r, update_labels, tag_type='RectangleLabels')
        return results

    def multiple_video_tasks(self, image_paths, from_name_p, from_name_r,to_name, prompt):
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self.get(prompt_tag.name)
        print('prmpts',prompts)
        li = self.label_interface
        from_name_t, to_name, value = li.get_first_tag_occurence('TextArea', 'Image', name_filter=lambda s: s == "response")

        predictions = []
        if len(prompt) > 0:
            res_list = video_pipeline.predict(image_paths, prompt)
            print(res_list)
            
            for res in res_list:                 
                predictions.extend(self.get_video_results(res, from_name_t, to_name))

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }    

    def get_video_results(self, res, from_name_t, to_name):
        results = []
        for rs in res:
            # random ID
            label_id = str(uuid4())[:9]
            results.append({
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "text": [
                        rs[0]
                    ]
                },
                "id": label_id,
                "from_name": from_name_t,
                "to_name": to_name,
                "type": "textarea",
                "origin": "manual"
            })

        return results
      
    def train(self, annotations, **kwargs):
        odvg_annos = []
        names = []
        for annos in annotations:
            for anno in annos['annotations']:
                if len(anno['result']) > 0:
                    names.append(anno['result'][0]['value']['text'][0].strip())
        names = list(set(names))
        for annos in annotations:
            if 'lspr' in annos['data']:
                raw_img_path = annos['data']['lspr']
                img_path = self.get_local_path(
                        raw_img_path,
                        task_id=annos['id']
                )
                image_source = Image.open(img_path)
                width, height = image_source.size 
                
                for anno in annos['annotations']:
                    if len(anno['result']) > 0:
                        ret = {"im_file": img_path,
                            "shape": (height,width),
                            "bboxes":[],
                            "cls":[],
                            "bbox_format":"xyxy",
                            "normalized":True}
                        x1 = anno['result'][0]['value']['x']
                        y1 = anno['result'][0]['value']['y']
                        x2 = x1 + anno['result'][0]['value']['width']
                        y2 = y1 + anno['result'][0]['value']['height']
                        ret["bboxes"].append([x1*width/100, y1*height/100, 
                                            x2*width/100, y2*height/100])
                        label = names.index(anno['result'][0]['value']['text'][0].strip())
                        ret["cls"].append([label])
                        odvg_annos.append(ret)
        # Connect to training server
        # train(odvg_annos, names)

    def _prompt_diff(self, old_prompt, new_prompt):
        """
        """
        old_lines = old_prompt.splitlines()
        new_lines = new_prompt.splitlines()
        diff = difflib.unified_diff(old_lines, new_lines, lineterm="")

        return "\n".join(
            line for line in diff if line.startswith(('+',)) and not line.startswith(('+++', '---')))
    
    def fit(self, event, data, **additional_params):
        """
        """
        print(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        # prompt_tag, object_tag = self._find_prompt_tags()
        # prompts = self._get_prompts(data['annotation'], prompt_tag)

        # if len(prompts) < 1:
        #     print(f'No prompts recorded.')
        #     return

        # prompt = ','.join(prompts)
        # current_prompt = self.get(prompt_tag.name)

        # # find substrings that differ between current and new prompt
        # # if there are no differences, skip training
        # if current_prompt:
        #     diff = self._prompt_diff(current_prompt, prompt)
        #     if not diff:
        #         print('No prompt diff found.')
        #         return

        #     print(f'Prompt diff: {diff}')
        # self.set(prompt_tag.name, prompt)
        # model_version = self.bump_model_version()

        # print(f'Updated model version to {str(model_version)}')