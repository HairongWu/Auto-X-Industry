import magic 
from uuid import uuid4
from PIL import Image
from typing import Union, List, Dict, Optional, Any, Tuple
import difflib

from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag

from server.model import AutoXMLBase

from pipelines.lspr_pipeline import *
from pipelines.document_pipeline import *
from pipelines.video_pipeline import *
from pipelines.llm_pipeline import *

#lspr_pipeline = LSPRPipeline()
#doc_pipeline = DocumentPipeline()
# video_pipeline = VideoPipeline()
llm_pipeline = LLMPipeline()

class AutoSolution(AutoXMLBase):
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs","Video")
    DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

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

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        """Getting prompt values
        """
        if context:
            # Interactive mode - get prompt from context
            result = context.get('result')
            for item in result:
                if item.get('from_name') == prompt_tag.name:
                    return item['value']['text']
        # Initializing - get existing prompt from storage
        elif prompt := self.get(prompt_tag.name):
            return [prompt]
        # Default prompt
        elif self.DEFAULT_PROMPT:
            if self.USE_INTERNAL_PROMPT_TEMPLATE:
                print('Using both `DEFAULT_PROMPT` and `USE_INTERNAL_PROMPT_TEMPLATE` is not supported. '
                             'Please either specify `USE_INTERNAL_PROMPT_TEMPLATE=0` or remove `DEFAULT_PROMPT`. '
                             'For now, no prompt will be used.')
                return []
            return [self.DEFAULT_PROMPT]

        return []

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

    def _get_text(self, task_data, object_tag):
        """
        """
        data = task_data.get(object_tag.value_name)

        if data is None:
            return None

        if isinstance(object_tag, ParagraphsTag):
            return json.dumps(data)
        else:
            return data
        
    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict, labels: Optional[List[str]]) -> str:
        """
        """
        if self.USE_INTERNAL_PROMPT_TEMPLATE:
            norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt, labels=labels)
        else:
            norm_prompt = prompt.format(labels=labels, **task_data)

        return norm_prompt
                                         
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        final_predictions = []
        lspr_tasks = []
        doc_tasks = []
        video_tasks = []
        
        # prompt tag contains the prompt in the config
        # object tag contains what we plan to label
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(context, prompt_tag)
        

        if prompts:
            prompt = "\n".join(prompts)

            textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)

            for task in tasks:
                try:
                    raw_img_path = None
                    if 'video_url' in task['data']:
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
                        
                        if len(video_tasks) > 0:
                            from_name_p, to_name, _ = li.get_first_tag_occurence(
                                # prompt tag
                                'TextArea',
                                # supported input types
                                self.SUPPORTED_INPUTS,
                                # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
                                name_filter=lambda s: s.startswith('prompt'))
                            from_name_r, _, _ = li.get_first_tag_occurence(
                                # prompt tag
                                'TextArea',
                                # supported input types
                                self.SUPPORTED_INPUTS,
                                # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
                                name_filter=lambda s: s.startswith('response'))
                            
                            final_predictions.append(self.multiple_video_tasks(video_tasks, from_name_p, from_name_r,to_name, prompt))

                    else:
                        # preload all task data fields, they are needed for prompt
                        task_data = self.preload_task_data(task, task['data'])
                        pred = self.llm_task(task_data, prompt_tag, object_tag, prompt,
                                                        textarea_tag, prompts)
                        final_predictions.append(pred)                  
                        
                except Exception as e:
                    print(f"Error getting local path: {e}")
                
        else:
            for task in tasks:
                try:
                    raw_img_path = None
                    if 'lspr' in task['data']:
                        raw_img_path = task['data']['lspr']
                    if 'doc' in task['data']:
                        raw_img_path = task['data']['doc']
                
                    if raw_img_path is not None:
                        img_path = self.get_local_path(
                            raw_img_path,
                            task_id=task.get('id')
                        )
                        # printing the mime type of the file 
                        mime = magic.from_file(img_path, mime = True)
                        print(mime)

                        if 'image/' in mime:
                            if 'lspr' in task['data']:
                                lspr_tasks.append(img_path)
                            elif 'doc' in task['data']:
                                doc_tasks.append(img_path)
                        elif 'audio/' in mime:
                            pass
                        elif mime == 'application/pdf':
                            doc_tasks.append(img_path)
                except Exception as e:
                    print(f"Error getting local path: {e}")
            
            li = self.label_interface
            if len(lspr_tasks) > 0:
                from_name_t, to_name, value = li.get_first_tag_occurence(
                    # prompt tag
                    'TextArea',
                    # supported input types
                    self.SUPPORTED_INPUTS)
                final_predictions.append(self.multiple_lspr_tasks(lspr_tasks, from_name_t, to_name))
            if len(doc_tasks) > 0:
                from_name_r, to_name, _ = li.get_first_tag_occurence(
                    # prompt tag
                    'RectangleLabels',
                    # supported input types
                    self.SUPPORTED_INPUTS)
                from_name_t, _, _ = li.get_first_tag_occurence(
                    # prompt tag
                    'TextArea',
                    # supported input types
                    self.SUPPORTED_INPUTS)
                from_name_lang, _, _ = li.get_first_tag_occurence(
                    # prompt tag
                    'Choices',
                    # supported input types
                    self.SUPPORTED_INPUTS,
                    # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
                    name_filter=lambda s: s.startswith('lang'))
                from_name_type, _, _ = li.get_first_tag_occurence(
                    # prompt tag
                    'Choices',
                    # supported input types
                    self.SUPPORTED_INPUTS,
                    # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
                    name_filter=lambda s: s.startswith('type'))
                from_name_rot, _, _ = li.get_first_tag_occurence(
                    # prompt tag
                    'Choices',
                    # supported input types
                    self.SUPPORTED_INPUTS,
                    # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
                    name_filter=lambda s: s.startswith('rotation'))
                
                final_predictions.append(self.multiple_doc_tasks(doc_tasks, from_name_r, from_name_t, from_name_lang, from_name_type, from_name_rot, to_name))
            
       
        return final_predictions
    
    def llm_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str,
                             textarea_tag: ControlTag, prompts: List[str]) -> Dict:
        """
        """
        text = self._get_text(task_data, object_tag)
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data, labels=None)

        # run inference
        # this are params provided through the web interface
        response = llm_pipeline.predict(norm_prompt)
         # random ID
        regions = []
        if textarea_tag:
            regions.append(textarea_tag.label(text=[response]))
        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))
        
        return PredictionValue(result=regions, score=0.1, model_version=str(self.model_version)).model_dump()
    
    def multiple_lspr_tasks(self, image_paths, from_name_t, to_name):
        
        predictions = []

        all_boxes, all_labels, all_logits, all_lengths = lspr_pipeline.predict(image_paths)

        for boxes_xyxy, label, logits, (H, W) in zip(all_boxes, all_labels, all_logits, all_lengths):                 
            predictions.extend(self.get_lspr_results(boxes_xyxy, label, logits, (H, W), from_name_t, to_name))

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }

    def get_lspr_results(self, boxes_xyxy, labels, logits, lengths, from_name_t, to_name):
        results = []
        height, width = lengths
        for box, label, score in zip(boxes_xyxy, labels, logits):
            # random ID
            label_id = str(uuid4())[:9]
            results.append({
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    'width': (box[2] - box[0]) / width * 100,
                    'height': (box[3] - box[1]) / height * 100,
                    'x': box[0] / width * 100,
                    'y': box[1] / height * 100,
                    "text": [
                        label
                    ]
                },
                "id": label_id,
                "from_name": from_name_t,
                "to_name": to_name,
                "type": "textarea",
                "origin": "manual"
            })

        return results
    
    def multiple_doc_tasks(self, image_paths, from_name_r, from_name_t, from_name_lang, from_name_type, to_name):
        
        predictions = []

        res_list = doc_pipeline.predict(image_paths)
        
        for res in res_list:                 
            predictions.extend(self.get_ocr_results(res, from_name_r, from_name_t, from_name_lang, from_name_type, to_name))

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }    

    def get_ocr_results(self, res, from_name_r, from_name_t, from_name_lang, from_name_type, from_name_rot, to_name):
        results = []
        height, width = res[2]
        lang = res[4]
        print(lang)

        label_id = str(uuid4())[:9]
        results.append({
          "from_name": from_name_lang,
          "to_name": to_name,
          "type": "choices",
          "value": { "choices": lang }
        })
        label_id = str(uuid4())[:9]
        results.append({
          "from_name": from_name_type,
          "to_name": to_name,
          "type": "choices",
          "value": { "choices": ["Others"] }
        })
        label_id = str(uuid4())[:9]
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

        for rs in res[1]:
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
                    "rectanglelabels": [rs['type']],
                    'width': (points[2] - points[0]) / width * 100,
                    'height': (points[3] - points[1]) / height * 100,
                    'x': points[0] / width * 100,
                    'y': points[1] / height * 100
                },
                'type': 'rectanglelabels',
            })
           
        return results

    def multiple_video_tasks(self, image_paths, from_name_p, from_name_r,to_name, prompt):
        
        predictions = []
        res_list = video_pipeline.predict(image_paths, prompt)
        print(res_list)
        
        for res in res_list:                 
            predictions.extend(self.get_video_results(res, from_name_p, from_name_r, to_name))

        return {
            'result': predictions,
            'score': 0,
            'model_version': self.get('model_version')
        }    

    def get_video_results(self, res, from_name_p, from_name_r, to_name):
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
                "from_name": from_name_p,
                "to_name": to_name,
                "type": "textarea",
                "origin": "manual"
            })
            results.append({
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "text": [
                        rs[1]
                    ]
                },
                "id": label_id,
                "from_name": from_name_r,
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

        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(data['annotation'], prompt_tag)

        if not prompts:
            print(f'No prompts recorded.')
            return

        prompt = '\n'.join(prompts)
        current_prompt = self.get(prompt_tag.name)

        # find substrings that differ between current and new prompt
        # if there are no differences, skip training
        if current_prompt:
            diff = self._prompt_diff(current_prompt, prompt)
            if not diff:
                print('No prompt diff found.')
                return

            print(f'Prompt diff: {diff}')
        self.set(prompt_tag.name, prompt)
        model_version = self.bump_model_version()

        print(f'Updated model version to {str(model_version)}')