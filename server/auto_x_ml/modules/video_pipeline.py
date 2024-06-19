import os
import torch
import transformers

from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from videollama2.model.builder import load_pretrained_model

class VideoPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.environ.get("videollama2_model_dir", 'DAMO-NLP-SG/VideoLLaMA2-7B-Base')
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, model, self.processor, self.context_len = load_pretrained_model(model_path, None, model_name)
        self.model = model.to(self.device)
        self.conv_mode = 'llama_2'

    def inference(self, paths, questions, modal_list):
       
        # 2. Visual preprocess (load & transform image or video).
        if modal_list[0] == 'video':
            tensor = process_video(paths[0], self.processor, self.model.config.image_aspect_ratio).to(dtype=torch.float16, device=self.device, non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
            modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
        else:
            tensor = process_image(paths[0], self.processor, self.model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device=self.device, non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
            modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
        tensor = [tensor]

        # 3. text preprocess (tag process & generate prompt).
        question = default_mm_token + "\n" + questions[0]
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_MMODAL_token(prompt, self.tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images_or_videos=tensor,
                modal_list=modal_list,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs[0]

