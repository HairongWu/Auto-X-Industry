import torch
import sys
from hf_mini.utils import input_wrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodePipeline(): 
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("aiXcoder/aixcoder-7b-base")
        self.model = AutoModelForCausalLM.from_pretrained("aiXcoder/aixcoder-7b-base", torch_dtype=torch.bfloat16)
        self.model.to(self.device)

    def generate_fuctions(self, code_string, test_code, output_file):
        text = input_wrapper(
            # for FIM style input, code_string stands for prefix context
            code_string=code_string,
            # for FIM style input, later_code stands for suffix context
            later_code=test_code,
            # file_path should be a path from project to file
            path=output_file
        )

        if len(text) == 0:
            sys.exit()

        inputs = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        inputs = inputs.to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    def doc_gen(self):
        pass

    def finetuning(self):
        pass