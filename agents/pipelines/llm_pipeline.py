from transformers import AutoTokenizer, AutoModel
import re

from .pipeline import *

class LLMPipeline(Pipeline): 
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device=self.device)
        self.model = model.eval()

    def predict(self, messages, history):
        response = []
        for message in messages:
            res, history = self.model.chat(self.tokenizer, message, history=[])
            response.append(res)
            print('history', history)
        
        return response
        

