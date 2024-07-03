from transformers import AutoTokenizer, AutoModel
import re

from .pipeline import *

class LLMPipeline(Pipeline): 
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device=self.device)
        self.model = model.eval()

        self.history = []

    def predict(self, messages):
        
        response, self.history = self.model.chat(self.tokenizer, messages, self.history)
        
        return response
        

