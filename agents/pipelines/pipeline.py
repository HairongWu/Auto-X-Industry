
import abc
import torch

class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self
           
class Pipeline(metaclass=abc.ABCMeta): 
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def predict(self):
        pass

    def generate_dataset(self):
        pass

    def scraping_dataset(self):
        pass