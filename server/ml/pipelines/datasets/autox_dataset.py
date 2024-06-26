import torch
from skimage import io, transform
from torch.utils.data import Dataset
from PIL import Image

__all__ = ['build']

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        boxes = [obj["bbox"] for obj in target]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
       
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
    
class AutoXDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        self.transform = transform
        self.prepare = ConvertCocoPolysToMask()

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.annotations[id][0]
        img = Image.open(img_name).convert("RGB")
        img, target = self.prepare(img, self.annotations[0][1])
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build(datasetinfo):
    dataset = AutoXDataset(datasetinfo)
    return dataset

