import os
from PIL import Image
import re
from typing import Tuple, Dict
import pathlib

# from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

import torch
from transformers import AutoTokenizer

from .utils.slconfig import SLConfig
from .utils import transforms as T
from .utils.box_ops import box_cxcywh_to_xyxy
from .utils.utils import  clean_state_dict
from .utils.misc import *

from .ram.models import ram_plus
from .ram import inference_ram as inference
from .ram import get_transform
from .utils.ram_utils import *

from .pipeline import *

def get_phrases_from_posmap(
    posmap: torch.BoolTensor, tokenized: Dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")
    
def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, device='cpu', token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

tform= transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class LSPRPipeline(Pipeline): 
    def __init__(self):
        super().__init__()

        self.BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.40)
        self.TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)

        from groundingdino.models import build_model
        args = SLConfig.fromfile("./pipelines/groundingdino/GroundingDINO_SwinT_OGC.py")
        #args.text_encoder_type = os.environ.get('bert-base-uncased')
        self.groundingdino_model = build_model(args)
        checkpoint = torch.load(pathlib.Path(os.environ.get('groundingdino_model')), map_location="cpu")
        self.groundingdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.groundingdino_model.eval()

        ram_model = ram_plus(pretrained=os.environ.get("ram_model"),
                                image_size=os.environ.get("ram_img_size", 384),
                                vit='swin_l')
        ram_model.eval()
        self.ram_model = ram_model.to(self.device)
        self.transform = get_transform(image_size=os.environ.get("ram_img_size", 384))

        embedding = models.resnet101(pretrained=True)
        return_nodes = {
            'avgpool': 'embedding'
        }
        self.embedding = create_feature_extractor(embedding, return_nodes=return_nodes)

        self.embedding.eval()
        self.embedding.to(self.device)

        vector_store_path = os.path.join(os.environ.get('MODEL_POOL_DIR'),'autox_lspr')
        # if os.path.exists(vector_store_path):
        #     self.vector_store = VectorStore(
        #         path = vector_store_path
        #     )
        # else:
        #     self.vector_store = VectorStore(
        #         path = vector_store_path,
        #         tensor_params = [{'name': 'embedding', 'htype': 'embedding'}, 
        #                         {'name': 'description', 'htype': 'tag'}],
        #     )

    def predict(self, image_paths, prompts=None):
        # Fix me: Change to batch inferences
        all_boxes = []
        all_labels = []
        all_logits = []
        all_lengths = []

        for img in image_paths:
            if prompts is None or len(prompts) < 1:
                image = self.transform(Image.open(img)).unsqueeze(0).to(self.device)
                res = inference(image, self.ram_model)
                res = res[0].split('|')
            else:
                res = prompts
            res = [r.strip().lower() for r in prompts]
            prompts = ','.join(res)

            if len(prompts) > 0:
                src, img = load_image(img)

                boxes_filt, pred_phrases = get_grounding_output(
                        model=self.groundingdino_model,
                        image=img,
                        caption=prompts,
                        box_threshold=float(self.BOX_THRESHOLD),
                        text_threshold=float(self.TEXT_THRESHOLD),
                        device=self.device
                    )
                
                H, W, _ = src.shape
                boxes_xyxy = box_cxcywh_to_xyxy(boxes_filt) * torch.Tensor([W, H, W, H])
                points = boxes_xyxy.cpu().numpy()
                
                all_boxes.append(points)
                all_lengths.append((H, W))
                score = []
                label = []
                for pred in pred_phrases:
                    score.append(re.search("[0-9.-]+", pred).group())
                    label.append(re.search("^[^(]+", pred).group())
                all_logits.append(score)
                all_labels.append(label)
            
        return all_boxes, all_labels, all_logits, all_lengths


    def get_from_vdb(self, image, boxes, cls):
        return cls

    def embedding_function(self, images, transform = tform, batch_size = 4):

        #Proceess the embeddings in batches, but return everything as a single list
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([transform(item) for item in images[i:i+batch_size]])
            batch = batch.to(self.device)
            with torch.no_grad():
                embeddings+= self.embedding(batch)['embedding'][:,:,0,0].cpu().numpy().tolist()

        return embeddings
    def save_to_vdb(self, annotations, names):
        images = []
        des = []
        for annos in annotations:
            des.append(names[annos['cls'][0][0]])
            boxes = annos['bboxes'][0]
            im_file = Image.open(annos['im_file'])
            images.append(im_file.crop(boxes))


        self.vector_store.add(
                 description = des,
                 embedding_function = self.embedding_function, 
                 embedding_data = images)