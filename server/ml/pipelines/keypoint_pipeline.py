import os
import pathlib
import numpy as np
from typing import Tuple
import clip
from collections import OrderedDict

import torch
from PIL import Image
from torchvision.ops import nms

from .utils.slconfig import SLConfig
from .utils import transforms as T
from .utils.box_ops import box_cxcywh_to_xyxy
from .UniPose.predefined_keypoints import *


class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def text_encoding(instance_names, keypoints_names, model, device):

    ins_text_embeddings = []
    for cat in instance_names:
        instance_description = f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
        text = clip.tokenize(instance_description).to(device)
        text_features = model.encode_text(text)  # 1*512
        ins_text_embeddings.append(text_features)
    ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

    kpt_text_embeddings = []

    for kpt in keypoints_names:
        kpt_description = f"a photo of {kpt.lower().replace('_', ' ')}"
        text = clip.tokenize(kpt_description).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)  # 1*512
        kpt_text_embeddings.append(text_features)

    kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)


    return ins_text_embeddings, kpt_text_embeddings

def get_unipose_output(model, image, instance_text_prompt,keypoint_text_prompt, box_threshold,iou_threshold, device):
    # instance_text_prompt: A, B, C, ...
    # keypoint_text_prompt: skeleton
    instance_list = instance_text_prompt.split(',')

    # clip_model, _ = clip.load("ViT-B/32", device=device)
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, model.clip_model, device)
    target={}
    target["instance_text_prompt"] = instance_list
    target["keypoint_text_prompt"] = keypoint_text_prompt
    target["object_embeddings_text"] = ins_text_embeddings.float()
    kpt_text_embeddings = kpt_text_embeddings.float()
    kpts_embeddings_text_pad = torch.zeros(100 - kpt_text_embeddings.shape[0], 512,device=device)
    target["kpts_embeddings_text"] = torch.cat((kpt_text_embeddings, kpts_embeddings_text_pad), dim=0)
    kpt_vis_text = torch.ones(kpt_text_embeddings.shape[0],device=device)
    kpt_vis_text_pad = torch.zeros(kpts_embeddings_text_pad.shape[0],device=device)
    target["kpt_vis_text"] = torch.cat((kpt_vis_text, kpt_vis_text_pad), dim=0)
    # import pdb;pdb.set_trace()
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], [target])


    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    keypoints = outputs["pred_keypoints"][0][:,:2*len(keypoint_text_prompt)] # (nq, n_kpts * 2)
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    keypoints_filt = keypoints.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    keypoints_filt = keypoints_filt[filt_mask]  # num_filt, 4


    keep_indices = nms(box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=iou_threshold)

    # Use keep_indices to filter boxes and keypoints
    filtered_boxes = boxes_filt[keep_indices]
    filtered_keypoints = keypoints_filt[keep_indices]
    filtered_logits = logits_filt[keep_indices]


    return filtered_boxes,filtered_keypoints, filtered_logits

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

class KeypointPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
        self.IOU_THRESHOLD = os.environ.get("IOU_THRESHOLD", 0.8)
        self.TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)

        from .UniPose.models import build_model
        args = SLConfig.fromfile('./ml/pipelines/UniPose/UniPose_SwinT.py')
        #args.text_encoder_type = os.environ.get('bert-base-uncased')
        args.device = self.device
        self.pose_model = build_model(args)
        checkpoint = torch.load(pathlib.Path(os.environ.get('pose_model')), map_location="cpu")
        self.pose_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.pose_model.eval()

    def run_keypoints(self, image_paths):
        # Fix me: Change to batch inferences
        all_keypoints = []
        all_lengths = []

        all_boxes = []
        all_logits = []
        all_labels = []

        for img in image_paths:
            src, img = load_image(img)
            H, W, _ = src.shape
            keypoints = []
            boxes_xyxy = []
            prompts = []
            logits = []
            for prompt in predefined_keypoints:
                keypoint_dict = globals()[prompt]
                keypoint_text_prompt = keypoint_dict.get("keypoints")
                num_kpts = len(keypoint_text_prompt)
                boxes_filt,keypoints_filt, logits_filt = get_unipose_output(
                    self.pose_model, 
                    img, 
                    prompt, 
                    keypoint_text_prompt, 
                    box_threshold=float(self.BOX_THRESHOLD), 
                    iou_threshold=float(self.IOU_THRESHOLD), 
                    device=self.device
                )

                for idx, ann in enumerate(keypoints_filt):
                    kp = np.array(ann.cpu())
                    Z = kp[:num_kpts*2] * np.array([W, H] * num_kpts)
                    x = Z[0::2]
                    y = Z[1::2]
                    for i in range(num_kpts):
                        keypoints.append((x[i], y[i], keypoint_text_prompt[i]))

                boxes_filt = box_cxcywh_to_xyxy(boxes_filt) * torch.Tensor([W, H, W, H])
                boxes_xyxy.extend(boxes_filt.cpu().numpy())
                prompts.extend([prompt]*len(boxes_xyxy))

                logits_filt = logits_filt.cpu().numpy()
                logits.extend(logits_filt[:,0:1])


            all_keypoints.append(keypoints)
            all_lengths.append((H, W))  

            all_boxes.append(boxes_xyxy)
            all_labels.append(prompts)
            all_logits.append(logits)
                        

        return all_keypoints, all_boxes, all_labels, all_logits, all_lengths
                            
    def train_keypoints(self, data):
        pass