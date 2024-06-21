import os
import pathlib
import numpy as np
from typing import Tuple, Dict
import clip
from collections import OrderedDict
import re
import datetime
import json
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.ops import nms
from transformers import AutoTokenizer

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

from .utils.slconfig import SLConfig
from .utils import transforms as T
from .utils.box_ops import box_cxcywh_to_xyxy
from .UniPose.predefined_keypoints import *

from .groundingdino.util.get_param_dicts import get_param_dict
# from .groundingdino.util.utils import  BestMetricHolder
# from .groundingdino.util.misc import *
# from .groundingdino.datasets import build_dataset, get_coco_api_from_dataset
# from .groundingdino.engine import evaluate, train_one_epoch
# from .groundingdino.util.utils import clean_state_dict


class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

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

class VisualPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
        self.IOU_THRESHOLD = os.environ.get("IOU_THRESHOLD", 0.8)
        self.TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)

        ram_model = ram_plus(pretrained=os.environ.get("ram_model"),
                                image_size=os.environ.get("ram_img_size", 384),
                                vit='swin_l')
        ram_model.eval()
        self.ram_model = ram_model.to(self.device)
        self.transform = get_transform(image_size=os.environ.get("ram_img_size", 384))

        from groundingdino.models import build_model
        args = SLConfig.fromfile("./auto_x_ml/pipelines/groundingdino/GroundingDINO_SwinT_OGC.py")
        #args.text_encoder_type = os.environ.get('bert-base-uncased')
        self.groundingdino_model = build_model(args)
        checkpoint = torch.load(pathlib.Path(os.environ.get('groundingdino_model')), map_location="cpu")
        self.groundingdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.groundingdino_model.eval()

        from .UniPose.models import build_model
        args = SLConfig.fromfile('./auto_x_ml/pipelines/UniPose/UniPose_SwinT.py')
        #args.text_encoder_type = os.environ.get('bert-base-uncased')
        args.device = self.device
        self.pose_model = build_model(args)
        checkpoint = torch.load(pathlib.Path(os.environ.get('pose_model')), map_location="cpu")
        self.pose_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.pose_model.eval()

        self.prompts = []

    def get_prompts(self, image_paths):
        # Fix me: Change to batch inferences
        for img in image_paths:
            image = self.transform(Image.open(img)).unsqueeze(0).to(self.device)
            res = inference(image, self.ram_model)
            self.prompts = ','.join(res[0])

    def run_detection(self, image_paths):
        # Fix me: Change to batch inferences
        self.get_prompts(image_paths)
        all_boxes = []
        all_labels = []
        all_logits = []
        all_lengths = []
        instance_text_prompt = self.prompts
        if len(instance_text_prompt) > 0:
            instance_text_prompt = set(instance_text_prompt)
            instance_text_prompt = ','.join(instance_text_prompt)
            print(instance_text_prompt)
            for img in image_paths:
                src, img = load_image(img)

                boxes_filt, pred_phrases = get_grounding_output(
                        model=self.groundingdino_model,
                        image=img,
                        caption=instance_text_prompt,
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

    def build_model_main(self, args):
        # we use register to maintain models from catdet6 on.
        from .groundingdino.models.registry import MODULE_BUILD_FUNCS
        assert args.modelname in MODULE_BUILD_FUNCS._module_dict

        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, criterion, postprocessors = build_func(args)
        return model, criterion, postprocessors

    def train_detection(self, data):
        args = AttrDict()
        args['output_dir'] = os.environ.get("MODEL_POOL_DIR") + "recognize_anything"
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cfg = SLConfig.fromfile("./auto_x_ml/modules/groundingdino/config/cfg_coco.py")
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k,v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError("Key {} can used by args only".format(k))
        
        with open("./auto_x_ml/modules/groundingdino/config/datasets_od_example.json") as f:
            dataset_meta = json.load(f)
        if args.use_coco_eval:
            args.coco_val_path = dataset_meta["val"][0]["anno"]

        model, criterion, postprocessors = self.build_model_main(args)
        wo_class_error = False
        model.to(self.device)

        param_dicts = get_param_dict(args, model)
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            dataset_train = ConcatDataset(dataset_train_list)

        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
        
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

        if args.onecyclelr:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
        elif args.multi_step_lr:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


        base_ds = get_coco_api_from_dataset(dataset_val)

        output_dir = pathlib.Path(args.output_dir)
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
            args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)


            
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if (not args.resume) and args.pretrain_model_path:
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
            from collections import OrderedDict
            _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
            ignorelist = []

            def check_keep(keyname, ignorekeywordlist):
                for keyword in ignorekeywordlist:
                    if keyword in keyname:
                        ignorelist.append(keyname)
                        return False
                return True

            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

            _load_output = model.load_state_dict(_tmp_st, strict=False)

        best_map_holder = BestMetricHolder(use_ema=False)

        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()

            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, self.device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=None)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

            if not args.onecyclelr:
                lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }

                    save_on_master(weights, checkpoint_path)
                    
            # eval
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, self.device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=None
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }


            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
                            
    def train_keypoints(self, data):
        pass