import os
import pathlib
import numpy as np
from typing import Tuple, Dict
from collections import OrderedDict
import re
import datetime
import json
import time
from PIL import Image

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .utils.slconfig import SLConfig
from .utils import transforms as T
from .utils.box_ops import box_cxcywh_to_xyxy

from .utils.get_param_dicts import get_param_dict
# from .groundingdino.util.utils import  BestMetricHolder
from .utils.misc import *
from .datasets import build_dataset
# from .groundingdino.engine import evaluate, train_one_epoch
# from .groundingdino.util.utils import clean_state_dict

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

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

class DetectionPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
        self.TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)

        from groundingdino.models import build_model
        args = SLConfig.fromfile("./auto_x_ml/pipelines/groundingdino/GroundingDINO_SwinT_OGC.py")
        #args.text_encoder_type = os.environ.get('bert-base-uncased')
        self.groundingdino_model = build_model(args)
        checkpoint = torch.load(pathlib.Path(os.environ.get('groundingdino_model')), map_location="cpu")
        self.groundingdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.groundingdino_model.eval()


    def run_detection(self, image_paths, prompts):
        # Fix me: Change to batch inferences
        all_boxes = []
        all_labels = []
        all_logits = []
        all_lengths = []
        if len(prompts) > 0:
            for img, prompt in zip(image_paths, prompts):
                src, img = load_image(img)

                boxes_filt, pred_phrases = get_grounding_output(
                        model=self.groundingdino_model,
                        image=img,
                        caption=prompt,
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
    
    def build_model_main(self, args):
        # we use register to maintain models from catdet6 on.
        from .groundingdino.models.registry import MODULE_BUILD_FUNCS
        assert args.modelname in MODULE_BUILD_FUNCS._module_dict

        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, criterion, postprocessors = build_func(args)
        return model, criterion, postprocessors

    def train_detection(self, annotations):
        args = AttrDict()
        args['output_dir'] = os.environ.get("MODEL_POOL_DIR") + "detection"
        args['config_file'] = "./auto_x_ml/pipelines/groundingdino/config/cfg_coco.py"
        args['device'] = self.device
        args['num_workers'] = 8
        args['start_epoch'] = 0
        args['pretrain_model_path'] = None
        args['resume'] = None
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cfg = SLConfig.fromfile(args.config_file)
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k,v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError("Key {} can used by args only".format(k))

        model, criterion, postprocessors = self.build_model_main(args)
        wo_class_error = False
        model.to(self.device)

        param_dicts = get_param_dict(args, model)
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    
        print(annotations)
        train, val = train_test_split(annotations, test_size=0.25)
        dataset_train = build_dataset(datasetinfo=train)
        dataset_val = build_dataset(datasetinfo=val)
        
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


        # base_ds = get_coco_api_from_dataset(dataset_val)

        if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
            args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if (not args.resume) and args.pretrain_model_path:
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']

        # best_map_holder = BestMetricHolder(use_ema=False)

        # for epoch in range(args.start_epoch, args.epochs):
        #     epoch_start_time = time.time()

        #     train_stats = train_one_epoch(
        #         model, criterion, data_loader_train, optimizer, self.device, epoch,
        #         args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=None)
        #     if args.output_dir:
        #         checkpoint_paths = [output_dir / 'checkpoint.pth']

        #     if not args.onecyclelr:
        #         lr_scheduler.step()
        #     if args.output_dir:
        #         checkpoint_paths = [output_dir / 'checkpoint.pth']
        #         # extra checkpoint before LR drop and every 100 epochs
        #         if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
        #             checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        #         for checkpoint_path in checkpoint_paths:
        #             weights = {
        #                 'model': model.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'epoch': epoch,
        #                 'args': args,
        #             }

        #             save_on_master(weights, checkpoint_path)
                    
        #     # eval
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, data_loader_val, base_ds, self.device, args.output_dir,
        #         wo_class_error=wo_class_error, args=args, logger=None
        #     )
        #     map_regular = test_stats['coco_eval_bbox'][0]
        #     _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        #     if _isbest:
        #         checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
        #         save_on_master({
        #             'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)
        #     log_stats = {
        #         **{f'train_{k}': v for k, v in train_stats.items()},
        #         **{f'test_{k}': v for k, v in test_stats.items()},
        #     }


        #     try:
        #         log_stats.update({'now_time': str(datetime.datetime.now())})
        #     except:
        #         pass
            
        #     epoch_time = time.time() - epoch_start_time
        #     epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        #     log_stats['epoch_time'] = epoch_time_str

        #     if args.output_dir and is_main_process():
        #         with (output_dir / "log.txt").open("a") as f:
        #             f.write(json.dumps(log_stats) + "\n")

        #         # for evaluation logs
        #         if coco_evaluator is not None:
        #             (output_dir / 'eval').mkdir(exist_ok=True)
        #             if "bbox" in coco_evaluator.coco_eval:
        #                 filenames = ['latest.pth']
        #                 if epoch % 50 == 0:
        #                     filenames.append(f'{epoch:03}.pth')
        #                 for name in filenames:
        #                     torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                             output_dir / "eval" / name)
