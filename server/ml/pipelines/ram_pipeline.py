import os
import clip
import json
import ruamel.yaml as yaml
from pathlib import Path

from PIL import Image
import torch
import clip

from .ram.models import ram_plus
from .ram import inference_ram as inference
from .ram import get_transform
from .utils.ram_utils import *
# from .ram.data import create_dataset, create_sampler, create_loader

class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

class RamPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ram_model = ram_plus(pretrained=os.environ.get("ram_model"),
                                image_size=os.environ.get("ram_img_size", 384),
                                vit='swin_l')
        ram_model.eval()
        self.ram_model = ram_model.to(self.device)
        self.transform = get_transform(image_size=os.environ.get("ram_img_size", 384))

    def run_ram(self, image_paths):
        prompts = []
        for img in image_paths:
            image = self.transform(Image.open(img)).unsqueeze(0).to(self.device)
            res = inference(image, self.ram_model)
            res = res[0].split('|')
            res = [r.strip() for r in res]
            prompts.append(','.join(res))

        return prompts

    def build_text_embed(self, model_clip, caption):
        with torch.no_grad():

            texts = clip.tokenize(caption,truncate = True)  # tokenize
            if self.device == "cuda":
                texts = texts.cuda()
                model_clip = model_clip.cuda()
            text_embeddings = model_clip.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            # text_embedding = text_embeddings.mean(dim=0)
            # text_embedding /= text_embedding.norm()
        return text_embeddings



    def train_ram_plus(self, model, data_loader, optimizer, epoch, device, config, model_clip):
        # train
        model.train()  
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_tag', SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_dis', SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_alignment', SmoothedValue(window_size=50, fmt='{value:.4f}'))
        
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50   
        
        data_loader.sampler.set_epoch(epoch)

        for i, (image, caption, image_tag, parse_tag) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
            if epoch==0:
                warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
                
            optimizer.zero_grad()

            batch_text_embed = self.build_text_embed(model_clip,caption)
            
            image = image.to(device,non_blocking=True)

            with torch.no_grad():
                clip_image_feature = model_clip.encode_image(image)

            loss_tag, loss_dis, loss_alignment = model(image, caption, image_tag, clip_image_feature, batch_text_embed)  
            loss = loss_tag + loss_dis + loss_alignment

            loss.backward()
            optimizer.step()    

            metric_logger.update(loss_tag=loss_tag.item())
            metric_logger.update(loss_dis=loss_dis.item())
            metric_logger.update(loss_alignment=loss_alignment.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())     
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


    def pretrain_ram(self, data):
        args = AttrDict()
        args['config'] = './configs/pretrain.yaml'
        args['output_dir'] = 'output/Pretrain'
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        datasets = [create_dataset('pretrain', config, min_scale=0.2)]

        num_tasks = get_world_size()
        global_rank = get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)         

        data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
        model_clip, _ = clip.load("ViT-B/16", device=self.device)
        model = ram_plus(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                                vit_ckpt_layer=config['vit_ckpt_layer'], stage = 'train_from_scratch')
        
        model = model.to(self.device) 

        ### Frozen CLIP model ###
        model_clip = model_clip.to(self.device)
        for _, param in model_clip.named_parameters():
            param.requires_grad = False

        ### Frozen label embedding for open-set recogniztion ###
        model.label_embed.requires_grad = False
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
        
        start_epoch = 0
        model_without_ddp = model 
            
        print("Start training")
        for epoch in range(start_epoch, config['max_epoch']):
            
            step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
            train_stats = self.train_ram_plus(model, data_loader, optimizer, epoch, self.device, config, model_clip) 


            if is_main_process():  
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }                     
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")      
                
    
    def finetune_ram(self, data):
        args = AttrDict()
        args['config'] = './configs/finetune.yaml'
        args['output_dir'] = 'output/finetune'
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

        datasets = [create_dataset('finetune', config, min_scale=0.2)]
        num_tasks = get_world_size()
        global_rank = get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
        data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
        model_clip, _ = clip.load("ViT-B/16", device=self.device)

        model = ram_plus(pretrained = args.checkpoint,image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                                vit_ckpt_layer=config['vit_ckpt_layer'])
        
        model = model.to(self.device)

        model_clip = model_clip.to(self.device)
        for _, param in model_clip.named_parameters():
            param.requires_grad = False

        ### Frozen label embedding for open-set recogniztion ###
        model.label_embed.requires_grad = False
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
        
        start_epoch = 0
        
        model_without_ddp = model
            
        print("Start training") 
        for epoch in range(start_epoch, config['max_epoch']):
            
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = self.train_ram_plus(model, data_loader, optimizer, epoch, self.device, config, model_clip) 

            if is_main_process():  
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }                     
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
