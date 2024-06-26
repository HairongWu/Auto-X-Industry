import os
import torch

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from .videollama2.conversation import conv_templates
from .videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from .videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from .videollama2.model.builder import load_pretrained_model
from .videollama2 import conversation as conversation_lib
from .videollama2.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from .videollama2.videollama2_trainer import VideoLLaMA2Trainer
from .videollama2.model import *
from .videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import pathlib

@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Other Arguments
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    pretrain_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "To train from previously trained checkpoints. E.g, further fine-tuning based on the finetuned version of the whole model."})


@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

class VideoPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.environ.get("videollama2_model_dir", 'DAMO-NLP-SG/VideoLLaMA2-7B-Base')
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, model, self.processor, self.context_len = load_pretrained_model(model_path, None, model_name)
        self.model = model.to(self.device)
        self.conv_mode = 'llama_2'

    def run_video_caption(self, paths, questions, modal_list):
       
        # 2. Visual preprocess (load & transform image or video).
        if modal_list[0] == 'video':
            tensor = process_video(paths[0], self.processor, self.model.config.image_aspect_ratio).to(dtype=torch.float16, device=self.device, non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
            modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
        else:
            tensor = process_image(paths[0], self.processor, self.model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device=self.device, non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
            modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
        tensor = [tensor]

        # 3. text preprocess (tag process & generate prompt).
        question = default_mm_token + "\n" + questions[0]
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_MMODAL_token(prompt, self.tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images_or_videos=tensor,
                modal_list=modal_list,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs[0]

    def train_video_caption(self, paths, questions, modal_list):
        attn_implementation = "flash_attention_2"
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

        bnb_model_from_pretrained_args = {}
        if training_args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                )
            ))
        if model_args.pretrain_model_name_or_path is not None:
            assert os.path.exists(model_args.pretrain_model_name_or_path)
            pretrain_model_name_or_path = model_args.pretrain_model_name_or_path
        else:
            pretrain_model_name_or_path = model_args.model_name_or_path
        if model_args.vision_tower is not None:
            if 'mistral' in model_args.model_name_or_path.lower():
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                config._attn_implementation = attn_implementation
                model = Videollama2MistralForCausalLM.from_pretrained(
                    pretrain_model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    do_sample=True,
                    **bnb_model_from_pretrained_args
                )
            elif 'mixtral' in model_args.model_name_or_path.lower():
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                config._attn_implementation = attn_implementation
                model = Videollama2MixtralForCausalLM.from_pretrained(
                    pretrain_model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    do_sample=True,
                    **bnb_model_from_pretrained_args
                )
                import deepspeed
                deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
            else:
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                config._attn_implementation = attn_implementation
                model = Videollama2LlamaForCausalLM.from_pretrained(
                    pretrain_model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    do_sample=True,
                    **bnb_model_from_pretrained_args
                )
        else:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = transformers.LlamaForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        model.config.use_cache = False

        if model_args.freeze_backbone:
            model.model.requires_grad_(False)

        if training_args.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)


        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
            else:
                if model_args.version == "v1":
                    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
                elif model_args.version == "v1_mistral":
                    conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]

        if model_args.vision_tower is not None:
            # initialize vision encoder + multi-modal projector
            model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = vision_tower.image_processor
            data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

            data_args.is_multimodal = True

            model.config.image_aspect_ratio = data_args.image_aspect_ratio
            model.config.tokenizer_padding_side = tokenizer.padding_side
            model.config.tokenizer_model_max_length = tokenizer.model_max_length

            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            if model_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            if training_args.bits in [4, 8]:
                model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

            model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_projector_lr = training_args.mm_projector_lr
            training_args.use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
            model.initialize_MM_tokenizer(model_args, tokenizer=tokenizer)

            model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames

        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        print("Current model:", model)
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        # select a Trainer
        trainer = VideoLLaMA2Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        model.config.use_cache = True

        if training_args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(training_args.output_dir)
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

