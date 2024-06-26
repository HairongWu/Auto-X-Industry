import torch
import sys
from hf_mini.utils import input_wrapper

import multiprocessing
import os
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
    set_seed,
)

from trl import SFTTrainer

class CodePipeline(): 
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("aiXcoder/aixcoder-7b-base")
        self.model = AutoModelForCausalLM.from_pretrained("aiXcoder/aixcoder-7b-base", torch_dtype=torch.bfloat16)
        self.model.to(self.device)

    def generate_fuctions(self, code_string, test_code, output_file):
        text = input_wrapper(
            # for FIM style input, code_string stands for prefix context
            code_string=code_string,
            # for FIM style input, later_code stands for suffix context
            later_code=test_code,
            # file_path should be a path from project to file
            path=output_file
        )

        if len(text) == 0:
            sys.exit()

        inputs = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        inputs = inputs.to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    def doc_gen(self):
        pass

    def finetuning(self):
        # config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        lora_config = LoraConfig(
            r=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

        # load model and dataset
        token = os.environ.get("HF_TOKEN", None)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map={"": PartialState().process_index},
            attention_dropout=args.attention_dropout,
            attn_implementation='flash_attention_2'
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        print_trainable_parameters(model)

        data = load_dataset(
            args.dataset_name,
            data_dir=args.subset,
            split=args.split,
            token=token,
            num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
        )

        train_data = RandomFIMDataset(
            tokenizer=tokenizer, dataset=data, fim_rate=args.fim_rate, dataset_text_field=args.dataset_text_field,
            infinite=True, seq_length=args.max_seq_length, eos_token_id=tokenizer.eos_token_id
        )

        # setup the trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            max_seq_length=args.max_seq_length,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                lr_scheduler_type=args.lr_scheduler_type,
                weight_decay=args.weight_decay,
                bf16=args.bf16,
                logging_strategy="steps",
                logging_steps=10,
                output_dir=args.output_dir,
                optim="paged_adamw_8bit",
                seed=args.seed,
                run_name=f"train-{args.model_id.split('/')[-1]}",
                report_to="none",
            ),
            peft_config=lora_config,
            dataset_text_field=args.dataset_text_field,
        )

        # launch
        print_rank_0("Training...")
        trainer.train()

        print_rank_0("Saving the last checkpoint of the model")
        model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
        if args.push_to_hub:
            trainer.push_to_hub("Upload model")
        print_rank_0("Training Done! ")
