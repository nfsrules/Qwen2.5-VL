# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
import shutil
import sys
from pathlib import Path
from accelerate import Accelerator

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
)
from qwenvl.data.data_qwen import make_supervised_data_module

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor
from peft import LoraConfig, get_peft_model, PeftModel

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # Check if the model is a PEFT model
    if isinstance(trainer.model, PeftModel):
        trainer.model.save_pretrained(output_dir)  # Save LoRA adapters
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    # Check if use_lora is defined, default to False if not
    use_lora = getattr(model_args, "use_lora", False)
    
    if use_lora:
        # When using LoRA, disable full fine-tuning
        for n, p in model.named_parameters():
            p.requires_grad = False
    else:
        # Original fine-tuning logic
        if model_args.tune_mm_vision:
            for n, p in model.visual.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.visual.named_parameters():
                p.requires_grad = False

        if model_args.tune_mm_mlp:
            for n, p in model.visual.merger.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.visual.merger.named_parameters():
                p.requires_grad = False

        if model_args.tune_mm_llm:
            for n, p in model.model.named_parameters():
                p.requires_grad = True
            model.lm_head.requires_grad = True
        else:
            for n, p in model.model.named_parameters():
                p.requires_grad = False
            model.lm_head.requires_grad = False


def apply_lora(model, model_args):
    """Apply LoRA configuration to the model."""
    lora_config = LoraConfig(
        r=16,  # Rank of LoRA adapters
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target attention and feed-forward layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Log trainable parameters
    return model


class CustomTrainer(Trainer):
    """Custom Trainer to disable GradScaler for bf16 and support DeepSpeed checks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize attributes expected by Trainer
        self.is_deepspeed_enabled = getattr(self.args, "deepspeed", None) is not None and self.args.deepspeed != ""

    def create_accelerator_and_postprocess(self):
        # Initialize Accelerator with bf16
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            device_placement=True,
            step_scheduler_with_optimizer=False,
        )
        return self.accelerator

    def clip_grad_norm_(self, max_grad_norm):
        # Skip gradient scaling for bf16
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load model with bf16
    torch_dtype = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    model = model.to("cuda")  # Move to GPU as per warning
    print(f"Memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Set training parameters
    set_model(model_args, model)  # Your existing function
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Enforce bf16 and disable fp16
    training_args.fp16 = False
    training_args.bf16 = True
    training_args.ddp_find_unused_parameters = False
    print(f"Final training args: {training_args}")

    # Initialize CustomTrainer
    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )

    # Start training
    trainer.train()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)