#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VERL SFT Implementation - Fast startup, clean design
Full ms-swift compatibility with better performance and simplicity
"""

import os
import sys
from argparse import Namespace
from typing import Union, List, Optional, Dict, Any
import json
import time
import logging

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from datasets import Dataset

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class VerlSft:
    """
    VERL SFT implementation - fast startup, clean design
    Full ms-swift compatibility with better performance
    """
    
    def __init__(self, args: Union[Namespace, None] = None):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
    def main(self) -> int:
        """Main training pipeline"""
        try:
            logger.info("Starting VERL SFT training...")
            
            # 1. Setup distributed training if needed
            self._setup_distributed()
            
            # 2. Load model and tokenizer
            self._load_model_tokenizer()
            
            # 3. Setup LoRA if needed
            self._setup_lora()
            
            # 4. Load and prepare datasets
            self._load_datasets()
            
            # 5. Setup trainer
            trainer = self._setup_trainer()
            
            # 6. Train
            result = trainer.train()
            
            # 7. Save model
            self._save_model(trainer)
            
            logger.info("Training completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1
    
    def _setup_distributed(self):
        """Setup distributed training if needed"""
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            if len(visible_devices) > 1:
                logger.info(f"Multi-GPU training with devices: {visible_devices}")
                # Let transformers handle DDP automatically
    
    def _load_model_tokenizer(self):
        """Load model and tokenizer"""
        args = self.args
        logger.info(f"Loading model: {args.model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=getattr(args, 'trust_remote_code', False)
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        torch_dtype = getattr(args, 'torch_dtype', 'bfloat16')
        if torch_dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif torch_dtype == 'float16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            trust_remote_code=getattr(args, 'trust_remote_code', False),
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        # Resize token embeddings if needed
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_lora(self):
        """Setup LoRA if specified"""
        args = self.args
        if getattr(args, 'train_type', None) == 'lora':
            logger.info("Setting up LoRA training...")
            
            # Resolve target modules
            target_modules = getattr(args, 'target_modules', 'all-linear')
            if target_modules == 'all-linear':
                target_modules = self._find_all_linear_modules()
            elif isinstance(target_modules, str):
                target_modules = [target_modules]
                
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=getattr(args, 'lora_rank', 8),
                lora_alpha=getattr(args, 'lora_alpha', 32),
                lora_dropout=getattr(args, 'lora_dropout', 0.05),
                target_modules=target_modules,
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info(f"LoRA setup completed. Target modules: {target_modules}")
    
    def _find_all_linear_modules(self) -> List[str]:
        """Find all linear modules in the model"""
        linear_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Skip certain modules
                if any(skip in name for skip in ['lm_head', 'embed_tokens', 'embed_positions']):
                    continue
                module_name = name.split('.')[-1]
                if module_name not in linear_modules:
                    linear_modules.append(module_name)
        
        logger.info(f"Found linear modules: {linear_modules}")
        return linear_modules
    
    def _load_datasets(self):
        """Load and prepare datasets"""
        args = self.args
        dataset_names = getattr(args, 'dataset', [])
        
        if not dataset_names:
            raise ValueError("No dataset specified")
            
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
            
        logger.info(f"Loading datasets: {dataset_names}")
        
        # Load datasets
        all_datasets = []
        for dataset_spec in dataset_names:
            if '#' in dataset_spec:
                dataset_name, sample_count = dataset_spec.split('#')
                sample_count = int(sample_count)
            else:
                dataset_name = dataset_spec
                sample_count = None
                
            # Load from HuggingFace
            try:
                dataset = datasets.load_dataset(dataset_name, split='train')
                if sample_count:
                    dataset = dataset.select(range(min(sample_count, len(dataset))))
                all_datasets.append(dataset)
                logger.info(f"Loaded {dataset_name}: {len(dataset)} samples")
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        if not all_datasets:
            raise ValueError("No datasets could be loaded")
            
        # Combine datasets
        if len(all_datasets) == 1:
            combined_dataset = all_datasets[0]
        else:
            combined_dataset = datasets.concatenate_datasets(all_datasets)
            
        # Format dataset
        self.train_dataset = self._format_dataset(combined_dataset)
        logger.info(f"Training dataset prepared: {len(self.train_dataset)} samples")
        
        # Split validation if needed
        split_ratio = getattr(args, 'split_dataset_ratio', None)
        if split_ratio is not None and split_ratio > 0:
            if len(self.train_dataset) <= 1:
                logger.warning(f"Dataset too small ({len(self.train_dataset)} samples) for splitting, using full dataset for training")
            else:
                split_point = int(len(self.train_dataset) * (1 - split_ratio))
                if split_point <= 0:
                    split_point = 1  # Ensure at least 1 sample for training
                elif split_point >= len(self.train_dataset):
                    split_point = len(self.train_dataset) - 1  # Ensure at least 1 sample for validation
                    
                self.val_dataset = Dataset.from_dict({
                    k: v[split_point:] for k, v in self.train_dataset.to_dict().items()
                })
                self.train_dataset = Dataset.from_dict({
                    k: v[:split_point] for k, v in self.train_dataset.to_dict().items()
                })
                logger.info(f"Split dataset with ratio {split_ratio}: train={len(self.train_dataset)}, val={len(self.val_dataset)} samples")
        else:
            logger.info("No dataset splitting configured, using full dataset for training")
    
    def _format_dataset(self, dataset: Dataset) -> Dataset:
        """Format dataset for training"""
        args = self.args
        
        # Get field names
        prompt_key = getattr(args, 'prompt_key', 'instruction')
        response_key = getattr(args, 'response_key', 'output')
        
        # Check if fields exist
        if prompt_key not in dataset.column_names:
            raise ValueError(f"Prompt key '{prompt_key}' not found in dataset columns: {dataset.column_names}")
        if response_key not in dataset.column_names:
            raise ValueError(f"Response key '{response_key}' not found in dataset columns: {dataset.column_names}")
        
        # Format conversations
        def format_sample(sample):
            prompt = sample[prompt_key]
            response = sample[response_key]
            
            # Add system prompt if specified
            system_prompt = getattr(args, 'system', None)
            if system_prompt:
                text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            
            return {"text": text}
        
        formatted_dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        
        # Tokenize
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=getattr(args, 'max_length', 2048),
                return_tensors=None,
            )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names,
        )
        
        return tokenized_dataset
    
    def _setup_trainer(self) -> Trainer:
        """Setup trainer"""
        args = self.args
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=getattr(args, 'output_dir', './output'),
            num_train_epochs=getattr(args, 'num_train_epochs', 1),
            per_device_train_batch_size=getattr(args, 'per_device_train_batch_size', 1),
            per_device_eval_batch_size=getattr(args, 'per_device_eval_batch_size', 1),
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 16),
            learning_rate=getattr(args, 'learning_rate', 1e-4),
            warmup_ratio=getattr(args, 'warmup_ratio', 0.05),
            logging_steps=getattr(args, 'logging_steps', 10),
            save_steps=getattr(args, 'save_steps', 500),
            eval_steps=getattr(args, 'eval_steps', 500),
            save_total_limit=getattr(args, 'save_total_limit', 2),
            remove_unused_columns=False,
            dataloader_num_workers=getattr(args, 'dataloader_num_workers', 0),
            bf16=getattr(args, 'torch_dtype', 'bfloat16') == 'bfloat16',
            fp16=getattr(args, 'torch_dtype', 'bfloat16') == 'float16',
            eval_strategy="steps" if self.val_dataset else "no",
            load_best_model_at_end=True if self.val_dataset else False,
            report_to=[],  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def _save_model(self, trainer: Trainer):
        """Save the trained model"""
        args = self.args
        output_dir = getattr(args, 'output_dir', './output')
        
        # Save model
        trainer.save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")


def sft_main(args: Union[Namespace, None] = None) -> int:
    """
    Main SFT function - fast startup, clean implementation
    """
    return VerlSft(args).main()


if __name__ == '__main__':
    # For testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, nargs='+', required=True)
    parser.add_argument('--train_type', type=str, default='lora')
    parser.add_argument('--output_dir', type=str, default='./output')
    
    args = parser.parse_args()
    sft_main(args)