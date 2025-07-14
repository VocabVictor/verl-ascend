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
VERL SFT Command - Supervised Fine-Tuning with multi-modal support
Uses VERL's native FSDP SFT trainer for multimodal training.
"""

import os
import sys
from argparse import Namespace
from typing import Union


def sft_main(args: Union[Namespace, None] = None) -> int:
    """
    Main entry point for VERL SFT command
    
    Args:
        args: Parsed arguments from CLI
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    
    try:
        # Use VERL's native FSDP SFT trainer
        from verl.trainer.fsdp_sft_trainer import main as fsdp_sft_main
        
        # Create hydra config from CLI arguments
        config_args = _convert_args_to_hydra_format(args) if args else []
        
        # Set up sys.argv for hydra
        original_argv = sys.argv.copy()
        sys.argv = ['sft_trainer.py'] + config_args
        
        try:
            fsdp_sft_main()
            return 0
        finally:
            # Restore original argv
            sys.argv = original_argv
            
    except Exception as e:
        print(f"Error running VERL SFT: {e}")
        return 1


def _convert_args_to_hydra_format(args: Namespace) -> list:
    """
    Convert VERL CLI arguments to Hydra config format for FSDP SFT trainer
    
    Args:
        args: Parsed arguments from VERL CLI
        
    Returns:
        List of arguments in Hydra format
    """
    hydra_args = []
    
    # Map VERL args to Hydra config overrides
    if hasattr(args, 'model') and args.model:
        hydra_args.append(f'model.partial_pretrain={args.model}')
    
    if hasattr(args, 'dataset') and args.dataset:
        hydra_args.append(f'data.train_files=[{args.dataset}]')
        hydra_args.append(f'data.val_files=[{args.dataset}]')
        # Set correct column mapping for geo3k dataset
        hydra_args.append('data.prompt_key=problem')
        hydra_args.append('data.response_key=answer')
    
    if hasattr(args, 'train_type') and args.train_type == 'lora':
        if hasattr(args, 'lora_rank') and args.lora_rank:
            hydra_args.append(f'model.lora_rank={args.lora_rank}')
        if hasattr(args, 'lora_alpha') and args.lora_alpha:
            hydra_args.append(f'model.lora_alpha={args.lora_alpha}')
        if hasattr(args, 'target_modules') and args.target_modules:
            hydra_args.append(f'model.target_modules=[{args.target_modules}]')
    
    if hasattr(args, 'num_train_epochs') and args.num_train_epochs:
        hydra_args.append(f'trainer.total_epochs={args.num_train_epochs}')
    
    if hasattr(args, 'per_device_train_batch_size') and args.per_device_train_batch_size:
        hydra_args.append(f'data.micro_batch_size_per_gpu={args.per_device_train_batch_size}')
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        hydra_args.append(f'optim.lr={args.learning_rate}')
    
    if hasattr(args, 'max_length') and args.max_length:
        hydra_args.append(f'data.max_length={args.max_length}')
    
    if hasattr(args, 'output_dir') and args.output_dir:
        hydra_args.append(f'trainer.default_local_dir={args.output_dir}')
    
    if hasattr(args, 'warmup_ratio') and args.warmup_ratio:
        hydra_args.append(f'optim.warmup_steps_ratio={args.warmup_ratio}')
    
    if hasattr(args, 'save_steps') and args.save_steps:
        hydra_args.append(f'trainer.save_freq={args.save_steps}')
    
    if hasattr(args, 'logging_steps') and args.logging_steps:
        hydra_args.append(f'trainer.test_freq={args.logging_steps}')
    
    # Disable wandb logging to avoid API key requirement
    hydra_args.append('trainer.logger=[console]')
    
    return hydra_args


if __name__ == '__main__':
    sys.exit(sft_main())