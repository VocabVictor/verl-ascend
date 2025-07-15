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
        # Choose implementation based on complexity/requirements
        use_native = getattr(args, 'use_native', True)  # Default to native for speed
        
        if use_native:
            # Use fast native implementation (no Swift dependencies)
            from .sft import sft_main as native_sft_main
            return native_sft_main(args)
        else:
            # Use VERL's FSDP SFT trainer for complex scenarios
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
            import subprocess
            
            # Path to VERL FSDP SFT trainer
            sft_script = os.path.join(os.path.dirname(__file__), '..', 'trainer', 'fsdp_sft_trainer.py')
            
            if args:
                # Convert args to Hydra format
                hydra_args = _convert_args_to_hydra_format(args)
                
                # Construct command
                cmd = [sys.executable, sft_script] + hydra_args
                
                # Run the FSDP SFT trainer
                result = subprocess.run(cmd, env=os.environ.copy())
                return result.returncode
            else:
                # Run with default args
                result = subprocess.run([sys.executable, sft_script], env=os.environ.copy())
                return result.returncode
            
    except Exception as e:
        print(f"Error running VERL SFT: {e}")
        return 1


def _convert_args_to_swift_format(args: Namespace) -> list:
    """
    Convert VERL CLI arguments to Swift-style command line format
    
    Args:
        args: Parsed arguments from VERL CLI
        
    Returns:
        List of arguments in Swift command line format
    """
    cmd_args = []
    
    # Direct parameter mapping (no conversion needed)
    if hasattr(args, 'model') and args.model:
        cmd_args.extend(['--model', args.model])
    
    if hasattr(args, 'dataset') and args.dataset:
        cmd_args.append('--dataset')
        if isinstance(args.dataset, list):
            cmd_args.extend(args.dataset)
        else:
            cmd_args.append(args.dataset)
    
    if hasattr(args, 'train_type') and args.train_type:
        cmd_args.extend(['--train_type', args.train_type])
    
    if hasattr(args, 'torch_dtype') and args.torch_dtype:
        cmd_args.extend(['--torch_dtype', args.torch_dtype])
    
    if hasattr(args, 'num_train_epochs') and args.num_train_epochs:
        cmd_args.extend(['--num_train_epochs', str(args.num_train_epochs)])
    
    if hasattr(args, 'per_device_train_batch_size') and args.per_device_train_batch_size:
        cmd_args.extend(['--per_device_train_batch_size', str(args.per_device_train_batch_size)])
    
    if hasattr(args, 'per_device_eval_batch_size') and args.per_device_eval_batch_size:
        cmd_args.extend(['--per_device_eval_batch_size', str(args.per_device_eval_batch_size)])
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        cmd_args.extend(['--learning_rate', str(args.learning_rate)])
    
    if hasattr(args, 'lora_rank') and args.lora_rank:
        cmd_args.extend(['--lora_rank', str(args.lora_rank)])
    
    if hasattr(args, 'lora_alpha') and args.lora_alpha:
        cmd_args.extend(['--lora_alpha', str(args.lora_alpha)])
    
    if hasattr(args, 'target_modules') and args.target_modules:
        cmd_args.extend(['--target_modules', args.target_modules])
    
    if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps:
        cmd_args.extend(['--gradient_accumulation_steps', str(args.gradient_accumulation_steps)])
    
    if hasattr(args, 'eval_steps') and args.eval_steps:
        cmd_args.extend(['--eval_steps', str(args.eval_steps)])
    
    if hasattr(args, 'save_steps') and args.save_steps:
        cmd_args.extend(['--save_steps', str(args.save_steps)])
    
    if hasattr(args, 'save_total_limit') and args.save_total_limit:
        cmd_args.extend(['--save_total_limit', str(args.save_total_limit)])
    
    if hasattr(args, 'logging_steps') and args.logging_steps:
        cmd_args.extend(['--logging_steps', str(args.logging_steps)])
    
    if hasattr(args, 'max_length') and args.max_length:
        cmd_args.extend(['--max_length', str(args.max_length)])
    
    if hasattr(args, 'output_dir') and args.output_dir:
        cmd_args.extend(['--output_dir', args.output_dir])
    
    if hasattr(args, 'warmup_ratio') and args.warmup_ratio:
        cmd_args.extend(['--warmup_ratio', str(args.warmup_ratio)])
    
    if hasattr(args, 'dataloader_num_workers') and args.dataloader_num_workers:
        cmd_args.extend(['--dataloader_num_workers', str(args.dataloader_num_workers)])
    
    # These parameters should work directly with Swift-style dataclass
    if hasattr(args, 'system') and args.system:
        cmd_args.extend(['--system', args.system])
    
    if hasattr(args, 'model_author') and args.model_author:
        cmd_args.extend(['--model_author', args.model_author])
    
    if hasattr(args, 'model_name') and args.model_name:
        cmd_args.extend(['--model_name', args.model_name])
    
    return cmd_args


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
        # Handle multiple datasets - use proper Hydra list syntax
        if isinstance(args.dataset, list):
            # Quote each dataset and join with commas for Hydra list syntax
            datasets_quoted = [f'"{dataset}"' for dataset in args.dataset]
            datasets_str = ','.join(datasets_quoted)
        else:
            datasets_str = f'"{args.dataset}"'
        hydra_args.append(f'data.train_files=[{datasets_str}]')
        hydra_args.append(f'data.val_files=[{datasets_str}]')
    
    # Handle dataset field mapping
    if hasattr(args, 'prompt_key') and args.prompt_key:
        hydra_args.append(f'data.prompt_key={args.prompt_key}')
    else:
        hydra_args.append('data.prompt_key=problem')  # default
        
    if hasattr(args, 'response_key') and args.response_key:
        hydra_args.append(f'data.response_key={args.response_key}')
    else:
        hydra_args.append('data.response_key=answer')  # default
    
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
    
    # Handle system parameter for template override
    if hasattr(args, 'system') and args.system:
        # Store system prompt in a way that can be accessed by the trainer
        hydra_args.append(f'data.system_prompt="{args.system}"')
    
    # Handle model metadata for self-cognition datasets
    if hasattr(args, 'model_author') and args.model_author:
        hydra_args.append(f'data.model_author="{args.model_author}"')
    
    if hasattr(args, 'model_name') and args.model_name:
        hydra_args.append(f'data.model_name="{args.model_name}"')
    
    # Disable wandb logging to avoid API key requirement
    hydra_args.append('trainer.logger=[console]')
    
    return hydra_args


if __name__ == '__main__':
    sys.exit(sft_main())