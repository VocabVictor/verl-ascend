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
Integrated with Swift's advanced SFT capabilities for multimodal training.
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
        # 直接调用Swift原生SFT命令，保持其强大的多模态能力
        import subprocess
        
        # 设置环境变量解决MKL问题
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'GNU'
        
        # 构建Swift SFT命令
        swift_cmd = ['swift', 'sft']
        
        if args is not None:
            # 将VERL参数转换为Swift参数
            swift_args = _convert_args_to_swift_format(args)
            swift_cmd.extend(swift_args)
        else:
            # 使用原始命令行参数
            swift_cmd.extend(sys.argv[2:])  # 跳过 'verl sft'
            
        # 执行Swift SFT命令
        result = subprocess.run(swift_cmd, cwd='/home/migu/.code/swift', env=env)
        return result.returncode
            
    except Exception as e:
        print(f"Error running VERL SFT: {e}")
        return 1


def _convert_args_to_swift_format(args: Namespace) -> list:
    """
    Convert VERL CLI arguments to Swift SFT format
    
    Args:
        args: Parsed arguments from VERL CLI
        
    Returns:
        List of arguments in Swift format
    """
    swift_args = []
    
    # Map VERL args to Swift args
    arg_mapping = {
        'model': '--model',
        'dataset': '--dataset', 
        'train_type': '--train_type',
        'lora_rank': '--lora_rank',
        'lora_alpha': '--lora_alpha',
        'target_modules': '--target_modules',
        'num_train_epochs': '--num_train_epochs',
        'per_device_train_batch_size': '--per_device_train_batch_size',
        'per_device_eval_batch_size': '--per_device_eval_batch_size',
        'learning_rate': '--learning_rate',
        'gradient_accumulation_steps': '--gradient_accumulation_steps',
        'eval_steps': '--eval_steps',
        'save_steps': '--save_steps',
        'save_total_limit': '--save_total_limit',
        'logging_steps': '--logging_steps',
        'torch_dtype': '--torch_dtype',
        'max_length': '--max_length',
        'output_dir': '--output_dir',
        'warmup_ratio': '--warmup_ratio',
        'dataloader_num_workers': '--dataloader_num_workers',
    }
    
    # Convert arguments
    for verl_arg, swift_arg in arg_mapping.items():
        if hasattr(args, verl_arg):
            value = getattr(args, verl_arg)
            if value is not None:
                swift_args.extend([swift_arg, str(value)])
    
    return swift_args


if __name__ == '__main__':
    sys.exit(sft_main())