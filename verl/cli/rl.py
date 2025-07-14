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
VERL RL Command - RLHF (Reinforcement Learning from Human Feedback) training
Integrated with Swift's advanced RLHF capabilities for multimodal and text training.
"""

import os
import sys
import subprocess
from argparse import Namespace
from typing import Union


def rl_main(args: Union[Namespace, None] = None) -> int:
    """
    Main entry point for VERL RL command
    
    Args:
        args: Parsed arguments from CLI
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    
    try:
        # 设置环境变量解决MKL问题
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'GNU'
        
        # 构建Swift RLHF命令
        swift_cmd = ['swift', 'rlhf']
        
        if args is not None:
            # 将VERL参数转换为Swift参数
            swift_args = _convert_args_to_swift_format(args)
            swift_cmd.extend(swift_args)
        else:
            # 使用原始命令行参数
            swift_cmd.extend(sys.argv[2:])  # 跳过 'verl rl'
            
        # 执行Swift RLHF命令
        result = subprocess.run(swift_cmd, cwd='/home/migu/.code/swift', env=env)
        return result.returncode
            
    except Exception as e:
        print(f"Error running VERL RL: {e}")
        return 1


def _convert_args_to_swift_format(args: Namespace) -> list:
    """
    Convert VERL CLI arguments to Swift RLHF format
    
    Args:
        args: Parsed arguments from VERL CLI
        
    Returns:
        List of arguments in Swift format
    """
    swift_args = []
    
    # 核心RLHF参数映射
    core_mapping = {
        # 算法选择
        'rlhf_type': '--rlhf_type',
        'algorithm': '--rlhf_type',  # 别名
        
        # 模型配置
        'model': '--model',
        'model_type': '--model_type',
        'model_revision': '--model_revision',
        'ref_model': '--ref_model',
        'ref_model_type': '--ref_model_type',
        'ref_model_revision': '--ref_model_revision',
        'reward_model': '--reward_model',
        'reward_model_type': '--reward_model_type',
        'reward_model_revision': '--reward_model_revision',
        
        # 训练配置
        'train_type': '--train_type',
        'torch_dtype': '--torch_dtype',
        'max_length': '--max_length',
        'max_completion_length': '--max_completion_length',
        'response_length': '--response_length',
        'num_train_epochs': '--num_train_epochs',
        'per_device_train_batch_size': '--per_device_train_batch_size',
        'per_device_eval_batch_size': '--per_device_eval_batch_size',
        'gradient_accumulation_steps': '--gradient_accumulation_steps',
        'learning_rate': '--learning_rate',
        'warmup_ratio': '--warmup_ratio',
        
        # RLHF特定参数
        'beta': '--beta',
        'label_smoothing': '--label_smoothing',
        'temperature': '--temperature',
        'center_rewards_coefficient': '--center_rewards_coefficient',
        
        # 数据配置
        'dataset': '--dataset',
        'val_dataset': '--val_dataset',
        'split_dataset_ratio': '--split_dataset_ratio',
        
        # LoRA配置
        'lora_rank': '--lora_rank',
        'lora_alpha': '--lora_alpha',
        'lora_dropout': '--lora_dropout',
        'target_modules': '--target_modules',
        'lora_bias': '--lora_bias',
        
        # 输出和日志
        'output_dir': '--output_dir',
        'logging_steps': '--logging_steps',
        'save_steps': '--save_steps',
        'eval_steps': '--eval_steps',
        'save_total_limit': '--save_total_limit',
        
        # 高级配置
        'deepspeed': '--deepspeed',
        'device_map': '--device_map',
        'dataloader_num_workers': '--dataloader_num_workers',
    }
    
    # DPO/CPO/SimPO参数
    dpo_mapping = {
        'rpo_alpha': '--rpo_alpha',
        'cpo_alpha': '--cpo_alpha',
        'simpo_gamma': '--simpo_gamma',
    }
    
    # KTO参数
    kto_mapping = {
        'desirable_weight': '--desirable_weight',
        'undesirable_weight': '--undesirable_weight',
    }
    
    # PPO参数
    ppo_mapping = {
        'num_ppo_epochs': '--num_ppo_epochs',
        'whiten_rewards': '--whiten_rewards',
        'kl_coef': '--kl_coef',
        'cliprange': '--cliprange',
        'vf_coef': '--vf_coef',
        'cliprange_value': '--cliprange_value',
        'gamma': '--gamma',
        'lam': '--lam',
        'num_mini_batches': '--num_mini_batches',
        'local_rollout_forward_batch_size': '--local_rollout_forward_batch_size',
        'num_sample_generations': '--num_sample_generations',
        'missing_eos_penalty': '--missing_eos_penalty',
    }
    
    # GRPO参数
    grpo_mapping = {
        'num_generations': '--num_generations',
        'reward_funcs': '--reward_funcs',
        'reward_weights': '--reward_weights',
        'use_vllm': '--use_vllm',
        'vllm_mode': '--vllm_mode',
        'vllm_server_host': '--vllm_server_host',
        'vllm_server_port': '--vllm_server_port',
        'vllm_tensor_parallel_size': '--vllm_tensor_parallel_size',
        'num_iterations': '--num_iterations',
        'truncation_strategy': '--truncation_strategy',
        'generation_batch_size': '--generation_batch_size',
    }
    
    # GKD参数
    gkd_mapping = {
        'teacher_model': '--teacher_model',
        'teacher_model_type': '--teacher_model_type',
        'teacher_model_revision': '--teacher_model_revision',
        'lmbda': '--lmbda',
    }
    
    # 合并所有映射
    all_mappings = {
        **core_mapping,
        **dpo_mapping,
        **kto_mapping,
        **ppo_mapping,
        **grpo_mapping,
        **gkd_mapping
    }
    
    # 转换参数
    for verl_arg, swift_arg in all_mappings.items():
        if hasattr(args, verl_arg):
            value = getattr(args, verl_arg)
            if value is not None:
                if isinstance(value, bool):
                    if value:  # 只有True时才添加flag
                        swift_args.append(swift_arg)
                elif isinstance(value, list):
                    # 处理列表参数
                    for item in value:
                        swift_args.extend([swift_arg, str(item)])
                else:
                    swift_args.extend([swift_arg, str(value)])
    
    return swift_args


if __name__ == '__main__':
    sys.exit(rl_main())