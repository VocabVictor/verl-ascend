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
VERL Command Line Interface
"""

import argparse
import sys
from typing import Optional, List


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point for VERL"""
    parser = argparse.ArgumentParser(
        prog='verl',
        description='VERL: Volcano Engine Reinforcement Learning for LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # SFT subcommand
    sft_parser = subparsers.add_parser(
        'sft',
        help='Supervised Fine-Tuning with multi-modal support',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add SFT arguments (forwarded to Swift SFT)
    _add_sft_arguments(sft_parser)
    
    # RL subcommand
    rl_parser = subparsers.add_parser(
        'rl',
        help='RLHF (Reinforcement Learning from Human Feedback) training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add RL arguments (forwarded to Swift RLHF)
    _add_rl_arguments(rl_parser)
    
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command == 'sft':
        from .sft import sft_main
        return sft_main(parsed_args)
    elif parsed_args.command == 'rl':
        from .rl import rl_main
        return rl_main(parsed_args)
    elif parsed_args.command is None:
        parser.print_help()
        return 1
    else:
        print(f"Unknown command: {parsed_args.command}")
        return 1


def _add_sft_arguments(parser: argparse.ArgumentParser) -> None:
    """Add SFT-specific arguments"""
    # Core training arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model path or identifier')
    parser.add_argument('--dataset', type=str,
                        help='Dataset name or path')
    parser.add_argument('--train_type', type=str, default='lora',
                        choices=['lora', 'full', 'qlora'],
                        help='Training type')
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--target_modules', type=str, default='all-linear',
                        help='Target modules for LoRA')
    
    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                        help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1,
                        help='Evaluation batch size per device')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    
    # Evaluation and saving
    parser.add_argument('--eval_steps', type=int, default=50,
                        help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Save steps')
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Save total limit')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Logging steps')
    
    # Model arguments
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Torch dtype')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    # Other arguments
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='Dataloader num workers')


def _add_rl_arguments(parser: argparse.ArgumentParser) -> None:
    """Add RL (RLHF) specific arguments"""
    # 算法选择
    parser.add_argument('--rlhf_type', '--algorithm', type=str, default='dpo',
                        choices=['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'],
                        help='RLHF algorithm type')
    
    # 模型配置
    parser.add_argument('--model', type=str, required=True,
                        help='Base model path or identifier')
    parser.add_argument('--model_type', type=str,
                        help='Model type (optional, auto-detected)')
    parser.add_argument('--ref_model', type=str,
                        help='Reference model for comparison')
    parser.add_argument('--reward_model', type=str,
                        help='Reward model for scoring')
    
    # 训练配置
    parser.add_argument('--train_type', type=str, default='lora',
                        choices=['lora', 'full'],
                        help='Training type')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model precision')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--max_completion_length', type=int, default=512,
                        help='Maximum completion length')
    parser.add_argument('--response_length', type=int,
                        help='Response length for generation (PPO/GRPO)')
    
    # 基础训练参数
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                        help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1,
                        help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio')
    
    # RLHF特定参数
    parser.add_argument('--beta', type=float, default=0.1,
                        help='KL regularization coefficient')
    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='Label smoothing')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Generation temperature')
    
    # DPO/CPO/SimPO参数
    parser.add_argument('--rpo_alpha', type=float, default=1.0,
                        help='RPO alpha parameter')
    parser.add_argument('--cpo_alpha', type=float, default=1.0,
                        help='CPO alpha parameter')
    parser.add_argument('--simpo_gamma', type=float, default=1.0,
                        help='SimPO gamma parameter')
    
    # KTO参数
    parser.add_argument('--desirable_weight', type=float, default=1.0,
                        help='Desirable outcome weight')
    parser.add_argument('--undesirable_weight', type=float, default=1.0,
                        help='Undesirable outcome weight')
    
    # PPO参数
    parser.add_argument('--num_ppo_epochs', type=int, default=4,
                        help='PPO epochs per update')
    parser.add_argument('--whiten_rewards', action='store_true',
                        help='Whether to whiten rewards')
    parser.add_argument('--kl_coef', type=float, default=0.05,
                        help='KL regularization coefficient')
    parser.add_argument('--cliprange', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--vf_coef', type=float, default=0.1,
                        help='Value function coefficient')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE lambda')
    
    # GRPO参数
    parser.add_argument('--num_generations', type=int, default=8,
                        help='Generations per prompt (GRPO)')
    parser.add_argument('--reward_funcs', type=str, nargs='+',
                        help='Reward functions list')
    parser.add_argument('--use_vllm', action='store_true',
                        help='Use vLLM for generation')
    parser.add_argument('--num_iterations', type=int, default=1,
                        help='Training iterations')
    
    # GKD参数
    parser.add_argument('--teacher_model', type=str,
                        help='Teacher model for distillation')
    parser.add_argument('--lmbda', type=float, default=0.5,
                        help='Mixing coefficient (GKD)')
    
    # 数据配置
    parser.add_argument('--dataset', type=str,
                        help='Training dataset')
    parser.add_argument('--val_dataset', type=str,
                        help='Validation dataset')
    
    # LoRA配置
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default='all-linear',
                        help='Target modules for LoRA')
    
    # 输出和日志
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--logging_steps', type=int, default=5,
                        help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save frequency')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluation frequency')
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Max checkpoints to keep')
    
    # 高级配置
    parser.add_argument('--deepspeed', type=str,
                        choices=['zero0', 'zero1', 'zero2', 'zero3'],
                        help='DeepSpeed configuration')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='Data loader workers')


if __name__ == '__main__':
    sys.exit(main())