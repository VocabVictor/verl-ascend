#!/usr/bin/env python
"""显示verl命令的使用帮助"""

print("""
VERL命令行工具使用说明
======================

verl是一个类似ms-swift的命令行工具，支持各种大模型训练方法。

基本用法：
  verl <子命令> [参数]

支持的子命令：
  sft         - 监督微调（Supervised Fine-Tuning）
  pt          - 预训练（Pre-Training）
  rlhf        - 强化学习人类反馈（RLHF）
  infer       - 推理
  eval        - 评估
  deploy      - 部署模型服务
  merge-lora  - 合并LoRA权重
  export      - 导出模型
  app         - 启动Web应用
  web-ui      - 启动Web UI
  rollout     - 批量生成
  sample      - 采样生成

使用示例：
  # 监督微调
  verl sft --model Qwen/Qwen2.5-7B-Instruct --dataset alpaca-gpt4-data-zh
  
  # 多GPU训练
  CUDA_VISIBLE_DEVICES=0,1,2,3 verl sft --model Qwen/Qwen2.5-7B-Instruct --train_type lora
  
  # 推理
  verl infer --model_path ./output/checkpoint-1000
  
  # 评估
  verl eval --model_path ./output/checkpoint-1000 --eval_dataset gsm8k

环境变量：
  CUDA_VISIBLE_DEVICES - 指定使用的GPU
  NPROC_PER_NODE      - 每个节点的进程数（自动使用torchrun）
  NNODES              - 节点数量
  NODE_RANK           - 当前节点编号
  MASTER_ADDR         - 主节点地址
  MASTER_PORT         - 主节点端口

注意：在使用verl之前，请确保已安装所有必要的依赖包，包括torch等。
""")