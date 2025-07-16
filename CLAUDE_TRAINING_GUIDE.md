# VERL训练指南和重要规则

## 🚨 重要规则：输出目录管理

### 严格禁止
1. **绝对不允许在verl-plus源码目录内创建任何输出文件**
   - 不要在verl-plus目录下创建output/、logs/、checkpoints/等目录
   - 不要在源码目录保存训练结果、模型权重、日志文件
   - verl-plus目录必须保持干净，只包含源代码

### 正确做法
1. **所有输出必须放在源码目录之外**
   ```bash
   # 推荐的输出目录结构
   ~/verl-outputs/           # 所有verl相关输出的根目录
   ├── experiments/          # 实验结果
   │   ├── exp1/
   │   ├── exp2/
   │   └── ...
   ├── models/              # 保存的模型
   ├── logs/                # 训练日志
   └── checkpoints/         # 检查点
   ```

2. **运行训练时指定外部输出目录**
   ```bash
   # 错误示例 ❌
   --output_dir output
   
   # 正确示例 ✅
   --output_dir ~/verl-outputs/experiments/exp1
   ```

## 📝 环境配置总结

### Conda环境
- **环境名称**: verl
- **Python版本**: 3.10
- **安装位置**: ~/miniconda3/envs/verl
- **激活命令**: 
  ```bash
  source ~/miniconda3/bin/activate
  conda activate verl
  ```

### 已安装的核心依赖
- PyTorch 2.7.1
- Transformers 4.53.2
- PEFT 0.16.0
- Ray 2.47.1
- Hydra-core 1.3.2
- Wandb 0.21.0
- TensorDict 0.6.2
- Datasets 4.0.0

### 镜像源配置
- **Conda源**: 清华大学镜像
- **Pip源**: 清华大学镜像
- **配置工具**: chsrc (~/chsrc)

## 🖥️ 系统环境

### GPU配置
- **GPU型号**: NVIDIA L20 × 2
- **GPU ID**: 0, 1
- **显存**: 每个46GB
- **CUDA版本**: 12.2
- **驱动版本**: 535.230.02

### 重要路径
- **代码仓库**: ~/verl-plus
- **模型缓存**: ~/.cache/huggingface/hub/
- **建议输出目录**: ~/verl-outputs/

## 🚀 训练脚本示例

### LoRA微调示例（修正版）
```bash
#!/bin/bash
# 正确的训练脚本示例

# 创建输出目录（在verl-plus之外）
mkdir -p ~/verl-outputs/experiments/lora_sft_$(date +%Y%m%d_%H%M%S)

# 激活环境
source ~/miniconda3/bin/activate
conda activate verl

# 运行训练
CUDA_VISIBLE_DEVICES=0,1 \
verl sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train_type lora \
    --dataset 'tatsu-lab/alpaca#200' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ~/verl-outputs/experiments/lora_sft_$(date +%Y%m%d_%H%M%S) \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --prompt_key instruction \
    --response_key output
```

## ⚠️ 注意事项

1. **运行训练前总是检查输出路径**
   - 确保--output_dir不在verl-plus目录内
   - 建议使用绝对路径

2. **GPU数量配置**
   - 系统只有2个GPU (0,1)
   - 不要配置CUDA_VISIBLE_DEVICES=0,1,2,3

3. **后台运行建议**
   ```bash
   # 使用nohup后台运行
   nohup bash your_training_script.sh > ~/verl-outputs/logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

## 📚 常用命令

```bash
# 激活环境
source ~/miniconda3/bin/activate && conda activate verl

# 检查GPU状态
nvidia-smi

# 查看verl帮助
verl --help

# 监控训练进程
ps aux | grep verl

# 查看训练日志（假设日志在正确位置）
tail -f ~/verl-outputs/logs/training_*.log
```

## 🔧 问题排查

1. **如果训练卡住**
   - 检查是否在下载模型（网络问题）
   - 查看GPU内存使用情况
   - 确认输出目录有写入权限

2. **如果找不到verl命令**
   - 确保已激活verl conda环境
   - 检查是否正确安装：`pip show verl`

3. **如果GPU不可用**
   - 检查CUDA_VISIBLE_DEVICES设置
   - 确认GPU驱动正常：`nvidia-smi`

---

**记住：保持源码目录干净，所有输出都放在外部目录！**