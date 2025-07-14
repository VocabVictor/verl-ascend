#!/bin/bash

# Qwen2.5-VL-3B SFT training using VERL SFT (with Swift multimodal capabilities)
# Usage: run_qwen2_5_vl_3b_geo3k_swift.sh

set -x

# 创建时间戳日志目录
LOG_DIR=/home/migu/.code/logs/$(date +%y-%m-%d)/$(date +%H-%M)
mkdir -p "$LOG_DIR"

# 保存PID
echo $$ > "$LOG_DIR/verl_sft_training.pid"

# 配置参数
MODEL_PATH="/home/migu/.code/models/Qwen2.5-VL-3B-Instruct"
DATA_DIR="/home/migu/.code/data/geo3k/jsonl"
SAVE_PATH="$LOG_DIR/checkpoints"

echo "Starting Qwen2.5-VL-3B SFT training with VERL SFT (Swift multimodal capabilities)"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Save path: $SAVE_PATH"
echo "Log directory: $LOG_DIR"

# 保存配置信息
cat > "$LOG_DIR/verl_sft_config.txt" << EOF
VERL SFT Training Configuration:
- Model: Qwen2.5-VL-3B-Instruct
- Dataset: Geo3K (geometry reasoning)
- Training files: $DATA_DIR/train.jsonl
- Validation files: $DATA_DIR/val.jsonl
- Save path: $SAVE_PATH
- Start time: $(date)
- Training method: VERL SFT (with Swift multimodal capabilities)
EOF

# 切换到verl-plus目录
cd /home/migu/.code/verl-plus

# 运行VERL SFT训练（集成Swift多模态能力）
micromamba run -n llm python -m verl.cli.main sft \
    --model "$MODEL_PATH" \
    --dataset "$DATA_DIR/train.jsonl" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir "$SAVE_PATH" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    2>&1 | tee "$LOG_DIR/verl_sft_training.log"

# 记录结束时间和状态
EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished at: $(date)" >> "$LOG_DIR/verl_sft_config.txt"
echo "Exit code: $EXIT_CODE" >> "$LOG_DIR/verl_sft_config.txt"

# 清理PID文件
rm -f "$LOG_DIR/verl_sft_training.pid"

if [ $EXIT_CODE -eq 0 ]; then
    echo "VERL SFT training completed successfully!"
    echo "Logs saved to: $LOG_DIR"
    echo "Model saved to: $SAVE_PATH"
else
    echo "VERL SFT training failed with exit code: $EXIT_CODE"
    echo "Check logs at: $LOG_DIR/verl_sft_training.log"
fi

exit $EXIT_CODE