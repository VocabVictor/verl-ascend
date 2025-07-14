#!/bin/bash

# Qwen2.5-VL-3B SFT training on Geo3K dataset
# Usage: run_qwen2_5_vl_3b_geo3k.sh <nproc_per_node> [other_configs...]

set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: run_qwen2_5_vl_3b_geo3k.sh <nproc_per_node> [other_configs...]"
    echo "Example: run_qwen2_5_vl_3b_geo3k.sh 2"
    exit 1
fi

nproc_per_node=$1
shift

# 创建时间戳日志目录
LOG_DIR=/home/migu/.code/logs/$(date +%y-%m-%d)/$(date +%H-%M)
mkdir -p "$LOG_DIR"

# 保存PID
echo $$ > "$LOG_DIR/training.pid"

# 配置参数
MODEL_PATH="/home/migu/.code/models/Qwen2.5-VL-3B-Instruct"
DATA_DIR="/home/migu/.code/data/geo3k/data"
SAVE_PATH="$LOG_DIR/checkpoints"

echo "Starting Qwen2.5-VL-3B SFT training on Geo3K dataset"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Save path: $SAVE_PATH"
echo "Log directory: $LOG_DIR"
echo "Number of processes: $nproc_per_node"

# 保存配置信息
cat > "$LOG_DIR/config.txt" << EOF
Training Configuration:
- Model: Qwen2.5-VL-3B-Instruct
- Dataset: Geo3K (geometry reasoning)
- Training files: $DATA_DIR/train-00000-of-00001.parquet
- Validation files: $DATA_DIR/validation-00000-of-00001.parquet
- Save path: $SAVE_PATH
- Processes: $nproc_per_node
- Start time: $(date)
EOF

# 运行训练
micromamba run -n llm torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train-00000-of-00001.parquet" \
    data.val_files="$DATA_DIR/validation-00000-of-00001.parquet" \
    data.prompt_key=problem \
    data.response_key=answer \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    optim.lr=5e-5 \
    optim.warmup_steps_ratio=0.1 \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=geo3k-sft \
    trainer.experiment_name=qwen2.5-vl-3b-geo3k-$(date +%Y%m%d-%H%M) \
    trainer.logger=['console'] \
    trainer.total_epochs=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_hdfs_dir=null \
    $@ 2>&1 | tee "$LOG_DIR/training.log"

# 记录结束时间和状态
EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished at: $(date)" >> "$LOG_DIR/config.txt"
echo "Exit code: $EXIT_CODE" >> "$LOG_DIR/config.txt"

# 清理PID文件
rm -f "$LOG_DIR/training.pid"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Logs saved to: $LOG_DIR"
    echo "Model saved to: $SAVE_PATH"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check logs at: $LOG_DIR/training.log"
fi

exit $EXIT_CODE