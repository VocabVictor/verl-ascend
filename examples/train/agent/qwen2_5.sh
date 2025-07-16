# 35GiB - Agent training with Qwen2.5-3B
CUDA_VISIBLE_DEVICES=0,1 \
verl sft \
    --model Qwen/Qwen2.5-3B \
    --train_type full \
    --dataset 'glaiveai/glaive-function-calling-v2#1000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir ~/verl-outputs/experiments/agent_qwen25_$(date +%Y%m%d_%H%M%S) \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --prompt_key system \
    --response_key chat
