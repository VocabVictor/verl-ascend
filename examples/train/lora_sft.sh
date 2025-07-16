# 22GB
# qwen3: https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh
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
