CUDA_VISIBLE_DEVICES=0 \
verl infer \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --infer_backend pt \
    --stream true \
    --max_new_tokens 2048
