CUDA_VISIBLE_DEVICES=0 \
verl infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --max_batch_size 16
