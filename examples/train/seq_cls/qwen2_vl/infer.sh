CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
verl infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --load_data_args true
