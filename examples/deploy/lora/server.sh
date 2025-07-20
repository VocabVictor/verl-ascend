# Since `swift/test_lora` is trained by verl and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
CUDA_VISIBLE_DEVICES=0 verl deploy \
    --host 0.0.0.0 \
    --port 8000 \
    --adapters lora1=swift/test_lora lora2=swift/test_lora2 \
    --infer_backend vllm
