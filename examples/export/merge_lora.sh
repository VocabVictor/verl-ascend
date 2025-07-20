# Since `output/vx-xxx/checkpoint-xxx` is trained by verl and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
verl export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true
