# 算法基准

最后更新时间：2025年6月18日。

## 与数学相关的数据集

### GSM8k

GSM8k 是一种音频编解码器，用于对 8 千赫兹（8k）采样率的音频进行编码和解码。GSM8k 通常用于语音通信领域，以实现高效的音频传输和存储。

假设GSM8k/math数据集已通过以下方式预处理：

```bash
python3 examples/data_preprocess/*.py
```

请参考下表，使用不同的预训练检查点来重现强化学习训练。如果没有特别说明，下面是在GSM8k数据集上的性能。更全面的基准结果可在配方文件夹中找到。

| 硬件         | 型号                              | 方法              | 测试得分    | 详情 |
|-------------|----------------------------------|-------------------|--------------|---------|
| NVIDIA GPU  | google/gemma-2-2b-it             | hf 检查点         | 23.9         | [Huggingface](https://huggingface.co/google/gemma-2-2b-it#benchmark-results) |
| NVIDIA GPU  | google/gemma-2-2b-it             | SFT               | 52.06        | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-sft-0.411.log) |
| NVIDIA GPU  | google/gemma-2-2b-it             | SFT + PPO         | 64.02        | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-ppo-bsz512_4-prompt1024-resp-512-0.640.log), [wandb](https://api.wandb.ai/links/verl-team/h7ux8602) |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | hf 检查点         | 36.4         | [Qwen 博客](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | PPO               | 56.7         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log) |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | PRIME             | 58.7         | [脚本](https://github.com/volcengine/verl/blob/main/recipe/prime/run_prime_qwen.sh), [wandb](https://api.wandb.ai/links/zefan-wang-thu-tsinghua-university/rxd1btvb) |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | GRPO-LoRA         | 54.3         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.543.log)|
| NVIDIA GPU  | Qwen/Qwen2.5-1.5B-Instruct       | GRPO-LoRA         | 77.9         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-1.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.779.log)|
| NVIDIA GPU  | Qwen/Qwen2.5-3B-Instruct         | GRPO-LoRA         | 86.1         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-3B-bsz64_2-prompt512-resp1024-lorarank32-score0.861.log)|
| NVIDIA GPU  | deepseek-ai/deepseek-llm-7b-chat | PPO (Megatron)    | 69.5 [1]     | [日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/deepseek-llm-7b-chat-megatron-bsz256_4-prompt512-resp512-0.695.log), [wandb](https://wandb.ai/verl-team/verl_megatron_gsm8k_examples/runs/10fetyr3) |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GRPO              | 89           | [脚本](https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/examples/grpo_trainer/run_qwen2-7b_seq_balance.sh) |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GRPO (FSDP2)      | 89.8         | [日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log) |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GRPO (Megatron)   | 89.6         | [日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b_math_megatron.log) |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | ReMax             | 97           | [脚本](https://github.com/eric-haibin-lin/verl/blob/main/examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh), [wandb](https://wandb.ai/liziniu1997/verl_remax_example_gsm8k/runs/vxl10pln) |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | SPPO              | 65.6 (数学)  | [SPPO 脚本](https://github.com/volcengine/verl/tree/main/recipe/sppo/README.md) |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | GRPO-LoRA         | 93.4         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-7B-bsz64_8-prompt512-resp1024-lorarank32-score0.934.log)|
| NVIDIA GPU  | Mixtral-8x22B-Instruct-v0.1      | Instruct model    | 83.7         | [Qwen 博客](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| NVIDIA GPU  | Mixtral-8x22B-Instruct-v0.1      | RLOO (Megatron)   | 92.3         | [wandb](https://api.wandb.ai/links/ppo_dev/sbuiuf2d) |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | SPIN              | 92           | [脚本](https://github.com/volcengine/verl/tree/main/recipe/spin/README.md) |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GPG               | 88           | [日志](https://github.com/diqiuzhuanzhuan/verldata/blob/main/run_logs/qwen2-7b_math.log), [wandb](https://wandb.ai/diqiuzhuanzhuan/verl_gpg_example_gsm8k_math/runs/ab86c4va) |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GPG (Megatron)    | 88           | [日志](https://github.com/diqiuzhuanzhuan/verldata/blob/main/run_logs/qwen2-7b_math_megatron.log), [wandb](https://wandb.ai/diqiuzhuanzhuan/verl_gpg_example_gsm8k_math/runs/yy8bheu8) |
| NVIDIA GPU  | Qwen/Qwen2.5-VL-7B-Instruct      | GRPO (Megatron)   | 65.4 (地理3k) | [脚本](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl-7b-megatron.sh), [wandb](https://api.wandb.ai/links/megatron-core-moe-dev/1yngvkek) |
| AMD MI300   | deepseek-ai/deepseek-llm-7b-chat | PPO               | 70.5 [1]     | [日志](https://github.com/yushengsu-thu/verl_training_log/blob/main/gsm8k/ppo_run_deepseek7b_llm.log) |
| AMD MI300   | deepseek-ai/deepseek-llm-7b-chat | GRPO              | 71.4 [1]     | [日志](https://github.com/yushengsu-thu/verl_training_log/blob/main/gsm8k/grpo_run_deepseek7b_llm.log) |
| NVIDIA GPU  | Qwen/Qwen2.5-14B-Instruct         | GRPO-LoRA         | 94.6         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-14B-bsz64_8-prompt512-resp1024-lorarank32-score0.946.log)|
| NVIDIA GPU  | Qwen/Qwen2.5-32B-Instruct         | GRPO-LoRA         | 95.8         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-32B-bsz64_8-prompt512-resp1024-lorarank32-score0.958.log)|
| NVIDIA GPU  | Qwen/Qwen2.5-72B-Instruct         | GRPO-LoRA         | 96.0         | [命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-72B-bs64_8-prompt512-resp1024-lorarank32-score0.960.log)|

### DAPO 数学-17k

- 训练 DAPO 数学-17k 数据集: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
- 测试: AIME'24: https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024

注意:
- 对于 Qwen/Qwen2.5-Math-7B，我们直接将 max_position_embeddings 修改为 32768，而没有观察到性能下降，以便训练更长的响应长度。

| 硬件        | 模型                              | 方法              | 测试得分     | 详情    |
|-------------|----------------------------------|-------------------|--------------|---------|
| NVIDIA GPU  | Qwen/Qwen2.5-Math-7B (32k)       | DAPO              | 36.3         | [命令](https://github.com/volcengine/verl/blob/main/recipe/dapo/test_dapo_7b_math.sh), [日志](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/runs/ow47vvon?nw=nwusertongyuxuan361)|

## 与编码相关的数据集

以下是在力扣(LeetCode)上的结果，除非另有说明。

| 硬件        | 型号                               | 方法              | 测试得分     | 详情    |
|-------------|----------------------------------|-------------------|--------------|---------|
| NVIDIA GPU  | PRIME-RL/Eurus-2-7B-SFT          | RPIME             | 36.1         | [脚本](https://github.com/volcengine/verl/blob/main/recipe/prime/run_prime_qwen_code.sh), [swanlab](https://swanlab.cn/@wangzefan/prime_example/runs/7f541qhspgmy8nmhdlx35/chart) |

### 注意事项

[1] 在评估过程中，我们仅提取了遵循格式 `"####"` 的答案。更灵活的答案提取、更长的响应长度和更好的提示工程可能会导致更高的分数。

[2] 自 2025-05-30 起，`actor_rollout_ref.actor.entropy_coeff` 的默认值在 verl 0.3.x 上被设置为 `0.0`，这与先前的版本不同。