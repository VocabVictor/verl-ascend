RL(HF) 算法与 LoRA 支持
===========================================

最后更新：2025年6月5日。

我们支持 LoRA（低秩适应，Low-Rank Adaptation）用于强化学习算法，如 PPO、GRPO 等。

LoRA 是一种参数高效的微调技术，它将可训练的低秩矩阵注入到预训练权重中（通常是线性层）。这减少了内存占用和计算成本，使得在有限硬件上微调大型模型成为可能。

这带来的好处包括：

- 使用适度硬件（例如 8x80G GPU）进行非常大型模型（例如 70B+）的强化学习，
- 由于减少内存使用，能够启用更大的批处理大小，
- 简化模型的迁移和部署，因为只需保存 LoRA 适配器，
- 与 `SLoRA <https://arxiv.org/abs/2311.03285>`_ 或 `CCoE <https://arxiv.org/abs/2407.11686>`_ 等技术结合，以高效服务多个 LoRA 适配器。

本指南解释了如何在强化学习训练中启用 LoRA 并配置相关参数。

使用指南
------------------------
1. LoRA 可在 `verl.trainer.ppo.ray_trainer.RayPPOTrainer` 中使用。示例通过 `verl.trainer.main_ppo` 入口点提供。

目前，LoRA 仅通过 huggingface peft 支持，只能与 fsdp/fsdp2 和 vllm 后端一起使用（sglang 支持即将到来）。

- `strategy=fsdp` 或 `strategy=fsdp2`
- `rollout.name=vllm`

LoRA 的必需配置：

- `actor_rollout_ref.model.lora_rank`：整数，设置为大于 0 的合理值（例如 8、16、32、64）
- `actor_rollout_ref.model.lora_alpha`：浮点数，LoRA 中的 alpha 参数
- `actor_rollout_ref.rollout.load_format="safetensors"`：必需。这使 vLLM 能够加载基础模型。
- `actor_rollout_ref.model.target_modules`：LoRA 的目标模块。通常设置为 "all-linear"。

推荐选项：

- `actor_rollout_ref.model.use_shm=True`：预加载模型到 `/dev/shm` 以提高模型加载速度。
- `actor_rollout_ref.rollout.layered_summon=True`：这使 actor 模型在将 LoRA 适配器与 vLLM 同步时能够按层收集 FSDP 分片，从而降低 GPU 峰值内存。如果模型非常庞大（70B+）或 GPU 内存有限（< 48GB），建议使用。

最佳实践和注意事项
-------------------------

1. **学习率**：建议将学习率的值增加一个数量级。

2. **LoRA Rank**：

- 排名过小可能会影响收敛性。
- @thelongestusernameofall 提供的 LoRA(局部感知自适应调整) 排名建议：

  - 过小的 lora_rank 可能导致收敛速度变慢或训练性能变差。建议将 lora_rank 设置为>=32。测试表明，对于一个0.5B模型，当 lora_rank=32 时，训练收敛速度和最终性能几乎与非 LoRA 训练相同。
  - 对于一个32B模型，当 lora_rank=128 时，训练收敛速度和最终性能也几乎与非 LoRA 训练相同。
  - 更全面的参考结果即将推出。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/f2b80b8b26829124dd393b7a795a0640eff11644/docs/lora.jpg?raw=true

3. 使用8块80GB GPU进行Qwen2.5-72B模型的RL训练参考配置（如有需要，增加lora_rank）:

.. code-block::

```plaintext
data.train_batch_size=64 \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \

示例脚本
-------------------

有关端到端示例，请参考以下脚本：
```

examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
