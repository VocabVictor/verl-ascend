性能调优指南
==============================

最后更新：2025年06月23日。

作者： `Guangming Sheng <https://github.com/PeterSH6>`_, `Jiali Zheng <https://github.com/CurryRice233>`_

在本节中，我们将讨论如何调优verl中所有阶段的性能，包括：

1. Rollout生成吞吐量。

2. 启用``use_remove_padding=True``以进行序列打包（即数据打包并移除填充）。

3. 前向和反向计算的批量大小调优。

4. 启用``use_dynamic_bsz=True``以获得更高的吞吐量。

5. 利用Ulysses序列并行进行长上下文训练。

6. LigerKernel用于SFT性能优化。

7. FSDP训练后端中的前向预取。

8. 从logits计算熵的内存优化。

Rollout生成调优
--------------------------

verl目前支持两种rollout后端：vLLM和TGI（即将支持SGLang）。

以下是调优基于vLLM的rollout的关键因素。在调优之前，我们建议设置``actor_rollout_ref.rollout.disable_log_stats=False``以便记录rollout统计信息。

- 增加``gpu_memory_utilization``。

对于 vLLM v0.7.0 及更高版本，vLLM 实例将仅使用**总**内存的 gpu_memory_utilization。
- 对于 SGLang，这是用于**静态**内存（如模型权重和 KV 缓存）的空闲 GPU 内存的分数。然而，剩余的 (1-gpu_memory_utilization) 也将在推断期间使用。

  但是，如果模型参数和优化器状态未被卸载，使用过高的分数可能导致内存溢出。通常，0.5 到 0.7 之间的值在高吞吐量和避免内存溢出之间取得良好平衡。

  注意：由于``gpu_memory_utilization``的定义在不同的推断引擎中有所不同，一个引擎中适用的值可能会导致另一个引擎中的内存溢出。

- 调整``max_num_seqs``或``max_num_batched_tokens``。
  如果日志中 GPU 缓存利用率相对较低，增加``max_num_seqs``或``max_num_batched_tokens``可以扩大解码阶段的有效批处理大小，允许每批处理更多并发请求。
  我们建议设置``max_num_batched_tokens > 2048``以获得更高的吞吐量。

- 使用更小的 ``tensor_parallel_size``。  
  当 GPU 资源允许时，更小的张量并行大小会生成更多的 vLLM 副本。  
  数据并行性 (DP) 可以比张量并行性 (TP) 产生更高的吞吐量，但也会增加 KVCache 的消耗。  
  小心平衡更多副本与更高内存使用之间的权衡。  
  我们在 `HybridFlow paper <https://arxiv.org/pdf/2409.19256v2>`_ 的第 8.4 节中评估了这一权衡。

有关处理抢占和分块预填充的更多调优细节，请参见 `vLLM 官方调优指南 <https://docs.vllm.ai/en/latest/performance/optimization.html>`_。

为了获得最佳性能，我们建议使用 vLLM v0.8.3 或更高版本。有关详细信息，请参见 https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md。

启用去除填充（序列打包）
-----------------------------------------

目前，对于基于 llama、mistral、gemma1 和 qwen 的模型，用户可以启用 `use_remove_padding=True` 来利用 transformers 库提供的序列打包实现。

对于其他模型，transformers库可能也支持它，但我们尚未进行测试。用户可以将所需的模型配置添加到 `test_transformer.py <https://github.com/volcengine/verl/blob/main/tests/models/test_transformer.py#L24>`_ 文件中。并通过运行以下命令来测试其功能：

.. code-block:: bash

```plaintext
pytest -s tests/models/test_transformer.py

如果测试通过，您可以将所需的模型添加到模型 `registry.py <https://github.com/volcengine/verl/blob/main/verl/models/registry.py#L24>`_ 文件中。
然后，您可以享受序列打包带来的性能提升，并欢迎将您测试过的模型提交到 verl 的 PR 中！

批量大小调优
-----------------

为了在经验准备（即模型前向传播）和模型更新（即演员/评论家前向/反向传播）中实现更高的吞吐量，
用户可能需要针对不同的计算调整 ``*micro_batch_size_per_gpu``。

在 verl 中，设置批量大小的核心原则是：

- **算法指标**（训练批量大小，PPO 小批量大小）是 *全局* 的（从单控制器的角度），
  在每个工作节点中进行归一化。请参见 `归一化代码 <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L120-L122>`_。

- **性能相关参数**（微批量大小，动态批量大小的最大令牌长度）是 *局部* 参数，定义每个 GPU 的数据分配。
  请参见 `归一化代码 <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L127>`_。
```

.. 注意:: 在您的训练脚本中，请使用``*每GPU微批大小(micro_batch_size_per_gpu)``而不是``*微批大小(micro_batch_size)``。这样您就不需要考虑``微批大小(micro_batch_size)``的规范化，而``微批大小(micro_batch_size)``将被弃用。

批大小调优提示
""""""""""""""""""""""""

因此，用户可能需要调整``*每GPU微批大小(micro_batch_size_per_gpu)``以加快训练速度。以下是一些提示：

1. **启用梯度检查点(gradient checkpointing)**: 
   设置``actor_rollout_ref.model.enable_gradient_checkpointing=True``和``critic.model.enable_gradient_checkpointing=True``。这通常允许更大的微批大小，并且对于大型小批量训练将是有益的。

2. 将``*每GPU微批大小(micro_batch_size_per_gpu)``增加到与规范化的``小批量大小(mini_batch_size)``相等为止。

3. **使用更大的前向仅参数**：  
   前向仅参数，例如 ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``、``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``、``critic.forward_micro_batch_size_per_gpu`` 可以比与训练相关的微批量大小大（例如，2倍），如 ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``、``critic.ppo_micro_batch_size_per_gpu``。

4. **允许更大的微批量大小用于评论者和奖励模型**：  
   评论者和奖励模型的微批量大小可以大于演员模型。这是因为演员模型在最终层具有更大的词汇表大小。

5. **启用激活卸载**：  
   设置 ``actor_rollout_ref.model.enable_activation_offload=True`` 和 ``critic.model.enable_activation_offload=True``。  
   这通常与梯度检查点一起工作，以获得更大的微批量大小，目前仅在 FSDP 后端可用。

动态批量大小的调优
-----------------------------

动态批量大小是一种技术，允许模型在单次前向传播中处理相似数量的标记（具有不同的实际批量大小）。这可以显著提高训练效率并减少内存使用。

要利用此技术，用户可以在演员（actor）、参考（ref）、评论家（critic）和奖励模型（reward models）中设置 ``use_dynamic_bsz=True``。  
使用 ``use_dynamic_bsz=True`` 时，用户无需调整 ``*micro_batch_size_per_gpu``。  
相反，用户应调整以下参数：

- ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``，``critic.ppo_max_token_len_per_gpu``：  
  在 ``update_policy`` 和 ``update_critic`` 的前向（fwd）和反向（bwd）传播中处理的最大标记数量。

- ``actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`` 和 ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``：  
  在 ``compute_log_prob`` 和 ``compute_ref_log_prob`` 的前向计算中处理的最大标记数量。

- ``critic.forward_micro_batch_size_per_gpu``，``reward_model.forward_micro_batch_size_per_gpu``：  
  在 ``compute_values`` 和 ``compute_rm_score`` 的前向计算中处理的最大标记数量。

动态批量大小调优技巧
""""""""""""""""""""""""""""""

以下是一些调优上述参数的技巧：

1. **增加** ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``  
   将其设置为至少2倍（max_prompt_length + max_response_length）。我们在 `run_qwen2-7b_rm_seq_balance.sh <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh#L25>`_ 中将其设置为3倍。
   尝试增加以获得更高的吞吐量。

2. **仅前向参数可以更大**： 
   与非动态批量情况类似，仅前向令牌限制可以超过前向/后向操作中使用的限制。
 
3. **为评论家和奖励模型使用更大的限制**：
   可以将评论家和奖励参数设置为至少是演员限制的2倍。例如，我们在这里将它们设置为4倍：  
   `run_qwen2-7b_rm_seq_balance.sh <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh#L40>`_
   
.. :math:`\text{critic.ppo_max_token_len_per_gpu}  = 2 \times  \text{actor.ppo_max_token_len_per_gpu})`.

为长上下文训练的尤利西斯序列并行化

要利用此技术，用户可以在 actor、ref、critic 和 reward 模型中设置 ``ulysses_sequence_parallel_size>1``。

我们支持不同模型使用不同的 ulysses_sequence_parallel_size 大小。

为了训练日志序列（>32k），用户可能需要减少 ``*micro_batch_size_per_gpu`` 和 ``*max_token_len_per_gpu`` 以避免 OOM（内存溢出）。

LigerKernel 用于 SFT
----------------------

LigerKernel 是一个高性能的内核，用于监督微调（Supervised Fine-Tuning, SFT），可以提高训练效率。要在您的 SFT 训练中启用 LigerKernel：

1. 通过 ``pip3 install liger-kernel`` 安装 liger-kernel。在您的 SFT 配置文件（例如，``verl/trainer/config/sft_trainer.yaml``）中，设置 ``use_liger`` 参数：

   .. code-block:: yaml

      model:
        use_liger: True  # 启用 LigerKernel 进行 SFT

2. 默认值为 ``False``。仅在您希望使用 LigerKernel 的优化时启用它。

3. LigerKernel 对于提高 SFT 场景中的训练性能特别有用。

FSDP 训练后端中的前向预取
----------------------

在训练阶段，用户可以通过设置 ``fsdp_config.forward_prefetch=True`` 来启用 FSDP 中的前向预取。例如，``actor_rollout_ref.actor.fsdp_config.forward_prefetch=True``。此配置在完成当前前向计算之前预取下一个前向传递的全收集操作，从而将通信与计算重叠，提高效率。有关更多详细信息，请参阅 `FSDP forward_prefetch <https://docs.pytorch.org/docs/stable/fsdp.html#module-torch.distributed.fsdp>`_ 文档。

.. note::
    不支持后向预取，因为 ``BACKWARD_POST`` 策略在嵌套模块情况下可能会错误地进行预取。有关详细信息，请参见 `FSDP 文档 <https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md?plain=1#L70>`_。

从 logits 计算熵的内存优化
----------------------

``logits`` 张量（通常形状为 ``[bsz*seq_len, voc]``）可能会消耗大量内存。当使用 ``compute_entropy_from_logits`` 时，内存使用量大约为 ``[bsz*seq_len, voc] × (4 bytes (float32) + 2 bytes (autocast for softmax+logsumexp) + 1 byte (softmax output))``。

为了减少内存峰值，通过设置以下选项启用分块计算：
``actor_rollout_ref.ref.entropy_from_logits_with_chunking = True``
这将在模型的前向传播过程中以形状为``[chunk_size, voc]``（例如，2048）的块处理张量，而不是处理完整的序列长度。

此外，在训练期间，标准的梯度检查点（``enable_gradient_checkpointing=True``）不适用于熵计算。为了在这种情况下减少内存峰值，请设置：
``actor_rollout_ref.actor.entropy_checkpointing = True``
这将专门为熵计算启用熵的重新计算，从而降低训练期间的内存使用。