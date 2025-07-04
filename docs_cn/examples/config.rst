.. _config-explain-page:

配置说明
===================

最后更新：2025年06月18日。

用于RL FSDP后端的ppo_trainer.yaml
-------------------------------------

数据
~~~~

.. code:: yaml

   data:
     tokenizer: null
     train_files: ~/data/rlhf/gsm8k/train.parquet
     val_files: ~/data/rlhf/gsm8k/test.parquet
     prompt_key: prompt
     max_prompt_length: 512
     max_response_length: 512
     train_batch_size: 1024
     return_raw_input_ids: False  # 当策略和rm之间的tokenizer不同的时候，这个应该设置为true
     return_raw_chat: False
     return_full_prompt: False
     shuffle: True
     filter_overlong_prompts: False
     filter_overlong_prompts_workers: 1
     truncation: error
     image_key: images
     trust_remote_code: True
     custom_cls:
        path: null
        name: null

- ``data.train_files``: 训练集 parquet 文件。可以是一个列表或单个文件。程序会将所有文件读取到内存中，因此文件不能太大（< 100GB）。路径可以是本地路径或 HDFS 路径。对于 HDFS 路径，我们提供工具将其下载到 DRAM 并将 HDFS 路径转换为本地路径。
- ``data.val_files``: 验证集 parquet 文件。可以是一个列表或单个文件。
- ``data.prompt_key``: 数据集中提示（prompt）所在的字段。默认值为 'prompt'。
- ``data.max_prompt_length``: 最大提示长度。所有提示将被左填充到该长度。如果长度过长，将报告错误。
- ``data.max_response_length``: 最大响应长度。在强化学习（RL）算法（例如 PPO）中，生成的响应长度最多为此值。
- ``data.train_batch_size``: 不同 RL 算法一个训练迭代中采样的批大小。
- ``data.return_raw_input_ids``: 是否返回原始的 input_ids，而不添加聊天模板。这主要用于适应奖励模型的聊天模板与策略不同的情况。需要先解码，然后应用 RM 的聊天模板。如果使用基于模型的 RM，并且策略和 RM 聊天模板不同，则需要设置此标志。
- ``data.return_raw_chat``: 是否返回原始聊天（提示），而不应用聊天模板。
- ``data.return_full_prompt``: 是否返回带有聊天模板的完整提示。
- ``data.shuffle``: 是否在数据加载器中打乱数据。
- ``data.filter_overlong_prompts``: 默认不进行过滤。
- ``data.filter_overlong_prompts_workers``: 对于大规模数据集，过滤过长提示可能会耗时。您可以设置 ``filter_overlong_prompts_workers`` 以使用多进程加速。默认值为 1。
- ``data.truncation``: 如果输入_ids 或提示长度超过 max_prompt_length，则截断。默认值为 'error'，不允许超过 max_prompt_length。如果抛出错误，用户应增加 max_prompt_length。您还可以设置 ``left`` 和 ``right``。
- ``data.image_key``: 多模态数据集中图像所在的字段。默认值为 'images'。
- ``data.trust_remote_code``: 如果远程分词器有 Python 文件，我们可以使用此字段来允许使用远程分词器。例如：moonshotai/Moonlight-16B-A3B-Instruct

定制数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~

定制数据集扩展已为SFT训练器实现，并可以通过类似的更改扩展到其他训练器。

.. code:: yaml

   custom_cls:
     path: null
     name: null

- ``data.custom_cls.path``: 包含您定制数据集类的文件路径。如果未指定，将使用预先实现的数据集。
- ``data.custom_cls.name``: 指定文件中数据集类的名称。

演员/回放/参考策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

```yaml
actor_rollout_ref:
    hybrid_engine: True
    model:
      path: ~/models/deepseek-llm-7b-chat
      external_lib: null
      override_config:
        model_config: {}
        moe_config:  # 仅适用于Megatron，可以调整moe配置
          freeze_moe_router: False  # 仅适用于Megatron，可以冻结moe路由器（无梯度）
      enable_gradient_checkpointing: False
      enable_activation_offload: False
      trust_remote_code: False
      use_remove_padding: False
    actor:
      strategy: fsdp  # 这是为了向后兼容
      ppo_mini_batch_size: 256
      ppo_micro_batch_size: null # 将被弃用，请使用ppo_micro_batch_size_per_gpu
      ppo_micro_batch_size_per_gpu: 8
      use_dynamic_bsz: False
      ppo_max_token_len_per_gpu: 16384 # n * ${data.max_prompt_length} + ${data.max_response_length}
      grad_clip: 1.0
      clip_ratio: 0.2
      entropy_coeff: 0.0
      use_kl_loss: False # 对于GRPO为True
      use_torch_compile: True # False以禁用torch编译
      kl_loss_coef: 0.001 # 对于grpo
      kl_loss_type: low_var_kl # 对于grpo
      ppo_epochs: 1
      data_loader_seed: null
      shuffle: False
      ulysses_sequence_parallel_size: 1 # sp大小
      optim:
        lr: 1e-6
        lr_warmup_steps: -1 # 优先级。负值表示委托给lr_warmup_steps_ratio。
        lr_warmup_steps_ratio: 0.  # 总步数将在运行时注入
        min_lr_ratio: 0.0   # 仅与余弦学习率调度器一起使用，默认为0.0
        num_cycles: 0.5     # 仅与余弦学习率调度器一起使用，默认为0.5
        warmup_style: constant  # 从constant/cosine中选择
        total_training_steps: -1  # 必须由程序覆盖
      fsdp_config:
        wrap_policy:
          # transformer_layer_cls_to_wrap: None
          min_num_params: 0
        param_offload: False
        optimizer_offload: False
        fsdp_size: -1
      checkpoint:
        # 保存检查点时包含的内容
        # 使用'hf_model'可以将整个模型保存为hf格式，现在仅使用分片模型检查点以节省空间
        save_contents: ['model', 'optimizer', 'extra']
        # 为了更大的灵活性，您可以指定从检查点加载的内容。
        load_contents: ${actor_rollout_ref.actor.checkpoint.save_contents}
    ref:
      fsdp_config:
        param_offload: False
        wrap_policy:
          # transformer_layer_cls_to_wrap: None
          min_num_params: 0
      log_prob_micro_batch_size: null # 将被弃用，请使用log_prob_micro_batch_size_per_gpu
      log_prob_micro_batch_size_per_gpu: 16
      log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
      log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
      ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp大小
    rollout:
      name: vllm
      temperature: 1.0
      top_k: -1 # 0表示hf rollout，-1表示vllm rollout
      top_p: 1
      prompt_length: ${data.max_prompt_length}  # 不用于开源
      response_length: ${data.max_response_length}
      # 对于vllm rollout
      dtype: bfloat16 # 应与FSDP对齐
      gpu_memory_utilization: 0.5
      ignore_eos: False
      enforce_eager: True
      free_cache_engine: True
      load_format: dummy_dtensor
      tensor_model_parallel_size: 2
      max_num_batched_tokens: 8192
      max_num_seqs: 1024
      log_prob_micro_batch_size: null # 将被弃用，请使用log_prob_micro_batch_size_per_gpu
      log_prob_micro_batch_size_per_gpu: 16
      log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
      log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
      # 对于hf rollout
      do_sample: True
      engine_kwargs: # 推理引擎参数
        vllm:
          swap_space: null # null表示“使用引擎默认值”（通常为4 GB），将其设置为，例如，32表示32 GB
          disable_mm_preprocessor_cache: False # 禁用多模型模型的预处理器缓存
        sglang:
          attention_backend: null # null表示使用引擎默认值，可用选项：flashinfer, triton, flashmla
```

```
n: 1 # 对于每个提示，采样 n 个响应（即样本次数）。将其设置为大于 1 的值以进行 grpo、rloo
      val_kwargs:
        # 验证的采样参数
        top_k: -1 # 0 表示 hf rollout，-1 表示 vllm rollout
        top_p: 1.0
        temperature: 0
        n: 1
        do_sample: False # 默认在验证时为 eager

      agent:
        custom_async_server: # 使用自定义异步服务器实现进行 rollout
          path: null
          name: null

**演员、rollout 和参考模型的通用配置**
```

- ``actor_rollout_ref.hybrid_engine``: 是否使用混合引擎，目前仅支持混合引擎
- ``actor_rollout_ref.model.path``: Huggingface 模型路径。这可以是本地路径或 HDFS 路径。对于 HDFS 路径，我们提供工具将其下载到 DRAM 并将 HDFS 路径转换为本地路径。
- ``actor_rollout_ref.model.external_libs``: 需要导入的额外 Python 包。用于将模型或分词器注册到 Huggingface 系统中。
- ``actor_rollout_ref.model.override_config``: 用于覆盖模型的一些原始配置，主要是 dropout。
- ``actor_rollout_ref.model.enable_gradient_checkpointing``: 是否为 actor 启用梯度检查点。
- ``actor_rollout_ref.model.enable_activation_offload``: 是否为 actor 启用激活卸载。
- ``actor_rollout_ref.model.trust_remote_code``: 是否启用加载远程代码模型。

**演员模型**

- ``actor_rollout_ref.actor.strategy``: fsdp 或 megatron。在这个例子中，我们使用 fsdp 后端。

- ``actor_rollout_ref.actor.ppo_mini_batch_size``: 一个样本被拆分为多个子批次，批次大小为 ppo_mini_batch_size，用于 PPO 更新。ppo_mini_batch_size 是所有工作节点/ GPU 的全局数量。

- ``actor_rollout_ref.actor.ppo_micro_batch_size``: [将被弃用，请使用 ppo_micro_batch_size_per_gpu] 类似于梯度累积，单次前向传播的 micro_batch_size_per_gpu，权衡速度与 GPU 内存。该值表示全局视图。

- ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``: 类似于梯度累积，单次前向传播的 micro_batch_size_per_gpu，权衡速度与 GPU 内存。该值表示每个 GPU 的本地数量。

- ``actor_rollout_ref.actor.grad_clip``: 用于演员更新的梯度裁剪。

- ``actor_rollout_ref.actor.use_kl_loss``: 在演员中使用 KL 损失。当使用时，我们不在奖励函数中应用 KL。

- ``actor_rollout_ref.actor.clip_ratio``: PPO 裁剪比率。

- ``actor_rollout_ref.actor.use_torch_compile``: 是否在演员中使用 torch 编译。

- ``actor_rollout_ref.actor.entropy_coeff``: 计算PPO损失时的熵权重。默认值自v0.3.x起更改为0.0。

- ``actor_rollout_ref.actor.ppo_epochs``: 在一组采样数据上进行PPO更新的轮数。

- ``actor_rollout_ref.actor.data_loader_seed``: 从torch 2.6.0开始，Megatron后端可能会获取由pytorch在cp排名之间生成的错误种子，从而导致这些排名之间的数据不对齐，因此我们需要手动设置种子以避免挂起问题。如果``actor_rollout_ref.actor.shuffle``不为null，则必须设置此项。

- ``actor_rollout_ref.actor.shuffle``: 当有多个轮次时，是否对数据进行洗牌。

- ``actor_rollout_ref.actor.optim``: Actor的优化器参数。

- ``actor_rollout_ref.actor.fsdp_config``: Actor训练的FSDP配置。

  - ``wrap_policy``: FSDP包装策略。默认使用Huggingface的包装策略，即通过DecoderLayer进行包装。

    - 不需要设置transformer_layer_cls_to_wrap，因此我们将其注释掉。

  - ``*_offload``: 是否启用参数、梯度和优化器的卸载。

    - 以速度换取GPU内存。

- ``actor_rollout_ref.actor.use_kl_loss``: 是否启用 KL 损失。默认值为 False。

- ``actor_rollout_ref.actor.kl_loss_coef``: KL 损失的系数。默认值为 0.001。

- ``actor_rollout_ref.actor.kl_loss_type``: 支持 ``kl`` (``k1``)、``abs``、``mse`` (``k2``)、``low_var_kl`` (``k3``) 和 ``full``。用于计算演员与参考策略之间的 KL 散度。有关具体选项，请参阅 `kl_penalty()` 在 `core_algos.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py>`_ 中。有关详细分析，请参阅此博客文章：http://joschu.net/blog/kl-approx.html

- ``actor_rollout_ref.actor.checkpoint``: 演员中检查点功能的配置

  - ``save_contents``: 在检查点中保存的内容。默认情况下，我们在检查点中保存模型、优化器和额外信息。
    额外信息目前包括 Rng 状态、FSDP 支持的 lr_scheduler，以及即将推出的 Megatron opt_param_scheduler。
    默认情况下，我们不在检查点中存储 hf_model，但我们在 ``scripts/model_merge.py`` 中提供了一个工具，将检查点格式转换为 hf 格式。

- ``load_contents``：要在检查点中加载的内容，您可以指定不同的检查点加载内容。默认情况下，它与 ``save_checkpoint`` 相同。

**参考模型**

参考模型将在 ``actor.use_kl_loss`` 或/和 ``algorithm.use_kl_in_reward`` 为 True 时启用。

- ``actor_rollout_ref.ref``: FSDP 配置与 actor 相同。**对于大于 7B 的模型，建议默认开启 ref 的 offload**

- ``actor_rollout_ref.ref.log_prob_micro_batch_size``: [将被弃用，请使用 log_prob_micro_batch_size_per_gpu] 在计算 ``ref_log_prob`` 时，单次前向传播的批量大小。该值表示全局数量。

- ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``: 在计算 ``ref_log_prob`` 时，单次前向传播的批量大小。该值表示每个 GPU 的本地数量。

**推广模型**

- ``actor_rollout_ref.rollout.name``: hf/vllm/sglang.

- Rollout (自回归) 参数。键应与 vLLM 的 ``SamplingParams`` 中的属性名称相等。

  - ``temperature``、``top_k``、``top_p`` 等：``SamplingParams`` 中的采样参数。

- ``actor_rollout_ref.rollout.dtype``: Rollout 模型参数类型。这应与 FSDP/Megatron 后端中的 actor 模型参数类型对齐。

- ``actor_rollout_ref.rollout.gpu_memory_utilization``:

- 对于 vLLM v0.7.0 及更高版本：用于 vLLM 实例的 **总** GPU 内存的比例。
  - 对于 SGLang：对应于 ``mem_fraction_static``，用于 **静态** 内存（如模型权重和 KV 缓存）的空闲 GPU 内存的比例。

- ``actor_rollout_ref.rollout.tensor_model_parallel_size``：用于 rollout 的 TP 大小。仅对 vllm 有效。

- ``actor_rollout_ref.rollout.log_prob_micro_batch_size``：[将被弃用，请使用 log_prob_micro_batch_size_per_gpu] 在计算 ``log_prob`` 时一次前向传播的批量大小。该值表示全局数量。

- ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``：每个 GPU 的微批量大小（一次前向传播的批量大小），用于重新计算 ``log_prob``。该值表示每个 GPU 的本地数量。

- ``actor_rollout_ref.rollout.do_sample``：在训练 rollout 期间是否进行采样。如果设置为 False，rollout 模型将执行贪婪采样。

- ``actor_rollout_ref.rollout.val_kwargs``：在验证期间专门使用的采样参数。

- ``top_k``: Top-k 采样参数。默认值为 -1（用于 vLLM rollout）或 0（用于 HF rollout）。
  - ``top_p``: Top-p 采样参数。默认值为 1.0（禁用）。
  - ``temperature``: 采样温度。默认值为 0（确定性贪婪）。
  - ``n``: 在验证期间生成的响应数量。默认值为 1。
  - ``do_sample``: 在验证期间是否使用采样。默认值为 False（确定性输出）。当设置为 True 时，rollout 将使用 ``actor_rollout_ref.rollout.val_kwargs`` 参数（top_k、top_p、temperature）来控制采样行为。

- ``actor_rollout_ref.rollout.engine_kwargs.vllm``: 额外的 vllm 引擎参数

  - ``swap_space``: 推理引擎使用的交换空间（以 GB 为单位）。正整数，例如，``32`` 表示 32 GB。``null``: 表示不设置并使用引擎默认值（通常，例如，vLLM 的默认值为 4 GB）。
  - ``disable_mm_preprocessor_cache``: 是否禁用多模型的预处理器缓存。

- ``actor_rollout_ref.rollout.engine_kwargs.sglang``: 额外的 sglang 引擎参数

  - ``attention_backend``: 用于推理引擎的注意力后端。

- ``null``: 表示不设置并使用引擎的默认值（通常，例如，``fa3`` 用于 SGLang）
    - ``flashinfer``: 使用 flashinfer 注意力后端。
    - ``triton``: 使用 triton 注意力后端。
    - ``flashmla``: 使用 flashmla 注意力后端。

- ``actor_rollout_ref.rollout.ignore_eos``: 是否忽略 EOS
  令牌，并在生成 EOS 令牌后继续生成令牌。

- ``actor_rollout_ref.rollout.free_cache_engine``: 在 rollout 生成阶段后卸载 KVCache。
  默认值为 True。当设置为 True 时，对于 vllm v0.5.4 和 v0.6.3，我们需要禁用 CUDAGraph 的使用
  （将 ``enforce_eager`` 设置为 True。）

- ``actor_rollout_ref.rollout.enforce_eager``: 是否在 vLLM 生成中使用 CUDAGraph。
  默认设置为 True，以禁用 CUDAGraph。

- ``actor_rollout_ref.rollout.load_format``: 使用哪个权重加载器
  将 actor 模型权重加载到 rollout 模型中。

- ``auto``: 使用 Megatron 权重加载器。
  - ``megatron``: 使用 Megatron 权重加载器。与 Megatron 后端一起部署。输入模型 ``state_dict()`` 已经沿 TP 维度进行了分区，并且已经沿 PP 维度进行了聚合。此权重加载器要求 Rollout 模型和 Actor 模型的参数形状和名称必须相同。
  - ``dtensor``: 使用 Huggingface 权重加载器时的默认解决方案。与 FSDP 后端一起部署，且 state_dict_type 为 ``StateDictType.SHARDED_STATE_DICT``。推荐使用此权重加载器。
  - ``hf``: 使用 Huggingface 权重加载器。与 FSDP 后端一起部署，且 state_dict_type 为 ``StateDictType.FULL_STATE_DICT``。此解决方案不需要为 vLLM 中实现的每个模型重写权重加载器，但会导致更大的峰值内存使用。
  - ``dummy_hf``, ``dummy_megatron``, ``dummy_dtensor``: 随机初始化。

.. note:: **注意**: 在此配置字段中，用户只需从 ``dummy_megatron``、``dummy_dtensor``、``dummy_hf`` 中选择用于回放初始化，我们的混合引擎将在演员/回放权重同步期间选择相应的权重加载器（即 ``megatron``、``dtensor``、``hf``）。

Megatron 优化器和优化器参数调度器
____________________________________________________

.. code:: yaml

```yaml
optim:
      optimizer: adam
      lr: 1e-6
      clip_grad: 1.0
      total_training_steps: -1  # 必须由程序覆盖
      lr_warmup_init: 0.0  # 预热的初始学习率，默认为0.0
      lr_warmup_steps: -1 # 优先级。负值意味着委托给 lr_warmup_steps_ratio。
      lr_warmup_steps_ratio: 0.  # 在运行时将注入的总步骤
      lr_decay_steps: null
      lr_decay_style: constant # 从 constant/linear/cosine/inverse_square_root 中选择
      min_lr: 0.0 # 最小学习率，默认为0.0
      weight_decay: 0.01
      weight_decay_incr_style: constant # 从 constant/linear/cosine 中选择
      lr_wsd_decay_style: exponential # 从 constant/exponential/cosine 中选择
      lr_wsd_decay_steps: null
      use_checkpoint_opt_param_scheduler: False # 使用检查点优化器参数调度器

注意，Megatron 优化器和 FSDP 优化器之间的 API 存在一些差异。
```

- Megatron 优化器调度器将 lr_warmup 之后的周期命名为 lr_decay_steps，因此 ``warmup_style`` 实际上指的是 warmup 之后的学习率衰减方式。
- Megatron 优化器还支持权重衰减机制。
- ``use_checkpoint_opt_param_scheduler`` 决定是否使用检查点优化器参数调度器。如果设置为 True，优化器参数调度器将在检查点中保存，并在恢复训练时从检查点加载。

对于学习率衰减，原始 Megatron 预训练的默认选项 ``lr_decay_style`` 为 ``linear``，这意味着学习率将在 ``lr_decay_steps`` 内从初始学习率线性衰减到 ``min_lr``。然而，在 verl 中，为了与 FSDP 的默认行为保持一致，我们将默认的 ``lr_decay_style`` 设置为 ``constant``，这意味着学习率将在 warmup 阶段后保持不变。

Critic 模型
~~~~~~~~~~~~

Critic 的大多数参数与 Actor 模型相似。

Reward 模型
~~~~~~~~~~~~

.. code:: yaml

reward_model:
     enable: False
     model:
       input_tokenizer: ${actor_rollout_ref.model.path}  # 如果聊天模板相同，请将其设置为null
       path: ~/models/Anomy-RM-v0.1
       external_lib: ${actor_rollout_ref.model.external_lib}
       trust_remote_code: False
       fsdp_config:
         min_num_params: 0
         param_offload: False
     micro_batch_size_per_gpu: 16
     max_length: null
     reward_manager: naive

- ``reward_model.enable``: 是否启用奖励模型。如果为False，我们仅使用用户定义的奖励函数来计算奖励。在GSM8K和数学示例中，我们禁用奖励模型。对于使用full_hh_rlhf的RLHF对齐示例，我们利用奖励模型来评估响应。如果为False，以下参数将无效。
- ``reward_model.model``

- ``input_tokenizer``: 输入分词器。如果奖励模型的聊天模板与策略不一致，我们需要先解码为明文，然后应用奖励模型的聊天模板。接着使用奖励模型进行评分。如果聊天模板一致，可以设置为 null。
  - ``path``: 奖励模型的 HDFS 路径或本地路径。请注意，奖励模型仅支持 AutoModelForSequenceClassification。其他模型类型需要定义自己的 RewardModelWorker 并从代码中传递。
  - ``trust_remote_code``: 是否启用加载远程代码模型，默认为 False。
- ``reward_model.reward_manager``: 奖励管理器。这定义了基于规则的奖励计算机制以及处理不同奖励来源的方式。默认值为 ``naive``。如果所有验证函数都是多进程安全的，奖励管理器可以设置为 ``prime`` 以进行并行验证。

自定义奖励函数
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml
  
   custom_reward_function:
     path: null
     name: compute_score

- ``custom_reward_function.path``：包含您自定义奖励函数的文件路径。如果未指定，将使用预先实现的奖励函数。
- ``custom_reward_function.name``（可选）：指定文件中奖励函数的名称。默认值为 'compute_score'。

算法
~~~~~~~

.. code:: yaml

   algorithm:
     gamma: 1.0
     lam: 1.0
     adv_estimator: gae
     use_kl_in_reward: False
     kl_penalty: kl  # 如何估计kl散度
     kl_ctrl:
       type: fixed
       kl_coef: 0.005
       horizon: 10000
       target_kl: 0.1

- ``gamma``: 折扣因子
- ``lam``: GAE估计器中偏差和方差之间的权衡
- ``adv_estimator``: 支持 ``gae``、``grpo``、``reinforce_plus_plus``、``reinforce_plus_plus_baseline``、``rloo``
- ``use_kl_in_reward``: 是否启用奖励中的kl惩罚。默认值为False。
- ``kl_penalty``: 支持 ``kl``、``abs``、``mse``、``low_var_kl`` 和 ``full``。如何计算演员和参考策略之间的kl散度。有关具体选项，请参阅 `kl_penalty()` 在 `core_algos.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py>`_ 中的说明。
- ``kl_ctrl``: 奖励中kl惩罚控制器的配置
  - ``kl_coef``: 奖励中kl惩罚的（初始）系数。默认值为0.001。
  - ``type``: 'fixed' 表示固定的KL控制器，'adaptive' 表示自适应的KL控制器。
  - ``horizon`` 和 ``target_kl``: 有关详细信息，请参阅自适应KL控制器的源代码。

训练器
~~~~~~~

```yaml
trainer:
  total_epochs: 30
  project_name: verl_examples
  experiment_name: gsm8k
  logger: ['console', 'wandb']
  log_val_generations: 0
  nnodes: 1
  n_gpus_per_node: 8
  save_freq: -1
  val_before_train: True
  test_freq: 2
  critic_warmup: 0
  default_hdfs_dir: ~/experiments/gsm8k/ppo/${trainer.experiment_name} # hdfs检查点路径
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name} # 本地检查点路径
  resume_mode: auto # 或者disable或者resume_path（如果设置了resume_from_path）
  resume_from_path: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  ray_wait_register_center_timeout: 300
```

- ``trainer.total_epochs``: 训练中的总epoch数。
- ``trainer.project_name``: 用于wandb、swanlab、mlflow的项目名称。
- ``trainer.experiment_name``: 用于wandb、swanlab、mlflow的实验名称。
- ``trainer.logger``: 支持控制台以及wandb、swanlab、mlflow、tensorboard。
- ``trainer.log_val_generations``: 在验证期间记录的代数数量（默认为``0``）。
- ``trainer.nnodes``: 训练中使用的节点数。
- ``trainer.n_gpus_per_node``: 每个节点的GPU数量。
- ``trainer.save_freq``: 保存actor和critic模型检查点的频率（按迭代计算）。
- ``trainer.val_before_train``: 是否在训练之前运行验证。
- ``trainer.test_freq``: 验证频率（按迭代计算）。
- ``trainer.critic_warmup``: 在实际策略学习之前训练评论者模型的迭代次数。
- ``trainer.resume_mode``: 恢复训练的模式。支持``disable``、``auto``和``resume_path``。如果设置为默认的``auto``，程序将自动从``default_local_dir``中的最新检查点恢复。如果设置为``resume_path``，程序将从``resume_from_path``指定的路径恢复。
- ``trainer.resume_from_path``: 恢复训练的路径。仅在``resume_mode``设置为``resume_path``时有效。
- ``trainer.remove_previous_ckpt_in_save``: 是否删除保存目录中的先前检查点。默认为False。
- ``trainer.del_local_ckpt_after_load``: 是否在加载后删除本地检查点。默认为False。
- ``trainer.ray_wait_register_center_timeout``: 等待ray注册中心准备就绪的超时时间。默认为300秒。

这幅图示了配置对训练的影响。

https://excalidraw.com/#json=pfhkRmiLm1jnnRli9VFhb,Ut4E8peALlgAUpr7E5pPCA

.. image:: https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d

evaluation.yaml
---------------

评估配置文件(evaluation.yaml)用于定义评估任务的相关参数和设置。在这个文件中，您可以指定评估任务的数据集、评估指标、评估频率等。以下是一个示例评估配置文件的结构：

```yaml
evaluation:
  dataset: data/evaluation_dataset.csv
  metrics: 
    - accuracy
    - precision
    - recall
  frequency: daily
```

在这个示例中，评估配置文件指定了评估任务使用的数据集为"data/evaluation_dataset.csv"，评估指标包括准确率、精确率和召回率，评估频率为每天一次。您可以根据实际需求修改这些参数以满足您的评估任务要求。

数据
~~~~

```yaml
数据:
  路径: /tmp/math_Qwen2-7B-Instruct.parquet
  提示键: prompt
  响应键: responses
  数据源键: data_source
  奖励模型键: reward_model

- ``data.path``: 数据集文件的路径（Parquet格式）。
- ``data.prompt_key``: 数据集中包含提示的字段。默认为'prompt'。
- ``data.response_key``: 该键保存生成的响应。这应该是表示响应的字符串列表。默认为'responses'。
- ``data.data_source_key``: 用于区分不同数据源的度量计算，确保为每个源独立计算度量。
- ``data.reward_model_key``: 该键保存参考答案。这些参考答案通常用作任务的基准或测试用例。

自定义奖励函数
~~~~~~~~~~~~~~

```yaml
自定义奖励函数:
  路径: null
  名称: compute_score
```

```rst
sft_trainer.yaml for SFT FSDP 后端

- ``custom_reward_function.path``: 自定义奖励函数文件的路径。如果未指定，将使用预先实现的奖励函数。
- ``custom_reward_function.name`` (可选) : 指定文件中的奖励函数的名称。默认值为'compute_score'。
```

优化器

```yaml
优化器配置:
  学习率: 1e-5
  权重衰减: 0.01
  热身步数比例: 0.1
  梯度裁剪: 1.0
  学习率调度器: 余弦

- ``optim.lr``: 优化器的学习率。
- ``optim.weight_decay``: 优化器的权重衰减。
- ``optim.warmup_steps_ratio``: 热身步数占总训练步数的比例。
- ``optim.clip_grad``: 梯度裁剪值。
- ``optim.lr_scheduler``: 学习率调度器类型。选项:

  - ``cosine``: 余弦学习率调度器带有热身（默认）。
  - ``wsd``: 热身稳定衰减调度器，在热身和衰减阶段之间提供稳定的学习率阶段。
```

模型
~~~~~~~~~~~~

大多数Model的参数与Reward Model类似。

```yaml
model:
  partial_pretrain: ~/models/gemma-1.1-7b-it
  fsdp_config:
    model_dtype: fp32
    wrap_policy:
      min_num_params: 0
    cpu_offload: False
    offload_params: False
  external_lib: null
  enable_gradient_checkpointing: False
  trust_remote_code: False
  lora_rank: 0
  lora_alpha: 16
  target_modules: all-linear
  use_liger: False
```

- ``partial_pretrain``: 预训练模型的HDFS路径或本地路径。
- ``fsdp_config``

  - ``model_dtype``: 模型参数类型，默认为``fp32``。
    支持：``bf16``，``fp16``，``fp32``。
  - ``cpu_offload``: 是否为FSDP启用CPU卸载。如果为True，
    将使用offload_params作为参数。
  - ``offload_params``: 是否将参数卸载到CPU
    当未参与计算时。如果为True，则将梯度也卸载到CPU，这意味着优化器步骤在CPU上运行。

- ``lora_rank``: LoRA模型的秩， 默认值为0。如果 ``lora_rank``>0，我们将训练LoRA模块而不是调整整个模型。
- ``lora_alpha``: LoRA缩放的alpha参数，默认值为16。
- ``target_modules``: 要应用适配器的模块名称，默认为 ``all-linear``。详细信息请参见`peft文档 <https://huggingface.co/docs/peft/v0.15.0/zh/package_reference/lora#peft.LoraConfig.target_modules>`_。
- ``use_liger``: 是否启用Liger内核，默认为False。如果为True，我们将在模型上应用Liger内核（取决于 `liger-kernel`）。