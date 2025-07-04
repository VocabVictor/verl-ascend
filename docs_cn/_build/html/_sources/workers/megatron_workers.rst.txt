Megatron-LM 后端
===================

最后更新：2025年6月24日。

我们通过实现各种工作者来支持 Megatron 后端，包括演员（actor）、评论家（critic）、参考（reference）、回滚（rollout）和奖励模型（reward models）。我们还在 `megatron_vllm.py <https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/megatron_vllm.py>`_ 和 `megatron_sglang.py <https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/megatron_sglang.py>`_ 中使用 Megatron-LM 和 vLLM/SGLang 实现了 ``3DHybridEngine``。

**Pros**

- 支持5维并行（TP、EP、CP、DP、PP）和序列并行，以实现最佳的可扩展性和吞吐量。
- 3D HybridEngine可以显著减少峰值内存使用，并降低演员与回滚之间的权重同步开销。

**Cons**

- Huggingface 模型和 Megatron 检查点需要转换工具。

开发进展
--------------------

请注意，[已弃用(Deprecated)] 意味着该功能在最新版本的 verl 中不再支持。  
[待优化(To-Optimize)] 意味着该功能已实现但尚未优化。  
[进行中(WIP)] 意味着该功能正在开发中。  
[已发布(In-Release)] 意味着该功能已准备就绪并正在审核过程中，随时可能发布。

+---------------+-----------------------------------------------------------+
| [已弃用]      | Megatron 3D 并行处理自定义模型                           |
+---------------+-----------------------------------------------------------+
| [完成]        | Megatron 0.11.0 ``GPTModel`` 支持                        |
+---------------+-----------------------------------------------------------+
| [完成]        | Megatron GRPO 支持                                       |
+---------------+-----------------------------------------------------------+
| [完成]        | Megatron 与 vLLM 0.8.2，支持每个张量权重加载            |
+---------------+-----------------------------------------------------------+
| [完成]        | Megatron 与上下文并行                                   |
+---------------+-----------------------------------------------------------+
| [完成]        | Qwen2MoE 模型支持                                       |
+---------------+-----------------------------------------------------------+
| [待优化]      | Megatron 分布式检查点                                   |
+---------------+-----------------------------------------------------------+
| [待优化]      | Huggingface 和 Megatron 检查点转换器                   |
+---------------+-----------------------------------------------------------+
| [待优化]      | 高效融合线性、熵和交叉熵                               |
+---------------+-----------------------------------------------------------+
| [完成]        | Megatron 卸载（参数、梯度、优化器）                     |
+---------------+-----------------------------------------------------------+
| [完成]        | Megatron 性能分析器                                     |
+---------------+-----------------------------------------------------------+
| [发布中]      | Megatron 0.12.0，TE 2.2 与 vLLM 0.8.3 和融合注意力     |
+---------------+-----------------------------------------------------------+
| [进行中]      | Moonlight/DeepSeek-V3 模型支持                          |
+---------------+-----------------------------------------------------------+
| [进行中]      | 专家并行支持                                           |
+---------------+-----------------------------------------------------------+
| [进行中]      | Megatron 支持动态批量大小                              |
+---------------+-----------------------------------------------------------+
| [待办]        | 性能调优                                              |
+---------------+-----------------------------------------------------------+
| [里程碑]      | 可与 DeepSeek-V3 671B 后训练运行                       |
+---------------+-----------------------------------------------------------+

Megatron Workers 的工具
-------------------------

MegatronWorker
^^^^^^^^^^^^^^

``MegatronWorker`` 是不同 Megatron 工作类的基类。在这个类中，``get_megatron_global_info`` 和 ``get_megatron_rank_info`` 函数用于检索在特定 GPU 上运行的每个 ``Worker`` 的 3D 并行世界大小和排名。这些信息将用于 Megatron 后端的传输协议。

以下的 ``Worker`` 类将用于构建 ``WorkerGroup``，适用于不同的模型。

我们为每个 ``Worker`` 类实现了各种 API，这些 API 由 ``@register(dispatch_mode=)`` 装饰器修饰。这些 API 可以被 ray 驱动进程调用。数据可以根据每个函数的 ``dispatch_mode`` 正确收集和分发。支持的 dispatch_model（即传输协议）可以在 `decorator.py <https://github.com/volcengine/verl/blob/main/verl/single_controller/base/decorator.py>`_ 中找到。

ActorRolloutRefWorker
^^^^^^^^^^^^^^^^^^^^^

此类是为 Actor/Rollout HybridEngine 或参考模型实现的，用于初始化它们的模型并执行计算。

Actor/Rollout HybridEngine
''''''''''''''''''''''''''

1. HybridEngine、Actor 和 Rollout 初始化 API。

.. code:: python

   @register(dispatch_mode=Dispatch.ONE_TO_ALL)
   def init_model(self):

``ONE_TO_ALL``：当从驱动程序进程调用 ``init_model`` 函数时，每个工作节点（在 GPU 上）将执行以下模型初始化过程。

HybridEngine、Actor 和 Rollout 的初始化细节如下：

1. ``MegatronPPOActor`` 实现了简单的 PPO 计算逻辑，当模型与 Megatron 构建时，包括计算对数概率、模型更新。
2. ``vLLMRollout`` 支持使用 vLLM 生成。我们修改了 vLLM 引擎，并使其在 SPMD 下执行，以适应我们的 ``WorkerGroup`` 设计。
3. ``MegatronVLLMShardingManager`` 是一个上下文管理器，用于在 Actor 和 Rollout 之间执行实际的重分片。

有关更多信息，请参见 `源代码 <https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py#L63>`_。

.. code:: python

# 构建演员模型
   self.actor = MegatronPPOActor(config=self.config.actor,
                                 model_config=self.actor_model_config,
                                 megatron_config=megatron_config,
                                 actor_module=self.actor_module,
                                 actor_optimizer=self.actor_optimizer,
                                 actor_optimizer_config=self.actor_optim_config)

   # 构建回滚
   # 回滚初始化
   rollout = vLLMRollout(actor_module=params,
                        config=self.config.rollout,
                        tokenizer=self.tokenizer,
                        model_hf_config=self.actor_model_config,
                        train_tp=mpu.get_tensor_model_parallel_world_size())
   # 在演员和回滚之间执行权重重分配
   sharding_manager = MegatronVLLMShardingManager(module=self.hybrid_engine,
                                                  inference_engine=rollout.inference_engine,
                                                  model_config=self.actor_model_config,
                                                  layer_name_mapping=layer_name_mapping)
   ...

1. 生成序列并重新计算对数概率

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
   def generate_sequences(self, prompts: DataProto):

- ``Dispatch.MEGATRON_PP_AS_DP_PROTO``：演员模型的PP维度将被视为DP维度。然后驱动进程将根据这种重组来调度和收集数据。这是因为在HybridEngine中，演员权重通常应用于更大的3D并行规模，将沿着PP维度和TP维度进行聚集。因此，相应的数据应该通过回放模型的3D并行组进行调度和收集，而不是演员模型。然而，世界大小和排名信息只能通过``get_megatron_global_info``和``get_megatron_rank_info``获取，这些信息记录了演员模型的3D信息。此外，TP维度内的数据重分片将在HybridEngine内处理。

- 在此函数中，回放模型将执行自回归生成，演员模型将重新计算生成响应的旧对数概率。

3. 更新演员模型

.. code:: python

```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
```

- ``Dispatch.MEGATRON_COMPUTE_PROTO``：用户传递按数据并行(DP)维度划分的数据。数据被分发到同一数据并行组内的所有张量并行(tp)/模型并行(pp)排名，最终仅从tp=0和最后一个pp收集输出数据。
- 使用PPO（策略梯度优化）和熵损失更新演员模型权重。


..note:: 

目前，训练的张量并行大小（Tensor Parallel Size）可能与推理的张量并行大小不同。

参考模型
''''''''''''''

1. 参考模型初始化

参考模型使用与演员模型相同的函数进行初始化，但不初始化 HybridEngine 和优化器。然后，演员模型也被包装在 ``MegatronPPOActor`` 中。

2. 计算参考对数概率

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_ref_log_prob(self, data: DataProto):

- 在此函数中，参考模型将调用 ``MegatronPPOActor`` 中的计算对数概率函数来计算参考对数概率。

CriticWorker 和 RewardWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 模型初始化

与参考模型非常相似。CriticWorker 将为优化器执行额外的初始化。

2. 计算 CriticWorker 的值

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_values(self, data: DataProto):

3. 更新 Critic

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def update_critic(self, data: DataProto):

4. 计算奖励

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_rm_score(self, data: DataProto):

训练优化的工具
---------------------------

卸载
^^^^^^^
当资源紧张时，卸载方法可以降低 GPU 内存使用，帮助训练和推理框架在 verl 下良好运行。它将参数、梯度和优化器移动到 CPU 内存中，仅在需要时再加载回 GPU。

如果您想使用卸载功能，可以为 actor 和 ref 分别添加以下参数。

.. code:: python

   # 对于 actor
   actor_rollout_ref.actor.megatron.param_offload=True \
   actor_rollout_ref.actor.megatron.grad_offload=True \
   actor_rollout_ref.actor.megatron.optimizer_offload=True \
   # 对于没有梯度和优化器的 ref
   actor_rollout_ref.ref.megatron.param_offload=True \


对于 critic，您可以包含这些参数。

.. code:: python

   # 对于 critic
   critic.megatron.param_offload=True \
   critic.megatron.grad_offload=True \
   critic.megatron.optimizer_offload=True \

Profiler
^^^^^^^^

```
分析器是一个帮助您了解模型性能的工具。它可以用于分析在不同操作上花费的时间，并识别瓶颈。您可以从 `torch.profiler <https://pytorch.org/docs/stable/profiler.html>`_ 获取更多信息。

在 verl 中，目前分析器仅支持 Megatron 中的演员角色。您可以设置开始步骤和结束步骤进行分析。请注意，一个步骤意味着一次梯度更新。分析结果将保存在 save_path 中。如果您只想在特定的排名中进行分析，可以设置 profile_ranks，默认情况下，它将是 [0]。

.. code:: python

   actor_rollout_ref.actor.profile.use_profile=True \
   actor_rollout_ref.actor.profile.profile_ranks=[0] \
   actor_rollout_ref.actor.profile.step_start=0 \
   actor_rollout_ref.actor.profile.step_end=1 \
   actor_rollout_ref.actor.profile.save_path="./profile"


相关 MCore 文档
----------------------

还有一个详细的文档，介绍如何使用 MCore 训练不同类型的模型，请参考 `MCore Document <https://github.com/volcengine/verl/blob/main/verl/models/mcore/readme.md>`_。
```