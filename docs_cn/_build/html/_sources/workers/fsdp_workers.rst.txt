PyTorch FSDP 后端
======================

最后更新：2025年2月12日。

我们通过实现各种工作者来支持 PyTorch FSDP 后端，包括演员（actor）、评论家（critic）、参考（reference）、回滚（rollout）和奖励（reward）模型。我们还实现了 ``FSDPVLLMShardingManager``，该管理器在 `fsdp_vllm.py <https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/fsdp_vllm.py>`_ 中在 FSDP 和 vLLM 之间重新分配权重。

**Pros**

- 轻松支持各种模型。

  - 用户只需实现相应的 ``dtensor_weight_loader`` 以实现 FSDP 和 vLLM 之间的权重同步。而对于 ``hf_weight_loader``，用户可以直接应用 HF 和 vLLM 中支持的任何模型，无需任何代码更改。

- 容易组织每个模型的前向和反向计算。

**Cons**

- 在处理大规模模型（例如 Llama 70B 和 405B）时，扩展性较差
- actor 和 rollout 之间的重新分片开销可能大于 Megatron-LM 后端。

由于其简单性，我们建议在算法研究和原型设计中使用 FSDP 后端。

FSDP Workers
--------------

ActorRolloutRefWorker
^^^^^^^^^^^^^^^^^^^^^

Actor/Rollout HybridEngine
''''''''''''''''''''''''''

1. HybridEngine、Actor 和 Rollout 初始化 API。

.. code:: python

   @register(dispatch_mode=Dispatch.ONE_TO_ALL)
   def init_model(self):

``ONE_TO_ALL``：当从驱动进程调用 ``init_model`` 函数时，每个工作节点（在 GPU 上）将执行以下模型初始化过程。

HybridEngine、Actor 和 Rollout 的初始化细节如下：

1. ``DataParallelPPOActor`` 实现了在使用 FSDP 构建模型时的简单 PPO 计算逻辑，包括计算对数概率、模型更新。
2. ``vLLMRollout`` 支持使用 vLLM 生成。我们修改了 vLLM 引擎，并使其在 SPMD 下执行，以适应我们的 ``WorkerGroup`` 设计。
3. ``FSDPVLLMShardingManager`` 是一个上下文管理器，用于在 actor 和 rollout 之间执行实际的重分片。

有关更多信息，请参见 `源代码 <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py>`_。

1. 生成序列并重新计算对数概率

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def generate_sequences(self, prompts: DataProto):

- ``Dispatch.DP_COMPUTE_PROTO``: 数据将沿着 DP 维度进行分发和收集

- 在此函数中，回滚模型将执行自回归生成，演员模型将重新计算生成响应的旧对数概率。

3. 更新演员模型

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def update_actor(self, data: DataProto):

- 使用 PPO 和熵损失更新演员模型权重。

参考模型  
''''''''''''''

1. 参考模型初始化

参考模型使用与演员模型相同的函数进行初始化，但不初始化 HybridEngine 和 Optimizer。然后，演员模型也被包装在 ``DataParallelPPOActor`` 中。

2. 计算参考对数概率

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def compute_ref_log_prob(self, data: DataProto):

- 在此函数中，参考模型将调用 ``DataParallelPPOActor`` 中的计算对数概率函数来计算参考对数概率。

CriticWorker 和 RewardWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 模型初始化

与参考模型非常相似。CriticWorker 将为 Optimizer 执行额外的初始化。

2. 计算 CriticWorker 的值

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def compute_values(self, data: DataProto):

3. 更新 Critic

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def update_critic(self, data: DataProto):

4. 计算奖励

.. code:: python

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def compute_rm_score(self, data: DataProto):

HybridShard
------------

混合分片

我们不支持 FSDP `HybridShard`。为了支持这一点，我们可能需要构建一个 2D 设备网格，并测试每个模型对应的 ``dtensor_weight_loader`` 和 ``hf_weight_loader``。