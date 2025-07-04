=========================================================
混合流( HybridFlow) 编程指南
=========================================================

最近更新日期：06/02/2025。

.. _vermouth: https://github.com/vermouth1992

作者：`张驰 <https://github.com/vermouth1992>`_

verl 是论文 `HybridFlow <https://arxiv.org/abs/2409.19256v2>`_ [1]_ 的开源实现。在本节中，我们将介绍 HybridFlow 的基本概念、动机以及如何使用 verl API 进行编程。

动机和设计
------------------------
我们使用数据流来表示强化学习系统。[4]_.

数据流程
~~~~~~~~~~~~~~~~~~~~

数据流(Dataflow)是计算的一种抽象形式。神经网络训练是一个典型的数据流。它可以用计算图(computational graph)来表示。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/dataflow.jpeg?raw=true
   :alt: The dataflow graph from CS231n 2024 lecture 4

这个图 [2]_ 表示了一个多项式函数后跟一个Sigmoid函数的计算图。在神经网络计算的数据流中，每个节点代表一个运算符，每条边代表前向/后向传播的方向。计算图决定了神经网络的架构。

作为数据流问题的强化学习
++++++++++++++++++++++++++++++++++++++++++++++

强化学习（RL）训练也可以表示为数据流。下面是代表RLHF 中使用的PPO算法的数据流图 [3]_:

.. image:: https://picx.zhimg.com/70/v2-cb8ab5ee946a105aab6a563e92682ffa_1440w.avis?source=172ae18b&biz_tag=Post
  :alt: PPO dataflow graph, credit to Zhihu 低级炼丹师

然而，强化学习的数据流与神经网络训练的数据流有以下基本区别：

+--------------------------+--------------------------------------------------+---------------------+
| 工作负载                 | 节点                                             | 边缘                |
+--------------------------+--------------------------------------------------+---------------------+
| 神经网络训练             | 运算符（+/-/matmul/softmax）                     | 张量移动            |
+--------------------------+--------------------------------------------------+---------------------+
| 强化学习                | 高级运算符（rollout/model forward）              | 数据移动            |
+--------------------------+--------------------------------------------------+---------------------+

在表格式强化学习中，每个运算符是一个简单的标量数学运算（例如，贝尔曼更新）。在深度强化学习（DRL）中，每个运算符是一个高级神经网络计算，如模型推断/更新。这使得强化学习成为一个两级数据流问题：

- 控制流(Control flow): 定义了高级操作符的执行顺序（例如，在PPO中，我们首先执行rollout，然后进行优势计算，最后进行训练）。它表达了**强化学习算法的核心逻辑**。
- 计算流(Computation flow): 定义了**神经网络计算**的数据流程（例如，模型的前向传播/反向传播/优化器）。

设计选择
~~~~~~~~~~~~~~~~~~~~
在LLM时代之前的深度强化学习(DRL)中，模型规模通常较小。因此，高级神经网络计算可以在单个进程中完成。这使得将计算流嵌入控制流作为单个进程成为可能。

然而，在LLM时代，计算流程（例如，训练神经网络）变成了一个多进程程序。这自然地引出了两种设计选择：

1. 将控制流也转换为多进程程序。然后与计算流程合并（统一的多控制器）。

- 优势:

- 在固定的计算流和控制流下实现**最佳性能**，因为在训练和数据传输中通信开销被最小化。

- 缺点:

- 从软件的角度来看，计算和/或控制流**难以重用**，因为计算代码与特定的控制器代码耦合在一起。例如，PPO的训练循环是通用的。假设我们有一个使用特定计算流（如FSDP）实现的PPO训练流。如果我们想将计算流从FSDP切换到Megatron，由于控制流和计算流的耦合，控制流和计算流都无法重用。
  - 在灵活和动态的控制流下，用户需要付出更多的努力，因为程序的多进程特性。

2. 分离流：控制流使用单进程，计算流使用多进程

- 优势:

- 在解耦之后，其他地方定义的计算流程可以**轻松重用**。
  - 控制器在单个进程上运行。实现一个具有**不同控制流程的新强化学习算法是简单且容易的**。

- 缺点:

- 每次控制器进程与计算进程交互时，都会增加额外的**数据通信开销**。数据必须来回传输。

在verl中，采用了控制流与计算流分离的策略。verl旨在解耦强化学习(RL)算法的控制流与计算引擎的实现。

总体执行图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面是一个简化的图示，表示强化学习作业的执行。在该图中，控制器在单个进程上运行，而生成器/演员工作进程和评论家工作进程则在多个进程上运行，并被放置在特定的资源组中。对于回滚，控制器将数据传递给生成器以执行样本生成。当回滚完成后，数据被传回控制器以进行算法的下一步。其他工作进程也以类似的方式执行。通过混合控制器设计，数据流与计算被解耦，从而在计算效率和定义算法训练循环的灵活性之间提供了平衡。

.. figure:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/driver_worker.png?raw=true
   :alt: 执行图

代码库概述 (PPO)
------------------------------------------------

入口函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
代码: https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py

在这个文件中，我们定义了一个远程函数 `main_task`，作为控制器（驱动）进程，如上图所示。我们还定义了一个 ``RewardManager``，用户可以根据数据集中的数据源自定义他们的奖励函数。请注意，`RewardManager` 应返回由强化学习（RL）算法优化的最终令牌级奖励。用户可以结合基于模型的奖励和基于规则的奖励。
``main_task`` 构造一个 RayPPOTrainer 实例并启动训练。请注意，``main_task`` **作为单个进程运行**。

我们强烈建议 ``main_task`` 不要在 ray 集群的头节点上调度，因为 ``main_task`` 会消耗大量内存，而头节点通常资源非常有限。

Ray训练器
~~~~~~~~~~~~~~~~~~~~
代码: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py

RayPPOTrainer 管理

- Worker 和 WorkerGroup 的构建
- 运行 PPO 算法的主循环

请注意，RayPPOTrainer 的 fit 函数 **作为单个进程运行**。

Worker 和 WorkerGroup 的构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个 WorkerGroup 管理一组远程运行的工作者（workers）。请注意，工作组在其构造函数的进程中运行。WorkerGroup 内的每个工作者都在 GPU 上运行。工作组作为控制进程与工作者列表进行交互的代理，以执行某些计算。**为了做到这一点，我们必须将工作者的方法绑定到 WorkerGroup 的方法中，并定义数据分发和数据收集**。这通过在工作者定义部分将介绍的简单装饰器来完成。

例如，在 PPO 中，我们定义了 3 个工作组：

- ActorRolloutRef(演员-回合-参考策略)：管理演员、回合和参考策略。ActorRolloutRefWorker(演员-回合-参考策略工作者)可以被实例化为单个演员、单个回合、单个参考策略、组合的演员/回合或组合的演员/回合/参考策略。这种设计旨在在各种场景中最大限度地重用代码。将演员和回合放在一起的原因是为了使用nccl进行快速权重传输。将演员和参考策略放在一起的原因是为了实现高效的lora PPO，因为参考策略简单地是lora中PPO的基本模型。
- Critic(评论家)：管理评论家模型
- Reward(奖励)：管理奖励模型

工作者组将在其指定的资源池上构建。资源池是ray集群中一组GPU。

工作者定义
~~~~~~~~~~~~~~~~~~~~

.. _ActorRolloutRefWorker: https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py

我们以`ActorRolloutRefWorker <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py>`_ 为例。
它应该向控制器进程公开的API包括：

- init_model: 构建基础模型
- generate_sequences: 给定提示，生成响应
- compute_log_prob: 使用演员计算生成序列的对数概率
- compute_ref_log_prob: 使用参考策略计算生成序列的对数概率
- save_checkpoint: 保存检查点

请注意，这些方法是在工作进程中定义的，只能通过远程调用来调用。例如，如果控制器进程想要初始化模型，它必须调用

.. code-block:: python

```python
for worker in actor_rollout_ref_wg:
       worker.init_model.remote()
```

如果控制器进程想要生成序列，它必须调用

.. code-block:: python

```python
data = xxx
   # 将数据拆分为 dp 块
   data_dp_lst = data.split(dp_size)
   output_dp_lst = []
   for i, worker in enumerate(actor_rollout_ref_wg):
       output_future = worker.generate_sequences.remote(data_dp_lst[i])
       output_dp_lst.append(output_future)
   output = torch.cat(ray.get(output_dp_lst), dim=0)

我们观察到，控制器进程调用工作组方法通常可以分为 3 个部分：

- 将数据拆分为数据并行大小
- 将相应的数据分发到每个工作者
- 在计算完成时收集并连接数据

在 verl 中，我们设计了一种语法糖，将这 3 个过程封装为控制器进程中的单个调用。
```

.. code-block:: python

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(data):
    ...

# 在驱动程序上
output = actor_rollout_ref_wg.generate_sequences(data)

我们用一个 ``register`` 装饰器装饰工作者的方法，该装饰器明确地定义了输入数据应该如何被拆分并分发到每个工作者，以及输出数据应该如何被控制器收集和连接。例如，``Dispatch.DP_COMPUTE_PROTO`` 将输入数据拆分为 dp 块，将每个数据分发到每个工作者，收集输出并连接结果。请注意，此函数要求输入和输出为此处定义的 DataProto (https://github.com/volcengine/verl/blob/main/verl/protocol.py)。

PPO 主循环
~~~~~~~~~~~~~~~~~~~~
通过上述 API，我们可以将 PPO 的主循环实现为一个单进程程序。
```

.. code-block:: python

```python
for prompt in dataloader:
       output = actor_rollout_ref_wg.generate_sequences(prompt)
       old_log_prob = actor_rollout_ref_wg.compute_log_prob(output)
       ref_log_prob = actor_rollout_ref_wg.compute_ref_log_prob(output)
       values = critic_wg.compute_values(output)
       rewards = reward_wg.compute_scores(output)
       # compute_advantages 直接在控制过程中运行
       advantages = compute_advantages(values, rewards)
       output = output.union(old_log_prob)
       output = output.union(ref_log_prob)
       output = output.union(values)
       output = output.union(rewards)
       output = output.union(advantages)
       # 更新演员
       actor_rollout_ref_wg.update_actor(output)
       critic.update_critic(output)

收获
~~~~~~~~~~~~~~~~~~~~
- 这种编程范式使用户能够在不修改控制过程的情况下使用不同的计算后端。
- 这种编程范式通过改变 WorkerGroup 和 ResourcePool 的映射，实现灵活的放置，而无需修改控制过程。

代码库组织
------------------------------------------------
```

重要的代码文件在仓库中的组织结构如下：

.. code-block:: bash

```
verl # verl 包
     trainer
       main_ppo.py  # 强化学习训练的入口点
       ppo
         ray_trainer.py  # 强化学习算法（如 PPO）的训练循环
       fsdp_sft_trainer.py  # 使用 FSDP 后端的 SFT 训练器
     config
       generation.yaml  # rollout 的配置模板
       ppo_trainer.yaml  # RL 训练器的配置模板
     workers
       protocol.py  # DataProto 的接口
       fsdp_workers.py   # FSDP 工作接口：ActorRolloutRefWorker, CriticWorker, RewardModelWorker
       megatron_workers.py  # Megatron 工作接口：ActorRolloutRefWorker, CriticWorker, RewardModelWorker
       actor
         dp_actor.py  # 使用 FSDP 后端的数据并行演员
         megatron_actor.py  # 使用 Megatron 后端的 nD 并行演员
       critic
         dp_critic.py  # 使用 FSDP 后端的数据并行评论员
         megatron_critic.py  # 使用 FSDP 后端的 nD 并行评论员
       reward_model
         megatron
           reward_model.py  # 使用 Megatron 后端的奖励模型
       rollout
         vllm
           vllm_rollout.py  # 使用 vllm 后端的 rollout
         hf_rollout.py  # 使用 huggingface TGI 后端的 rollout
       sharding_manager
         fsdp_ulysses.py  # 使用 FSDP + ulysses 时的数据和模型重分片
         fsdp_vllm.py  # 使用 FSDP + ulysses + vllm 时的数据和模型重分片
         megatron_vllm.py  # 使用 Megatron + vllm 时的数据和模型重分片
     utils
       dataset  # SFT/RM/RL 的数据集
       reward_score  # 基于函数的奖励
         gsm8k.py  # gsm8k 数据集的奖励函数
         math.py  # 数学数据集的奖励函数
       seqlen_balancing.py  # 序列平衡优化
     models
       llama  # Megatron 对 llama、deepseek、mistral 等的实现
       transformers  # 与 transformer 模型（如 llama、qwen 等）的 ulysses 集成
       weight_loader_registery.py  # 用于将 hf ckpt 加载到 Megatron 的权重加载器注册表
     third_party
       vllm  # vllm 在 RL 中使用的适配器
         vllm_spmd  # vllm >= v0.7 的适配器
   examples  # 示例脚本
   tests  # 集成和单元测试
   .github  # 持续集成测试的配置
```

.. [1] HybridFlow: 一个灵活高效的RLHF框架: https://arxiv.org/abs/2409.19256v2  
.. [2] 数据流图感谢CS231n 2024年第4讲: https://cs231n.stanford.edu/slides/2024/lecture_4.pdf  
.. [3] PPO数据流图感谢来自知乎的低级炼丹师: https://zhuanlan.zhihu.com/p/635757674  
.. [4] RLFlow