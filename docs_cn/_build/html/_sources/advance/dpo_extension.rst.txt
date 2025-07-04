扩展到其他强化学习(HF)算法
=================================

最近更新日期：2025年2月25日。

我们已经实现了完整的PPO算法训练流程。为了扩展到其他算法，我们分析了使用verl的高级原则，并提供了一个实现DPO算法的教程。用户可以按照类似的范式来扩展到其他强化学习算法。

.. 注意:: **关键思想**：单进程驱动多进程计算和数据通信。

整体方法
----------------

步骤1：考虑每个模型需要的多机多GPU计算，比如在actor_rollout模型中的``generate_sequence``、``compute_log_prob``和``update_policy``。实现分布式单进程多数据（SPMD）计算，并将其封装成API。

步骤2：根据不同的分布式场景，包括Megatron-LM中的FSDP和3D并行性，实现对多进程计算之间数据交互的单进程控制。

步骤3：利用封装的API实现控制流程

示例：在线DPO
-------------------

我们使用verl来实现一个简单的在线DPO(确定性策略优化)算法。在线DPO的算法流程如下：

1. 存在一个提示(rollout)生成器，其权重与actor模型相同。在将一批提示输入生成器后，它为每个提示生成N个响应。
2. 将所有提示+响应发送给一个验证器进行评分，验证器可以是奖励模型或基于规则的函数。然后将它们成对排序以形成一个训练批次。
3. 使用这个训练批次来训练使用DPO的actor模型。在这个过程中，需要一个参考策略。

第1步：什么是多机多GPU计算
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**样本生成器**

实现细节:

```python
   from verl.single_controller.base import Worker
   from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs, RayResourcePool
   import ray

   @ray.remote
   class 样本生成器(SampleGenerator, Worker):
       def __init__(self, config):
           super().__init__()
           self.config = config
           
       def 生成序列(self, data):
           pass

在这里，``样本生成器(SampleGenerator)`` 可以被视为由``torchrun``拉起的多进程，每个进程运行相同的代码（SPMD）。
``样本生成器(SampleGenerator)`` 需要实现一个 ``生成序列(Generate Sequences)`` API，供控制流调用。内部的实现细节可以使用任何推理引擎，包括 vllm、sglang 和 huggingface。用户可以在 verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py 中大量重用代码，这里我们不会详细展开。
```

**引用策略推断**

API: 计算参考对数概率

.. code:: python

   from verl.single_controller.base import Worker
   import ray

   @ray.remote
   class 参考策略(Worker):
       def __init__(self):
           super().__init__()
           self.model = Model()
           
       def 推断(self, 数据):
           return self.model(数据)

**Actor update(更新参与者)**

API: 更新actor模型参数

.. code:: python

   from verl.single_controller.base import Worker
   import ray

   @ray.remote
   class DPOActor(Worker):
       def __init__(self):
           super().__init__()
           self.model = Model()
           self.model = FSDP(self.model)  # 或其他分布式策略
           self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
           self.loss_fn = xxx
           
       def update(self, data):
           self.optimizer.zero_grad()
           logits = self.model(data)
           loss = self.loss_fn(logits)
           loss.backward()
           self.optimizer.step()

**注意: 如何区分控制过程和分布式计算过程**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 控制过程通常是直接使用``@ray.remote``装饰的函数
- 计算过程都被封装到一个``RayWorkerGroup``中。

用户可以重用PPO算法中实现的大部分分布式计算逻辑，包括verl/verl/trainer/ppo中的FSDP和Megatron-LM后端。

第二步: 基于不同的分布式场景，实现单进程控制多进程数据交互
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**这里需要解决的核心问题是单个进程如何向多个进程发送数据，驱动多进程计算，以及控制进程如何获取多进程计算的结果。**
首先，在控制进程中初始化多进程的``WorkerGroup``。

.. code:: python

   @ray.remote(num_cpus=1)
   def main_task(config):
       # 构建SampleGenerator
       resource_pool = RayResourcePool(process_on_nodes=[8] * 2)  # 16 GPUs
       ray_cls = RayClassWithInitArgs(SampleGenerator, config=config)
       # 将SampleGenerator放入资源池
       worker_group = RayWorkerGroup(resource_pool, ray_cls)
       
       # 构建参考策略

正如我们所看到的，在控制过程中，多个进程被封装到一个``RayWorkerGroup``中。在这个``WorkerGroup``内部，有一个``self._workers``成员，其中每个worker都是一个RayActor（https://docs.ray.io/en/latest/ray-core/actors.html）的SampleGenerator。ray_trainer.md还提供了``MegatronRayWorkerGroup``的实现。

假设模型使用FSDP进行分布式处理，并且在控制过程中有一批数据，为了数据并行处理，底层的调用过程如下：

```python
data = xxx
data_list = data.chunk(dp_size)

output = []
for d in data_list:
    # worker_group._workers[i] 是一个SampleGenerator
    output.append(worker_group._workers[i].generate_sequences.remote(d))

output = ray.get(output)
output = torch.cat(output)
```

单个进程调用多个进程涉及以下3个步骤：

1. 在控制过程中将数据分割成DP部分。
2. 将数据发送到远程，通过RPC调用远程计算，并利用多进程计算。
3. 在控制过程中获取每个worker的计算结果并合并它们。

频繁调用控制器进程上的这3个步骤会严重影响代码的可读性。**在verl中，我们已经将这3个步骤抽象并封装起来，使得worker的方法 + 分发 + 收集可以注册到worker_group中**

```python
from verl.single_controller.base.decorator import register

def dispatch_data(worker_group, data):
    return data.chunk(worker_group.world_size)
    
def collect_data(worker_group, data):
    return torch.cat(data)

dispatch_mode = {
    'dispatch_fn': dispatch_data,
    'collect_fn': collect_data
}

@register(dispatch_mode=dispatch_mode)
def generate_sequences(self, data):
    pass
```

这样一来，我们可以通过在控制（驱动）进程上直接通过``worker_group``调用worker内的方法：

```python
output = worker_group.generate_sequences(data)
```

这一行代码包含了数据分割、数据分发和计算，以及数据收集。

此外，每个模型的模型并行大小通常是固定的，包括数据并行(dp)、模型并行(tp)和管道并行(pp)。因此，针对这些常见的分布式场景，我们在 `decorator.py <https://github.com/volcengine/verl/blob/main/verl/single_controller/base/decorator.py>`_ 中预先实现了特定的调度和收集方法，可以直接用于封装计算。

.. code:: python

   from verl.single_controller.base.decorator import register, Dispatch

   @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
   def generate_sequences(self, data: DataProto) -> DataProto:
       pass

这里需要数据接口为``DataProto``。``DataProto``的定义在 `protocol.py <https://github.com/volcengine/verl/blob/main/verl/protocol.py>`_ 中。

第三步：主训练循环
~~~~~~~~~~~~~~~~~~~~

通过上述训练流程，我们可以实现算法的控制流程。建议``main_task``也是一个 ray 远程进程。

.. code:: python

```python
@ray.remote(num_cpus=1)
def main_task(config):
    # 构建SampleGenerator
    resource_pool = RayResourcePool(process_on_nodes=[8] * 2)  # 16 GPUs
    ray_cls = RayClassWithInitArgs(SampleGenerator, config=config) 
    # 将SampleGenerator放入资源池
    sample_gen = RayWorkerGroup(resource_pool, ray_cls)
    
    # 构建参考策略
    ray_cls = RayClassWithInitArgs(ReferencePolicy)
    ref_policy = RayWorkerGroup(resource_pool, ray_cls)
    
    # 构建actor
    ray_cls = RayClassWithInitArgs(DPOActor)  
    dpo_policy = RayWorkerGroup(resource_pool, ray_cls)
    
    dataloader = DataLoader()
    
    for data in dataloader:
        # 生成数据
        data = sample_gen.generate_sequences(data)
        # 为每个数据生成分数
        data = generate_scores(data)
        # 使用分数生成成对数据
        data = generate_pairwise_data(data)
        # 生成ref_log_prob
        data.batch['ref_log_prob'] = ref_policy.infer(data)
        # 使用dpo进行更新
        dpo_policy.update(data)
        # 记录日志
```

在这里，可以使用 `create_colocated_worker_cls` 将不同的"工作组(WorkerGroups)"放置在同一个资源池中或不同的资源池中，类似于 `ray_trainer.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py>`_。