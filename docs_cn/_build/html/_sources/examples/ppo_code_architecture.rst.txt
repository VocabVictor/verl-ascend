PPO示例架构
========================

最近更新日期：02/17/2025。

让我们从Proximal Policy Optimization算法开始，这是LLM后训练中最广泛使用的算法。

PPO算法示例的主要入口点是：
`main_ppo.py <https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py>`_。
在本教程中，我们将深入介绍`main_ppo.py <https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py>`_中的代码架构。

定义数据
---------------

用户需要对数据集进行预处理并存储在parquet文件中。
我们实现了`RLHFDataset`来加载和标记parquet文件。

对于``RLHFDataset``（默认情况下），至少需要1个字段：

- ``prompt``：包含字符串提示

我们已经提供了一些示例，用于将数据集处理成parquet文件，位于`data_preprocess目录 <https://github.com/volcengine/verl/blob/main/examples/data_preprocess>`_。目前，我们支持对GSM8k、MATH、Hellasage、Full_hh_rlhf数据集进行预处理。有关更多信息，请参见:doc:`../preparation/prepare_data`。

为不同数据集定义奖励函数
--------------------------------------------------

在这个主入口点中，用户只需要基于在PPO训练中使用的数据集(或应用程序)定义自己的奖励函数。

例如，我们已经为`GSM8k <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py>`和`MATH <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py>`数据集在``_select_rm_score_fn``中提供了奖励函数。在``RewardManager``中，我们将根据数据源计算奖励分数以选择相应的奖励函数。对于一些RLHF数据集(例如full_hh_rlhf)，奖励模型用于评估响应而无需任何奖励函数。在这种情况下，``RewardManager``将直接返回奖励模型计算的``rm_score``。

查看`奖励函数 <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score>`以获取详细实现。

定义工作类
---------------------

.. code:: python

```python
if config.actor_rollout_ref.actor.strategy == 'fsdp': # 适用于FSDP后端
       assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
       from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
       from verl.single_controller.ray import RayWorkerGroup
       ray_worker_group_cls = RayWorkerGroup

   elif config.actor_rollout_ref.actor.strategy == 'megatron': # 适用于Megatron后端
       assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
       from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
       from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
       ray_worker_group_cls = NVMegatronRayWorkerGroup # 用于Megatron-LM的Ray工作类

   else:
       raise NotImplementedError

   from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

   role_worker_mapping = {
       Role.ActorRollout: ActorRolloutRefWorker,
       Role.Critic: CriticWorker,
       Role.RefPolicy: ActorRolloutRefWorker
   }
```

```python
global_pool_id = '全局资源池'
   resource_pool_spec = {
       global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
   }
   mapping = {
       Role.ActorRollout: global_pool_id,
       Role.Critic: global_pool_id,
       Role.RefPolicy: global_pool_id,
   }

步骤 1: 构建角色和工作者之间的映射
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一个角色代表了同一进程中的一组工作者。我们在 `ray_trainer.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L38>`_ 中预定义了几个角色。

.. code:: python

   class Role(Enum):
       """
       要动态创建更多角色，您可以对 Role 进行子类化并添加新成员
       """
       Actor = 0  # 该工作者仅包含 Actor
       Rollout = 1 # 该工作者仅包含 Rollout
       ActorRollout = 2 # 该工作者同时包含 Actor 和 Rollout，是一个 HybridEngine
       Critic = 3 # 该工作者仅包含 Critic
       RefPolicy = 4 # 该工作者仅包含参考策略
       RewardModel = 5 # 该工作者仅包含奖励模型
       ActorRolloutRef = 6 # 该工作者同时包含 Actor、Rollout 和参考策略
```

第二步：定义与此角色对应的工作器类
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 我们预先实现了``ActorRolloutRefWorker``。通过不同的配置，它可以是独立的actor，独立的rollout，ActorRollout混合引擎，或ActorRolloutRef混合引擎。
- 我们还预先实现了针对两种不同后端（PyTorch FSDP和Megatron-LM）的``Actor``、``Rollout``、``Critic``、``Reward Model``和``Reference model``的工作器。
  有关更多信息，请参阅`FSDP Workers <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py>`_ 和 `Megatron-LM Workers <https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py>`_。

第三步：定义资源池 ID 和资源池规格
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 资源池是全局 GPU 资源的一个划分，``resource_pool_spec`` 是一个字典，将 ID 映射到 GPU 数量。

在上面的示例中，我们定义了一个全局资源池：global_pool_id，然后将所有角色放置在这个资源池中，所有GPU都用于这个后训练任务。这是指*共同定位*放置，其中所有模型共享相同的一组GPU。

- 有关高级用法，请参阅资源池和放置。

定义奖励模型/函数
------------------------------

```python
# 在这里我们应该采用多源奖励函数
# - 对于基于规则的奖励模型，我们直接调用奖励分数
# - 对于基于模型的奖励模型，我们调用一个模型
# - 对于与代码相关的提示，如果有测试用例，我们将其发送到沙盒
# - 最后，我们将所有奖励组合在一起
# - 奖励类型取决于数据的标签
if config.reward_model.enable:
    from verl.workers.fsdp_workers import RewardModelWorker
    role_worker_mapping[角色.奖励模型] = RewardModelWorker
    mapping[角色.奖励模型] = global_pool_id

reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

# 请注意，我们始终对验证使用基于函数的奖励模型
val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
```

```python
resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

由于并非所有任务都使用基于模型的资源管理(Resource Management)，用户需要在这里定义是基于模型的RM还是基于函数的RM。

- 如果是基于模型的RM，在资源映射中直接添加``RewardModel``角色，并将其添加到资源池映射中。

  - 请注意，预定义的``RewardModelWorker``仅支持具有huggingface结构的模型``AutoModelForSequenceClassification``。如果不是这个模型，您需要在`FSDP Workers <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py>`_ 和 `Megatron-LM Workers <https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py>`_ 中定义自己的RewardModelWorker。

- 如果是基于函数的RM，则需要为每个数据集分类别用户的奖励函数。

```python
def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        raise NotImplementedError
```

查看在`目录 <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/>`_ 中实现的奖励函数，以获取更多信息。

定义、初始化和运行 PPO 训练器
------------------------------------

.. code:: python

   trainer = RayPPOTrainer(config=config,
                           tokenizer=tokenizer,
                           role_worker_mapping=role_worker_mapping,
                           resource_pool_manager=resource_pool_manager,
                           ray_worker_group_cls=ray_worker_group_cls,
                           reward_fn=reward_fn,
                           val_reward_fn=val_reward_fn)
   trainer.init_workers()
   trainer.fit()

- 首先使用用户配置、分词器以及上述所有的工作映射、资源池、工作组和奖励函数来初始化``RayPPOTrainer``
- 首先调用``trainer.init_workers()``来在分配的 GPU 上初始化模型（在资源池中）
- 实际的 PPO 训练将在``trainer.fit()``中执行

verl可以通过重用Ray模型工作者、资源池和奖励函数，轻松扩展到其他强化学习算法。有关更多信息，请参阅:doc:`扩展(../advance/dpo_extension)`。

``RayPPOTrainer``的详细信息在:doc:`Ray训练器(../workers/ray_trainer)`中讨论。