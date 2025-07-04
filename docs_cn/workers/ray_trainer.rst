PPO Ray Trainer
===============

最后更新：2025年2月12日。

我们实现了 `RayPPOTrainer`，这是一个在单个 CPU/GPU 节点（默认是 CPU）上的驱动进程中运行的训练器。

`PPORayTrainer` 包含 3 个核心功能，用于数据准备、WorkerGroup 初始化和 PPO 训练循环。

数据准备
----------------

`PPORayTrainer` 作为一个单进程，负责从数据集中加载一整批样本（提示），然后将其分发到在不同 GPU 上运行的不同 worker_groups。

为了实现数据加载的通用性，我们实现了 `RLHFDataset` 类，用于加载预处理的 parquet 文件，应用聊天模板到提示，添加填充，截断超过最大提示长度的提示，然后进行分词。

.. code:: python

   self.train_dataset = RLHFDataset(data_files=self.config.data.train_files,
                                       tokenizer=self.tokenizer,
                                       config=self.config.data)

然后，数据加载器将在 PPO 小批量大小下迭代数据集。

WorkerGroup 初始化
--------------------------

首先介绍在给定一组GPU上初始化"工作组(WorkerGroup)"的actor模型的基本实现。

```python
# max_colocate_count 表示每个 RayResourcePool 中 WorkerGroups(即进程)的数量
# 对于 FSDP 后端，我们建议使用 max_colocate_count=1，将所有 WorkerGroups 合并为一个。
# 对于 Megatron 后端，我们建议使用 max_colocate_count>1，可以为不同模型利用不同的 WorkerGroup
resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
                               use_gpu=True,
                               max_colocate_count=1)
# 定义要在远程初始化的 actor rollout 类
actor_rollout_cls = RayClassWithInitArgs(cls=ActorRolloutWorker)
# 定义 actor_rollout 工作组
actor_rollout_worker_group = MegatronRayWorkerGroup(resource_pool=resource_pool,
                                                   ray_cls_with_init=actor_rollout_cls,
                                                   default_megatron_kwargs=config.actor_rollout.megatron)
```

在上述实现中，像"actor_rollout_worker_group"、"critic_worker_group"和"ref_worker_group"这样的不同WorkerGroups位于单独的进程中。

然后，驱动进程可以调用"actor_rollout_worker_group"内的分布式计算函数以及其他角色，以构建RL训练循环。

对于位于相同一组GPU中的模型，我们进一步提供了一种细粒度优化，将不同角色的"worker_group"合并到同一进程中。这种优化可以节省不同进程中的冗余CUDA/分布式上下文。

```python```

# 初始化 WorkerGroup
   # 注意: 如果您想为每个角色使用不同的资源池，以支持不同的并行大小，
   # 您不应该使用 `create_colocated_worker_cls`。相反，直接将不同的资源池传递给不同的 worker groups。
   # 有关更多信息，请参见 TODO(url)。
   all_wg = {}
   for resource_pool, class_dict in self.resource_pool_to_cls.items():
       worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
       wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
       spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
       all_wg.update(spawn_wg)

   if self.use_critic:
       self.critic_wg = all_wg['critic']
       self.critic_wg.init_model()

   if self.use_reference_policy:
       self.ref_policy_wg = all_wg['ref']
       self.ref_policy_wg.init_model()

   if self.use_rm:
       self.rm_wg = all_wg['rm']
       self.rm_wg.init_model()

   # 在最后创建 rollout，以便 vllm 可以更好地估计 kv 缓存内存
   self.actor_rollout_wg = all_wg['actor_rollout']
   self.actor_rollout_wg.init_model()

```rst
.. 注意:: 对于 megatron(梅加特龙) 后端，如果我们将 ``worker_groups`` 合并到同一进程中，所有角色将利用相同的 3D 并行大小。为了优化这一点，我们可能需要在同一分布式上下文中为每个角色维护几个 3D 进程组。如果您想为不同的角色使用不同的 3D 并行大小，请按照第一个代码块的类似架构初始化每个角色的 ``worker_group``

PPO 训练循环
-----------------

我们通过调用每个角色的 worker_group 中的函数来实现 PPO 训练循环。每个函数的输入和输出数据是在 `protocol.py <https://github.com/volcengine/verl/blob/main/verl/protocol.py>`_ 中实现的 ``DataProto`` 对象。在训练循环中，训练器将根据包装在工作函数中的传输协议将数据分发/收集到不同的 GPU。PPO 微批处理的计算在 ``update_actor`` 和 ``update_critic`` 函数中进行。

要扩展到其他 RLHF 算法，如 DPO、GRPO，请参考 :doc:`../advance/dpo_extension`。

.. code:: python
```

```python
def fit(self):
       """
       PPO的训练循环。
       驱动进程只需要通过RPC调用worker组的计算函数来构建PPO数据流。
       轻量级优势计算在驱动进程上完成。
       """
       from verl.utils.tracking import Tracking
       from omegaconf import OmegaConf

       logger = Tracking(project_name=self.config.trainer.project_name,
                           experiment_name=self.config.trainer.experiment_name,
                           default_backend=self.config.trainer.logger,
                           config=OmegaConf.to_container(self.config, resolve=True))

       global_steps = 0

       # 在训练之前执行验证
       # 目前，我们只支持使用reward_function进行验证。
       if self.val_reward_fn is not None:
           val_metrics = self._validate()
           pprint(f'初始验证指标: {val_metrics}')

       for epoch in range(self.config.trainer.total_epochs):
           for batch_dict in self.train_dataloader:
               metrics = {}
```

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)
               # 将batch转移到cuda设备上

               # 为生成弹出这些键
               gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

               # 生成一个batch
               with Timer(name='gen', logger=None) as timer:
                   gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
               metrics['timing/gen'] = timer.last

               batch = batch.union(gen_batch_output)

               if self.use_reference_policy:
                   # 计算参考log_prob
                   with Timer(name='ref', logger=None) as timer:
                       ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                       batch = batch.union(ref_log_prob)
                   metrics['timing/ref'] = timer.last

               # 计算值
               with Timer(name='values', logger=None) as timer:
                   values = self.critic_wg.compute_values(batch)
                   batch = batch.union(values)
               metrics['timing/values'] = timer.last
```

```python
with Timer(name='adv', logger=None) as timer:
                   # 计算得分。支持模型和基于函数的两种方式。
                   # 我们首先使用奖励模型计算得分。然后，我们调用 reward_fn 将奖励模型和基于规则的结果结合起来。
                   if self.use_rm:
                       # 首先计算奖励模型得分
                       reward_tensor = self.rm_wg.compute_rm_score(batch)
                       batch = batch.union(reward_tensor)

                   # 与基于规则的奖励模型结合
                   reward_tensor = self.reward_fn(batch)
                   batch.batch['token_level_scores'] = reward_tensor

                   # 计算奖励。如果可用，应用 kl 惩罚
                   batch, kl_metrics = apply_kl_penalty(batch,
                                                           kl_ctrl=self.kl_ctrl_in_reward,
                                                           kl_penalty=self.config.algorithm.kl_penalty)
                   metrics.update(kl_metrics)
```

# 计算优势值，在驱动进程上执行
                   batch = compute_advantage(batch,
                                               self.config.algorithm.gamma,
                                               self.config.algorithm.lam,
                                               adv_estimator=self.config.algorithm.adv_estimator)
               metrics['timing/adv'] = timer.last

               # 更新评论家网络
               if self.use_critic:
                   with Timer(name='update_critic', logger=None) as timer:
                       critic_output = self.critic_wg.update_critic(batch)
                   metrics['timing/update_critic'] = timer.last
                   critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                   metrics.update(critic_output_metrics)

# 实现评论家(critic)预热
               if self.config.trainer.critic_warmup <= global_steps:
                   # 更新演员(actor)
                   with Timer(name='update_actor', logger=None) as timer:
                       actor_output = self.actor_rollout_wg.update_actor(batch)
                   metrics['timing/update_actor'] = timer.last
                   actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                   metrics.update(actor_output_metrics)

               # 验证
               if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                   with Timer(name='testing', logger=None) as timer:
                       val_metrics: dict = self._validate()
                       val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                   metrics['timing/testing'] = timer.last
                   metrics.update(val_metrics)

               # 收集指标
               data_metrics = compute_data_metrics(batch=batch)
               metrics.update(data_metrics)

```python
# 待办事项: 创建一个支持各种后端的规范日志记录器
               logger.log(data=metrics, step=global_steps)

               if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                   actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                   f'global_step_{global_steps}')
                   actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'actor')
                   self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                   if self.use_critic:
                       critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                           f'global_step_{global_steps}')
                       critic_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'critic')
                       self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

               global_steps += 1
```

# 训练后执行验证
       if self.val_reward_fn is not None:
           val_metrics = self._validate()
           pprint(f'最终验证指标: {val_metrics}')