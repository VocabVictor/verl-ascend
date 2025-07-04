SGLang 后端
==============

最后更新日期：05/31/2025。

**由 SGLang RL 团队撰写，并按姓氏字母顺序列出**

`Jingyi Chen (陈靖仪) <https://github.com/fzyzcjy>`_, `Yitong Guan (关一桐) <https://github.com/minleminzui>`_, `Zhuobin Huang (黄卓斌) <https://zobinhuang.github.io/sec_about/>`_, `Jiajun Li (李佳俊) <https://github.com/guapisolo>`_, `Ji Li (李吉) <https://github.com/GeLee-Q>`_, `Shenggui Li (李胜贵) <https://franklee.xyz/about>`_, `Junrong Lin (林俊荣) <https://github.com/ocss884>`_, `Xiang Long (龙翔) <https://github.com/SwordFaith>`_, `Rui Lu (卢锐) <https://scholar.google.com/citations?user=-MGuqDcAAAAJ>`_, `Jin Pan (潘劲) <https://jhinpan.github.io/>`_, `Shuai Shi (石帅) <https://github.com/shuaills>`_, `Yushen Su (苏雨申) <https://yushengsu-thu.github.io/>`_, `Xinyuan Tong (童欣源) <https://github.com/JustinTong0323>`_, `Chendong Wang (王晨东) <https://github.com/cedricbeta>`_, `Hanchen Zhang (张瀚辰) <https://scholar.google.com/citations?user=pGcJcagAAAAJ>`_, `Haoran Wang (王浩然) <https://ubecc.github.io/about/>`_, `Yongan Xiang (向永安) <https://github.com/BearBiscuit05>`_, `Chengxing Xie (谢成兴) <https://yitianlian.github.io/>`_, `Yuhao Yang (杨宇浩) <https://github.com/yhyang201>`_, `Jinwei Yao (姚金伟) <https://kivi-yao.github.io/>`_, `Qiaolin Yu (于乔林) <https://github.com/Qiaolin-Yu>`_, `Yuzhen Zhou (周雨臻) <https://github.com/zyzshishui>`_, `Chenyang Zhao (赵晨阳) <https://github.com/zhaochenyang20>`_

# 介绍
[SGLang](https://github.com/sgl-project/sglang)是一款开源的最先进的推理服务引擎，由xAI全面采用，以支持Grok在研究和服务过程中的所有推理需求。

目前，verl完全支持在推出阶段使用SGLang作为推理引擎。作为一个推出引擎，SGLang提供了与vLLM相同的功能覆盖，包括内存节省和多节点推出功能。安装verl和SGLang后，只需在启动脚本中简单地添加``actor_rollout_ref.rollout.name=sglang``，即可在两个推理框架之间无缝切换。

此外，SGLang团队正在积极开发支持诸如多轮主体RL、VLM RLHF、基于服务器的RLHF和部分推出等功能。您可以在[跟踪路线图](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/74)中跟踪相关的开发进展。

# 安装
请始终按照以下命令安装带有verl的SGLang。

.. code-block:: bash
    
    pip install --upgrade pip
    # Currently 0.4.6.post5, subject to updates at any time, please refer to the latest version specified in `setup.py`
    pip install -e ".[sglang]"

您可以检查您的环境中是否存在以下依赖项：

.. 注意::

- **PyTorch**: 2.6.0+cu124
    - **CUDA**: 12.4
    - **flashinfer-python**: 0.2.5+cu124torch2.6
    - **sgLang**: 0.4.6.post5
    - **sgl-kernel**: 0.1.4

使用 SGLang 作为单机 PPO 训练的推理后端
-------------------------------------------------------------------------
我们在 gsm8k 数据集上使用 Qwen/Qwen2-7B-Instruct 进行简单测试。

1. 运行以下命令以准备 gsm8k 数据集：

.. code-block:: bash

```python
python3 examples/data_preprocess/gsm8k.py
```

2. 运行以下脚本在一台配备4个GPU的单机上进行PPO (Proximal Policy Optimization) 实验：

.. code-block:: bash

```bash
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=4096 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2-7B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console'] \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

为什么要导出SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``verl`` 在推出期间初始化了一个``SGLangRollout``模块，用于评估/生成样本。

2. ``SGLangRollout`` 将初始化``Engine``，进一步初始化一个用于支持张量并行（Tensor Parallel，TP）的``torch.distributed.DeviceMesh``。

3. ``DeviceMesh.init()`` 内部检查所有参与设备的空闲GPU内存。如果差异太大（超过约10%），它会直接报告错误，以避免初始化失败或死锁。

为什么会存在不一致的GPU内存？
"""""""""""""""""""""""""""""""""""""""""""

**1. Ray分布式Actor在不同时间加载模型**

``verl`` 使用基于Ray的多进程、多GPU并发训练。每个``WorkerDict``可能在不同时间被调用：

.. code-block:: python

```rst
self.rollout = SGLangRollout(...)

不同的Worker在不同时间初始化模型 → 导致内存使用不同。

**2. 延迟初始化导致内存偏差**

一些Worker在其他Worker之前开始模型加载/推理（例如，``generate_sequences()``，``compute_log_prob()``）。
早期的Worker已经使用了GPU内存 → 后续的Worker仍有空闲内存 → 出现内存差异。

**3. SGLang的TP初始化使用"全设备广播"，但没有统一的释放时间**

尽管``SGLangRollout``可能仅涉及GPU的子集，但其``Engine``初始化调用``torch.distributed.init_process_group()``并广播权重，因此：

- 非Rollout GPU也加入通信。
- 后续，由于"内存不一致"，``DeviceMesh``初始化将失败。

**4. 不同的FSDP/TP加载行为也导致不匹配**
```

如果使用：

.. code-block:: bash

```plaintext
    actor.fsdp_config.param_offload=True  
    ref.fsdp_config.param_offload=True
```

部分工作进程将参数保留在 CPU 上，而另一些已经分片到 GPU 上 → 导致不对称的内存布局。

使用SGLang作为PPO训练跨多台机器的推理后端
--------------------------------------------------
SGLang还支持在IPv4和IPv6场景下运行verl的基于RAY的跨机器推理。在下面的脚本中，我们使用TP=16进行跨机器推理。假设我们有两台互连的机器：node0 的 IP 是 10.94.16.4，node1 的 IP 是 10.94.16.5。

1. 在 node0 上启动 Ray：

.. code-block:: bash

```plaintext
ray start --head --dashboard-host=0.0.0.0

您将看到以下提示：
```

.. code-block:: bash

启用了使用统计信息收集。要禁用此功能，请在启动集群的命令中添加`--disable-usage-stats`，或在启动集群之前运行以下命令：`ray disable-usage-stats`。有关更多详细信息，请参阅https://docs.ray.io/en/master/cluster/usage-stats.html。

    本地节点 IP: 10.94.16.4

    --------------------
    Ray 运行时已启动。
    --------------------

    下一步
    要将另一个节点添加到此 Ray 集群，请运行
        ray start --address='10.94.16.4:6379'

2. 让节点1加入 Ray 集群：

在节点1上运行以下命令：

.. code-block:: bash

```bash
ray start --address='10.94.16.4:6379'
```

运行以下命令以确认 Ray 集群现在有两个节点：

.. code-block:: bash

ray 状态

您可以看到该集群有两个节点，每个节点有16个GPU：

.. code-block:: bash

======== 自动扩展器状态: 2025-04-09 09:25:37.694016 ========
    节点状态
    ---------------------------------------------------------------
    活跃:
     1 节点_ef382ffd687d8f6b060c1b68e63ada7341b936fe5b1901dd04de1027
     1 节点_1eb4d7d07e793114c23a89d1a41f1f76acf6ef5b35af844a4ee8e4ba
    待处理:
     (无待处理节点)
    最近失败:
     (无失败)

    资源
    ---------------------------------------------------------------
    使用情况:
     0.0/360.0 CPU
     0.0/16.0 GPU
     0B/3.39TiB 内存
     0B/372.53GiB 对象存储内存

3. 运行以下脚本，使用 16 个 GPU 在 2 台机器上训练 meta-llama/Llama-3.1-8B-Instruct，TP=16:

.. code-block:: bash

    DATA_DIR=$HOME/data/gsm8k

```bash
python3 -m verl.trainer.main_ppo \
        actor_rollout_ref.rollout.name=sglang \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_batch_size=4096 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=meta-llama/Llama-3.1-8B-Instruct \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size=16 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.val_before_train=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log
```