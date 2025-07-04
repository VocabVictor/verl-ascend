.. _快速入门:

=========================================================
快速入门：在GSM8K数据集上进行PPO训练
=========================================================

使用GSM8K数据集对大型语言模型(LLM)进行后训练。

介绍
------------

.. _hf_dataset_gsm8k: https://huggingface.co/datasets/gsm8k

在这个示例中，我们训练一个大语言模型(LLM)来处理 `GSM8k <hf_dataset_gsm8k>`_ 任务，使用基于函数的奖励。[1]_

Prerequisite:

- 按照安装指南安装最新版本的 ``verl`` 及其依赖项。推荐使用 Docker 镜像。

- 至少具有 24 GB HBM 的 GPU


数据集介绍
--------------------

GSM8k 是一个数学问题数据集。提示是一个小学数学问题。LLM 模型被要求解决这个数学问题。以下是一个示例：

Prompt

Katy 使用糖和水的比例为 7:13。如果她总共使用了 120 茶匙的糖和水，请计算她使用了多少茶匙的糖。

Solution

制作咖啡所用成分的总比例为 7+13 = <<7+13=20>>20。由于她使用的茶匙数量的分数为 7/20，因此她使用了 7/20 * 120 = <<7/20*120=42>>42 #### 42

步骤 1：准备数据集
----------------------------

我们将数据集预处理为 parquet 格式，以便 (1) 包含计算 RL 奖励所需的字段，并且 (2) 读取速度更快。

.. code-block:: bash

```plaintext
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

步骤 2：下载后训练模型
-------------------------------------------

在这个示例中，我们使用 ``Qwen2.5-0.5B-Instruct`` 模型。

如果您想在强化学习（RL）之前执行监督微调（SFT），请参考 :doc:`完整的 GSM8K 示例<../examples/gsm8k_example>`、`sft 目录 <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k>`_ 和 `SFT 训练器 <https://github.com/volcengine/verl/blob/main/verl/trainer/fsdp_sft_trainer.py>`_ 以获取更多详细信息。
```

.. code-block:: bash

```bash
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

步骤 3：使用指令模型进行 PPO 训练
----------------------------------------------------------------------

**奖励模型/函数**

我们使用一个预定义的基于规则的奖励模型。我们强制模型在解决方案中按照4个“#”后生成最终答案。我们使用正则表达式匹配从解决方案和模型输出中提取最终答案。我们为正确答案分配1的奖励，为错误答案分配0.0的奖励，为没有答案的情况分配0的奖励。

有关更多详细信息，请参阅 `verl/utils/reward_score/gsm8k.py <https://github.com/volcengine/verl/blob/v0.4.1/verl/utils/reward_score/gsm8k.py>`_。

**训练脚本**

现在让我们使用上述数据集和模型运行 PPO 训练。[2]_

根据您的数据集和模型名称或路径设置 ``data.train_files``、``data.val_files``、``actor_rollout_ref.model.path`` 和 ``critic.model.path``。
您可以设置 ``VERL_USE_MODELSCOPE=True`` 以从 `modelscope <https://www.modelscope.cn>`_ 下载模型，而不是从 `huggingface <https://huggingface.co>`_ 下载。

.. code-block:: bash

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

您应该看到以下日志，指示训练正在进行中。关键指标 ``val/test_score/openai/gsm8k`` 每 ``trainer.test_freq`` 步计算一次：

.. code-block:: bash

```
step:0 - timing/gen:21.470 - timing/ref:4.360 - timing/values:5.800 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty_coeff:0.001 - timing/adv:0.109 - timing/update_critic:15.664 - critic/vf_loss:14.947 - critic/vf_clipfrac:0.000 - critic/vpred_mean:-2.056 - critic/grad_norm:1023.278 - critic/lr(1e-4):0.100 - timing/update_actor:20.314 - actor/entropy_loss:0.433 - actor/pg_loss:-0.005 - actor/pg_clipfrac:0.000 - actor/ppo_kl:0.000 - actor/grad_norm:1.992 - actor/lr(1e-4):0.010 - critic/score/mean:0.004 - critic/score/max:1.000 - critic/score/min:0.000 - critic/rewards/mean:0.004 - critic/rewards/max:1.000 - critic/rewards/min:0.000 - critic/advantages/mean:-0.000 - critic/advantages/max:2.360 - critic/advantages/min:-2.280 - critic/returns/mean:0.003 - critic/returns/max:0.000 - critic/returns/min:0.000 - critic/values/mean:-2.045 - critic/values/max:9.500 - critic/values/min:-14.000 - response_length/mean:239.133 - response_length/max:256.000 - response_length/min:77.000 - prompt_length/mean:104.883 - prompt_length/max:175.000 - prompt_length/min:68.000
    step:1 - timing/gen:23.020 - timing/ref:4.322 - timing/values:5.953 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty:0.001 - timing/adv:0.118 - timing/update_critic:15.646 - critic/vf_loss:18.472 - critic/vf_clipfrac:0.384 - critic/vpred_mean:1.038 - critic/grad_norm:942.924 - critic/lr(1e-4):0.100 - timing/update_actor:20.526 - actor/entropy_loss:0.440 - actor/pg_loss:0.000 - actor/pg_clipfrac:0.002 - actor/ppo_kl:0.000 - actor/grad_norm:2.060 - actor/lr(1e-4):0.010 - critic/score/mean:0.000 - critic/score/max:0.000 - critic/score/min:0.000 - critic/rewards/mean:0.000 - critic/rewards/max:0.000 - critic/rewards/min:0.000 - critic/advantages/mean:0.000 - critic/advantages/max:2.702 - critic/advantages/min:-2.616 - critic/returns/mean:0.000 - critic/returns/max:0.000 - critic/returns/min:0.000 - critic/values/mean:-2.280 - critic/values/max:11.000 - critic/values/min:-16.000 - response_length/mean:232.242 - response_length/max:256.000 - response_length/min:91.000 - prompt_length/mean:102.398 - prompt_length/max:185.000 - prompt_length/min:70.000
```

查看 :ref:`algo-baseline-page` 以获取完整的训练和验证日志作为参考。

检查点默认保存在以下目录：``checkpoints/${trainer.project_name}/${trainer.experiment_name}``。您可以使用 ``verl.model_merger`` 模块将保存的检查点合并到 huggingface 模型中，例如：

.. code-block:: bash

```bash
python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_1/actor \
        --target_dir checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_1/actor/huggingface
```

有关检查点（checkpoint）和模型合并（model merging）的更多详细信息，请参阅 :ref:`checkpoint-page`。

要启用 ``wandb`` 进行实验跟踪，请设置以下配置：

.. code-block:: bash

```plaintext
trainer.logger=['console','wandb'] \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \

如果您在HBM小于32GB的情况下遇到内存不足的问题，启用以下配置可能会有所帮助：
```

.. code-block:: bash

actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_micro_batch_size_per_gpu=1 \

有关完整的配置集，请参阅 :ref:`config-explain-page` 以获取详细说明和性能调优。

.. [1] 原始论文 (https://arxiv.org/pdf/2110.14168) 主要集中于训练一个验证器（奖励模型）通过 Best-of-N 采样来解决数学问题。在这个例子中，我们使用基于规则的奖励模型训练一个强化学习（RL）代理。
.. [2] 更多关于 FSDP 和 Megatron-LM 后端的训练脚本示例存储在 `examples/ppo_trainer <https://github.com/volcengine/verl/tree/main/examples/ppo_trainer>`_ 目录中。