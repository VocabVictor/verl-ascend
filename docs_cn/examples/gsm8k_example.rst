GSM8K 示例
=============

最后更新日期：2025年3月25日。

介绍
------------

在这个示例中，我们训练一个长短期记忆网络(LLM)来解决GSM8k任务。

文档: gsm8k_example.rst

论文链接: https://arxiv.org/pdf/2110.14168

数据集: https://huggingface.co/datasets/gsm8k

请注意，原始论文主要侧重于通过Best-of-N抽样训练一个验证器（一种奖励模型）来解决数学问题。在这个示例中，我们使用基于规则的奖励模型来训练一个RLHF代理。

数据集介绍
-----------

GSM8k是一个数学问题数据集。提示是一个小学问题。LLM模型需要回答这个数学问题。

训练集包含7473个样本，测试集包含1319个样本。

**一个示例**

Prompt

Katy在冲咖啡时使用糖勺(teaspoons)和水杯(cups)的比例为7:13。如果她总共使用了120勺糖和水杯，计算她使用了多少勺糖。

Solution

代表她用来制作咖啡的成分比例总和为7+13 = <<7+13=20>>20，因此代表她使用的茶匙数量的分数为7/20，她使用了7/20\ *120 = <<7/20*\ 120=42>>42 #### 42

步骤1：准备数据集
-----------------------

.. code:: bash

   cd examples/data_preprocess
   python3 gsm8k.py --local_dir ~/data/gsm8k

步骤2：下载模型
----------------------

有三种方式准备用于后训练的模型检查点：

- 从huggingface或modelscope下载所需模型

.. code:: bash

   huggingface-cli download deepseek-ai/deepseek-math-7b-instruct --local-dir ~/models/deepseek-math-7b-instruct --local-dir-use-symlinks False
   # 或者
   modelscope download --model deepseek-ai/deepseek-math-7b-instruct --local_dir ~/models/deepseek-math-7b-instruct

- 已经将您的存储模型存储在本地目录或HDFS路径中。
- 此外，您可以直接在运行脚本中的``actor_rollout_ref.model.path``和``critic.model.path``字段中使用huggingface中的模型名称（例如，deepseek-ai/deepseek-math-7b-instruct）。您也可以通过设置环境变量``VERL_USE_MODELSCOPE=True``从modelscope下载模型。例如，请参阅examples/ppo_trainer/run_deepseek7b_llm_modelscope.sh。

请注意，用户应为actor、critic和reward模型准备检查点。

[可选] 第三步：对模型进行SFT
---------------------------------

我们提供了一个使用PyTorch FSDP的SFT Trainer，位于`fsdp_sft_trainer.py <https://github.com/volcengine/verl/blob/main/verl/trainer/fsdp_sft_trainer.py>`_中。
用户可以使用我们的FSDP SFT Trainer自定义他们自己的SFT脚本。

我们还在`gsm8k sft directory <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k/>`_中为SFT在GSM8K数据集上提供了各种训练脚本。

.. code:: shell

```bash
set -x
```
```bash
set -x
```

```bash
torchrun -m verl.trainer.fsdp_sft_trainer \
       data.train_files=$HOME/data/gsm8k/train.parquet \
       data.val_files=$HOME/data/gsm8k/test.parquet \
       data.prompt_key=question \
       data.response_key=answer \
       data.micro_batch_size_per_gpu=8 \
       model.partial_pretrain=deepseek-ai/deepseek-coder-6.7b-instruct \
       trainer.default_hdfs_dir=hdfs://user/verl/experiments/gsm8k/deepseek-coder-6.7b-instruct/ \
       trainer.project_name=gsm8k-sft \
       trainer.experiment_name=gsm8k-sft-deepseek-coder-6.7b-instruct \
       trainer.total_epochs=4 \
       trainer.logger=['console','wandb']


如果您使用 AMD GPU（ROCm 内核），需要将以下环境变量添加到运行脚本中：

    .. code-block:: bash

        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
        export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


第四步：在 GSM8K 数据集上使用您的模型执行 PPO 训练
-------------------------------------------------------------
```

## 准备运行脚本

准备您自己的 run.sh 脚本。以下是针对 GSM8k 数据集和 deepseek-llm-7b-chat 模型的示例。

用户可以根据其环境替换 `data.train_files`、`data.val_files`、`actor_rollout_ref.model.path` 和 `critic.model.path`。

有关每个配置字段的详细解释，请参阅[配置](config)文档。

**奖励模型/函数**

我们使用基于规则的奖励模型。我们强制模型在解决方案中显示的4个“#”后生成最终答案。我们使用正则表达式匹配从解决方案和模型输出中提取最终答案。我们将它们进行比较，并对正确答案奖励1分，错误答案奖励0.1分，没有答案则奖励0分。

**训练脚本**

FSDP和Megatron-LM后端的训练脚本示例存储在examples/ppo_trainer目录中。

.. code:: bash

   cd ../ppo_trainer
   bash run_deepseek7b_llm.sh

run_deepseek7b_llm.sh脚本内容如下：

.. code:: bash

```bash
set -x
```
```bash
设置 -x
```

```bash
python3 -m verl.trainer.main_ppo \
      data.train_files=$HOME/data/gsm8k/train.parquet \
      data.val_files=$HOME/data/gsm8k/test.parquet \
      data.train_batch_size=1024 \
      data.max_prompt_length=512 \
      data.max_response_length=512 \
      actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=256 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      critic.optim.lr=1e-5 \
      critic.model.use_remove_padding=True \
      critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
      critic.model.enable_gradient_checkpointing=True \
      critic.ppo_micro_batch_size_per_gpu=32 \
      critic.model.fsdp_config.param_offload=False \
      critic.model.fsdp_config.optimizer_offload=False \
      algorithm.kl_ctrl.kl_coef=0.001 \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.project_name='verl_example_gsm8k' \
      trainer.experiment_name='deepseek_llm_7b_function_rm' \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=1 \
      trainer.total_epochs=15 $@
```

如果您使用AMD GPU（ROCm内核），您需要将以下环境变量添加到运行脚本中：

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
```

如果在使用AMD GPU运行VeRL时遇到任何问题，请随时联系我 - `苏玉生(Yusheng Su) <https://yushengsu-thu.github.io/>`_。