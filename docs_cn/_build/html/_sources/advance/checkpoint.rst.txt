.. _checkpoint-page:

使用检查点支持容错训练
=========================

最后更新时间：2025年6月25日。

在整个强化学习超高频（RLHF）训练过程中可能会出现训练错误或机器故障，因此建议启用检查点以最小化损失。

API接口已在:ref:`config-explain-page`中列出，我们将不再重复。但仍有一些技术细节希望澄清。

.. 注意::

请注意，``checkpoint.contents`` 字段对于 FSDP 检查点除了 ``hf_model`` 之外没有任何影响，其他三个字段都绑定在一起进行保存和加载。我们建议同时包含 ``model``、``optimizer`` 和 ``extra``。

检查点保存目录结构
-------------------------------------

通常，我们使用在 ``ppo_trainer.yaml`` 或 ``ppo_megatron_trainer.yml`` 中声明的 ``default_local_dir`` 作为保存检查点时的前缀，即 ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``。

因此，**FSDP** 的内部检查点结构如下：

.. 代码块::

```plaintext
checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface      # 默认保存配置和分词器，如果在checkpoint.contents中包含``hf_model``，则保存huggingface模型
    │   │   └── fsdp_config.json # FSDP配置文件，包括world_size和fsdp版本
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── critic
    │   │   ├── huggingface
    │   │   └── fsdp_config.json
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    └── latest_checkpointed_iteration.txt

所有模型分片、优化器和额外状态都以分片和分布式方式存储在一起。

而**Megatron**当前的checkpoint结构为：
```

.. 代码块::

```plaintext
checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface     # 默认保存配置和分词器，如果在checkpoint.contents中包含``hf_mode``，则保存huggingface模型
    │   │   └── dist_ckpt       # 保存分片模型/优化器/rng_states，命名与Megatron相同
    │   └── critic
    │   │   ├── huggingface
    │   │   └── dist_ckpt
    └── latest_checkpointed_iteration.txt

将FSDP和Megatron Checkpoints转换为HuggingFace格式模型
--------------------------------------------------

我们提供了一个工具，用于将FSDP和Megatron的检查点转换为HuggingFace格式模型。
该工具位于``verl/model_merger``中。对于不包含fsdp_config.json在检查点中的较旧版本的verl，您可以使用位于``verl/scripts/legacy_model_merger.py``的传统模型合并工具。

该脚本支持两个主要子命令：`merge`（用于转换和保存检查点）和`test`（用于验证合并后的检查点与参考模型）。
`merge` 子命令的参数如下：

.. code:: bash
```

```bash
用法: python -m verl.model_merger merge [-h] --backend {fsdp,megatron} [--local_dir LOCAL_DIR] [--tie-word-embedding] [--is-value-model] [--use_cpu_initialization] [--target_dir TARGET_DIR]
                         [--hf_upload_path HF_UPLOAD_PATH] [--private]
```

选项:
    -h, --help            显示此帮助消息并退出
    --backend {fsdp, megatron}
                            模型的后端
    --local_dir LOCAL_DIR
                            已保存模型检查点的路径
    --tie-word-embedding  是否绑定单词嵌入权重（目前仅支持Megatron）
    --is-value-model      模型是否为值模型（目前仅支持Megatron）
    --use_cpu_initialization
                            是否对模型使用CPU初始化。这对于在初始化期间无法适应GPU内存的大型模型非常有用。
    --target_dir TARGET_DIR
                            保存合并的huggingface模型的目录
    --hf_upload_path HF_UPLOAD_PATH
                            上传模型的Hugging Face存储库ID
    --private             是否将模型上传到私有的Hugging Face存储库

合并Megatron检查点的示例用法:

.. code:: bash

```bash
python -m verl.model_merger merge \
        --backend megatron \
        --tie-word-embedding \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model
```

用于合并 FSDP checkpoints 的示例用法:

```bash
python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model
```

Megatron(梅加特龙) 合并器详细信息
-----------------------

当前的解码器层实现使用 ``nn.ModuleList`` 来存储层，因此每个 PP(rank) 和 VPP(rank) 上的模型层都从索引 0 开始。

有三种方法可以纠正这种行为:

1. 修改解码器层的 state_dict，在每个层的索引上添加 ``offset``，从而重写 ``nn.ModuleList`` 的实现。
2. 在保存检查点时修改层索引，并在加载检查点时恢复它们。
3. 检查点合并器执行此工作，仅从 ``state_dict`` 中计算实际的 ``offset``，稍微复杂。

当前的实现使用解决方案 2。

HuggingFace转换为Megatron DistCheckpoint详细信息
----------------------------------------------

如果您的模型非常庞大，我们建议您使用Megatron dist-checkpoint来加载模型。
Megatron dist-checkpoint支持使用不同类型的模型并行性，并且比原始的检查点加载速度要快得多。

要将原始的HuggingFace模型转换为Megatron dist-checkpoint，
您可以使用``scripts/converter_hf_to_mcore.py``脚本。目前支持使用CPU初始化临时支持大型MoE模型，
这会稍微慢一些。我们正在努力寻找更好的解决方案来支持大型模型。

将模型转换的示例命令如下：

```bash
python scripts/converter_hf_to_mcore.py \
    --hf_model_path Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --output_path /mnt/disk/Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --use_cpu_initialization    # 仅适用于MoE模型
```

原始检查点工具
-------------------------

原始检查点工具指的是``verl/models/[model]/megatron/checkpoint_utils``中的原始检查点实现。

现在在原始的检查点工具中，我们只需要``[model]_loader.py``，因为我们不再每次都存储``hf_model``（这在训练大型模型时并不推荐，如果可能的话尝试只保存分片模型）。

.. 注意::

请注意，``[model]_loader`` 仅支持**存储集群能够与每个计算节点连接**的环境。这是因为它利用**分片加载方式来最小化加载检查点的开销**。每个等级从``state_dict``中加载自己的数据，所有等级都可以访问这些数据。而且，在DP等级之间也无需广播，因为保存的``state_dict``仅由DP等级0生成。

对于**只能将huggingface模型放在一个设备上**的用户，我们在``[model]_loader_deprecated``中保留了原始昂贵的实现。在这个实现中，等级0将所有权重广播到每个tp和pp等级，然后dp等级0将其广播到所有dp等级。这可能存在OOM的风险。

要使用已弃用的加载器，请更改``load_state_dict_to_megatron_llama``的导入包。