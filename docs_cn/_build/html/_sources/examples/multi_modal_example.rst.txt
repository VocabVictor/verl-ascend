多模态示例架构
=================================

最后更新日期：2025年4月28日。

介绍
------------

现在，verl已经支持多模态训练。您可以使用fsdp和vllm/sglang来启动一个多模态强化学习任务。Megatron的支持也正在进行中。

按照以下步骤快速开始一个多模态强化学习任务。

步骤 1: 准备数据集
-----------------------

.. code:: python

    # 数据集将保存在 $HOME/data/geo3k 文件夹中
    python examples/data_preprocess/geo3k.py

步骤 2: 下载模型
----------------------

.. code:: bash

    # 从huggingface下载模型
    python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

步骤 3: 在Geo3K数据集上使用多模态模型进行GRPO训练
---------------------------------------------------------------------

.. code:: bash

    # 运行任务
    bash examples/grpo_trainer/run_qwen2_5_vl-7b.sh