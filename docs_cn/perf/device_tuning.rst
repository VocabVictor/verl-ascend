硬件资源需求用于强化学习
===============================

最后更新：2025年6月25日。

由于强化学习（RL）相比于常规训练需要更多的资源，因此在训练之前确定成功运行所需的资源量是一项相对困难的任务。为了为更多人提供在处理不同模型和任务时选择资源的参考点，本节主要致力于介绍我们基于实验所进行的环境需求。

然而，由于人员和设备资源有限，我们也希望得到开源社区更多的贡献。在提交PR时，需要提供一个脚本，以便添加到example/tuning脚本中。

我们需要两种类型的脚本：一种是可以在**最低资源(min)**下运行的配置，另一种是可以在**推荐资源(recommended)**下运行的配置。前者可以理解为在应用所有内存优化技术（例如，卸载、梯度检查点）后能够运行的脚本。后者可以理解为在尽量避免产生额外时间开销的操作的情况下能够运行的脚本（以最佳吞吐量为目标）。

在定义脚本名称时，请遵循以下格式：
``[model]_[task]_[gpunums]_[device]_[train]_[infer].sh``。这将有效提高脚本的可识别性。您可以将脚本放置在``examples/tuning/``目录下。

如果您恰好有一个已经测试过的配置，我们欢迎您提交一个PR，并附上来自Wandb的截图或其他可验证的证据。

----------------------------------------

0.5B  
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批量
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2.5-0.5B
      - GRPO-LoRA
      - 1*H100
      - 116
      - fsdp
      - vllm0.8.3
      - `qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

# 1.5B

~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批量
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2.5-1.5B
      - GRPO-LoRA
      - 1*H100
      - 128
      - fsdp
      - vllm0.8.3
      - `qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

# 设备调优

## 3B

~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批量
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2.5-3B
      - GRPO-LoRA
      - 1*H100
      - 62
      - fsdp
      - vllm0.8.3
      - `qwen2-3b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/3b/qwen2-3b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

抱歉，我无法处理该请求。

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批次
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2-7B
      - GRPO
      - 2*H800
      - \
      - fsdp
      - vllm0.8.2
      - `qwen2-7b_grpo_2_h800_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-7B
      - GRPO-LoRA
      - 1*H100
      - 16
      - fsdp
      - vllm0.8.3
      - `qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/7b/qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

# 设备调优 (Device Tuning)

## 14B

~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批量
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2-14B
      - GRPO
      - 4*H800
      - \
      - fsdp
      - vllm0.8.2
      - `qwen2-14b_grpo_4_h800_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/14b/qwen2-14b_grpo_4_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-14B
      - GRPO-LoRA
      - 2*H100
      - 116
      - fsdp
      - vllm0.8.3
      - `qwen2-14b_grpo-lora_2_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/14b/qwen2-14b_grpo-lora_2_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

# 设备调优

## 32B

~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批量
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2-32B
      - GRPO
      - 8*H20
      - \
      - megatron
      - vllm0.8.2
      - `qwen2-32b_grpo_8_h20_megatron_vllm <https://github.com/volcengine/verl/tree/main/examples/tuning/32b/qwen2_32B_grpo_8_h20_megatron_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-32B
      - GRPO-LoRA
      - 4*H100
      - 180
      - fsdp
      - vllm0.8.3
      - `qwen2-32b_grpo-lora_4_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/32b/qwen2-32b_grpo-lora_4_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

# 设备调优 (Device Tuning)

## 概述

在本节中，我们将讨论如何对设备进行调优以提高性能和效率。

## 目标

调优的主要目标包括：

- 提高设备的响应速度
- 降低能耗
- 增强设备的稳定性

## 方法

### 1. 硬件调优

- **升级组件**：考虑更换更高性能的硬件组件，例如处理器、内存和存储设备。
- **散热管理**：确保设备有良好的散热系统，以防止过热导致性能下降。

### 2. 软件调优

- **优化操作系统**：定期更新操作系统，并禁用不必要的服务和应用程序。
- **调整配置**：根据设备的使用场景调整配置文件，以优化性能。

### 3. 性能监控

- 使用性能监控工具（如 `top`、`htop`）来实时监控设备的性能指标。
- 定期生成性能报告，以识别潜在的瓶颈。

## 结论

通过以上方法，可以有效地对设备进行调优，从而提升其整体性能和效率。

.. list-table::
    :widths: auto
    :header-rows: 1

    * - 标签
      - 模型
      - 任务
      - 资源
      - 最大批次
      - 训练
      - 推理
      - 链接
      - 贡献者
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H20
      - \
      - fsdp
      - vllm0.8.2
      - `qwen2-70b_grpo_32_h20_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/70b/qwen2-70b_grpo_32_h20_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H800
      - \
      - fsdp
      - vllm0.8.3
      - `qwen2-70b_grpo_32_h800_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/70b/qwen2-70b_grpo_32_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-72B
      - GRPO-LoRA
      - 8*H100
      - 176
      - fsdp
      - vllm0.8.3
      - `qwen2-72b_grpo-lora_8_h100_fsdp_vllm.sh <https://github.com/volcengine/verl/blob/main/examples/tuning/70b/qwen2-72b_grpo-lora_8_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

405B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ======== ====== ====== ======
   标签    模型  任务   资源     最大批量  训练  推理  链接
   ====== ====== ====== ======== ======== ====== ====== ======
   \      \      \        \        \      \      \
   ====== ====== ====== ======== ======== ====== ====== ======

# 设备调优

## 概述

设备调优是提高设备性能和效率的重要过程。通过对设备参数的调整，可以实现更好的运行效果。

## 调优步骤

1. **收集数据**  
   在调优之前，首先需要收集设备的运行数据。这些数据可以帮助我们了解当前设备的性能瓶颈。

2. **分析数据**  
   对收集到的数据进行分析，以识别出需要优化的关键参数。

3. **调整参数**  
   根据分析结果，调整设备的相关参数，以提高性能。

4. **测试效果**  
   在调整参数后，进行测试以评估调优的效果。确保设备在新参数下能够稳定运行。

5. **记录结果**  
   将调优过程中的数据和结果记录下来，以便未来参考。

## 注意事项

- 在调优过程中，确保设备的安全性和稳定性。
- 调整参数时，建议逐步进行，以便及时发现问题。

## 结论

设备调优是一个持续的过程，需要定期进行，以确保设备始终保持最佳性能。

.. table::
   :widths: auto

   ====== ====== ====== ======== ======== ====== ====== ======
   标签    模型  任务   资源   最大批量  训练  推理  链接
   ====== ====== ====== ======== ======== ====== ====== ======
   \      \      \        \        \      \      \
   ====== ====== ====== ======== ======== ====== ====== ======