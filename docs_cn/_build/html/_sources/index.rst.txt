欢迎来到verl的文档！
================================================

verl是一个灵活、高效且适用于生产环境的强化学习（RL）训练框架，专为大型语言模型（LLMs）后训练设计。它是`HybridFlow <https://arxiv.org/pdf/2409.19256>`_ 论文的开源实现。

verl灵活且易于使用，具有以下特点：

- **多样化RL算法的简单扩展**：混合编程模型结合了单控制器和多控制器范式的优点，能够灵活表示和高效执行复杂的后训练数据流。用户可以用几行代码构建RL数据流。

- **与现有LLM基础设施的无缝集成和模块化API**：解耦计算和数据依赖，能够与现有的LLM框架（如PyTorch FSDP、Megatron-LM、vLLM和SGLang）无缝集成。此外，用户可以轻松扩展到其他LLM训练和推理框架。

- **灵活的设备映射和并行性**：支持将模型放置到不同的GPU集上，以实现高效的资源利用和在不同集群规模上的可扩展性。

- 与流行的 HuggingFace 模型的快速集成

verl 的速度表现优异，具体体现在：

- **最先进的吞吐量**：通过无缝集成现有的 SOTA (最先进技术) LLM (大语言模型) 训练和推理框架，verl 实现了高效的生成和训练吞吐量。

- **使用 3D-HybridEngine 的高效 Actor 模型重分片**：消除了内存冗余，并显著减少了训练和生成阶段之间转换时的通信开销。

--------------------------------------------

.. _目录:

.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   start/install
   start/quickstart
   start/multinode
   start/ray_debug_tutorial
   start/more_resources

.. toctree::
   :maxdepth: 2
   :caption: 编程指南

```rst
hybrid_flow
===========

single_controller
=================
```

.. toctree::
   :maxdepth: 1
   :caption: 数据准备

# 准备数据 (preparation/prepare_data)

在本节中，我们将讨论如何准备数据以供模型使用。数据准备是机器学习流程中的关键步骤，它直接影响模型的性能和准确性。

## 数据收集

首先，您需要收集相关的数据。这可以通过多种方式完成，例如从公开数据集下载、使用API获取数据或通过爬虫抓取网页数据。

## 数据清洗

收集到的数据通常需要进行清洗，以去除噪声和不相关的信息。数据清洗的步骤包括：

- 删除重复项
- 处理缺失值
- 标准化数据格式

## 数据划分

在准备好数据后，您需要将其划分为训练集和测试集。通常，训练集占总数据的70%-80%，而测试集占20%-30%。这种划分有助于评估模型的性能。

# 奖励函数 (preparation/reward_function)

奖励函数是强化学习中的一个重要概念。它用于评估代理在特定状态下采取某个动作的好坏。设计一个合适的奖励函数对于训练有效的模型至关重要。

## 奖励函数的设计

设计奖励函数时，您需要考虑以下几点：

- **明确目标**：确定您希望代理实现的目标，并据此设计奖励。
- **即时奖励与长期奖励**：考虑代理在短期和长期内的奖励，确保奖励函数能够引导代理朝着正确的方向发展。
- **避免稀疏奖励**：如果奖励过于稀疏，代理可能难以学习。可以通过提供更频繁的反馈来解决这个问题。

## 奖励函数的示例

以下是一个简单的奖励函数示例：

```python
def reward_function(state, action):
    if action == 'desired_action':
        return 1  # 正奖励
    else:
        return -1  # 负奖励
```

通过合理设计奖励函数，您可以有效地引导代理学习并优化其策略。

.. toctree::
   :maxdepth: 2
   :caption: 配置

   examples/config

```rst
.. toctree::
   :maxdepth: 1
   :caption: PPO 示例

   examples/ppo_code_architecture
   examples/gsm8k_example
   examples/multi_modal_example

.. toctree::
   :maxdepth: 1
   :caption: 算法

   algo/ppo.md
   algo/grpo.md
   algo/dapo.md
   algo/spin.md
   algo/sppo.md
   algo/entropy.md
   algo/opo.md
   algo/baseline.md
   algo/gpg.md

.. toctree::
   :maxdepth: 1
   :caption: PPO 训练器和工作者

   workers/ray_trainer
   workers/fsdp_workers
   workers/megatron_workers
   workers/sglang_worker

.. toctree::
   :maxdepth: 1
   :caption: 性能调优指南

   perf/dpsk.md
   perf/perf_tuning
   README_vllm0.8.md
   perf/device_tuning
   perf/nsight_profiling.md

.. toctree::
   :maxdepth: 1
   :caption: 添加新模型
```

# advance/fsdp_extension

# advance/megatron_extension

.. toctree::
   :maxdepth: 1
   :caption: 高级功能

   advance/checkpoint
   advance/rope
   advance/ppo_lora.rst
   sglang_multiturn/multiturn.rst
   sglang_multiturn/interaction_system.rst
   advance/placement
   advance/dpo_extension
   examples/sandbox_fusion_example

.. toctree::
   :maxdepth: 1
   :caption: 硬件支持

   amd_tutorial/amd_build_dockerfile_page.rst
   amd_tutorial/amd_vllm_page.rst
   ascend_tutorial/ascend_quick_start.rst

.. toctree::
   :maxdepth: 1
   :caption: API 参考

   api/data
   api/single_controller.rst
   api/trainer.rst
   api/utils.rst


.. toctree::
   :maxdepth: 2
   :caption: 常见问题解答

   faq/faq

.. toctree::
   :maxdepth: 1
   :caption: 开发笔记

   sglang_multiturn/sandbox_fusion.rst

贡献
-------------

verl 是一款自由软件；您可以根据 Apache 许可证 2.0 的条款重新分发和/或修改它。我们欢迎贡献。
请在 `GitHub <https://github.com/volcengine/verl>`_、`Slack <https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA>`_ 和 `Wechat <https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG>`_ 加入我们进行讨论。

欢迎社区的贡献！请查看我们的 `项目路线图 <https://github.com/volcengine/verl/issues/710>`_ 和 `适合新手的问题 <https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`_ 以了解您可以贡献的地方。

代码检查和格式化
^^^^^^^^^^^^^^^^^^^^^^

我们使用 pre-commit 来帮助提高代码质量。要初始化 pre-commit，请运行：

.. code-block:: bash

```rst
pip install pre-commit
   pre-commit install

要在本地解决 CI 错误，您还可以手动运行 pre-commit：
```

.. code-block:: bash

pre-commit 运行

添加 CI 测试
^^^^^^^^^^^^^^^^^^^^^^^^

如果可能，请为您的新功能添加 CI 测试：

1. 找到最相关的工作流 yml 文件，通常对应于 ``hydra`` 默认配置（例如 ``ppo_trainer``、``ppo_megatron_trainer``、``sft_trainer`` 等）。
2. 如果尚未包含，请将相关路径模式添加到 ``paths`` 部分。
3. 尽量减少测试脚本的工作量（请参阅现有脚本以获取示例）。

我们正在招聘！如果您对 MLSys/LLM 推理/多模态对齐的实习/全职机会感兴趣，请发送电子邮件给我们 `email <mailto:haibin.lin@bytedance.com>`_。