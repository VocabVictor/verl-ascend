沙盒融合示例
============================

最后更新日期：2025年6月27日。

介绍
----

Sandbox Fusion(沙盒融合)是一个远程代码沙盒服务，为运行和评估大型语言模型（LLMs）生成的代码提供安全环境。此示例演示了如何训练一个LLM，并使用Sandbox Fusion来验证生成的代码，从而提高安全性和性能。

通过利用具有更大CPU资源的远程代码沙盒服务进行并发代码验证，您可以将奖励阶段时间缩短10-30%，具体取决于生成代码的质量。

第1步：准备数据集
---------------------------

我们使用Eurus-2-RL-Data数据集进行训练。该数据集结合了数学和代码问题，非常适合LLM训练任务。您可以从HuggingFace下载它：`Eurus-2-RL-Data数据集 <https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data>`_。

第2步：设置Sandbox Fusion服务
-----------------------------------------

Sandbox Fusion是一个远程代码沙盒服务，旨在安全运行和评估LLM生成的代码。要使用它：

1. **查看完整文档**: 有关详细的设置说明，请参阅`沙盒融合文档 <https://bytedance.github.io/SandboxFusion/>`_。

2. **部署服务**: 选择以下一种部署方法：

   - **本地部署**: 参考`此处的指南 <https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment>`_。
   - **FaaS实例（火山引擎）**: 使用`火山引擎文档 <https://www.volcengine.com/docs/6662/1539235>`_ 创建一个实例。

部署完成后，您将收到一个API端点，格式为: ``https://<ip地址或域名>/run_code``。

第三步: 配置训练脚本
-------------------------------------

要将沙盒融合集成到您的训练脚本中，请配置以下参数：

**沙盒融合的关键设置**

- ``reward_model.sandbox_fusion.url='<API-endpoint>'``: 通过指定 API 端点（必须以 ``/run_code`` 结尾）启用沙盒融合。
- ``reward_model.sandbox_fusion.max_concurrent=256``: 设置与 Sandbox Fusion 服务的最大并发 API 请求数为 256。
- ``reward_model.sandbox_fusion.memory_limit_mb=1024``: 为每个沙盒实例设置内存限制（以 MB 为单位）。如果未指定，默认为 1024MB。

**额外优化**

为了进一步减少代码验证时间，可以通过以下方式启用并行处理：

- ``reward_model.reward_manager=prime``: Prime(主要)奖励管理器可以同时跨多个子进程验证代码。

**示例脚本**

对于实际实现，请参考示例脚本：sandbox_fusion_example.rst。

``examples/ppo_trainer/run_deepseek7b_llm_sandbox_fusion.sh``

一旦在脚本中设置了API端点，您就可以开始训练作业。