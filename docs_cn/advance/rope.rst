RoPE缩放覆盖
=======================================

最近更新日期：2025年5月14日。

一些模型，如`Qwen/Qwen2.5-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts>`_，支持RoPE缩放，但在它们的config.json文件中没有定义。
例如，该模型支持以下配置：

.. code:: python

    {
        ...,
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    }

为了支持这种模型的更长上下文，您必须在启动训练器时覆盖模型配置。

PPO示例：

```bash
+actor_rollout_ref.model.override_config.rope_scaling.type=yarn \
+actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
+actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=32768 \

对于评论家模型

```bash
+critic.model.override_config.rope_scaling.type=yarn \
+critic.model.override_config.rope_scaling.factor=4.0 \
+critic.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
```