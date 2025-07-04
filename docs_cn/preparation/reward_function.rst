实现数据集的奖励函数
====================

最后更新：2025年6月2日。

对于每个数据集，我们需要实现一个奖励函数或利用奖励模型来计算生成响应的奖励。
我们已经在 `reward_score 目录 <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score>`_ 中预先实现了一些奖励函数。
您也可以使用自定义的奖励函数。

目前，我们支持 GSM8k 和 MATH 数据集的奖励函数。对于 RLHF 数据集（例如，full_hh_rlhf）和代码生成（例如，APPS），我们分别利用奖励模型和 SandBox（将很快开源）进行评估。

RewardManager
-------------

在PPO后训练脚本的入口点`main_ppo.py <https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py#L33>`_中，我们实现了一个``奖励管理器(RewardManager)``，利用预先实现的奖励函数来计算每个回复的分数。

在``奖励管理器``中，我们实现了一个``__call__``函数来计算每个回复的分数。
所有的奖励函数都由``compute_score_fn``执行。
输入是一个``DataProto``，其中包括:

- ``input_ids``，``attention_mask``: 应用了chat_template后的``input_ids``和``attention_mask``，包括提示和回复
- ``responses``: 回复标记
- ``ground_truth``: 当前提示的真实字符串。
  存储在``DataProto``的``non_tensor_batch``中，应该在parquet文件中进行预处理。
- ``data_source``: 当前提示的数据集名称。存储在``DataProto``的``non_tensor_batch``中，应该在parquet文件中进行预处理。

在解标记化(responses)之后，回复字符串和真实字符串将被输入到``compute_score_fn``中，以计算每个回复的分数。

奖励函数
----------------

预先实现
~~~~~~~~~~~~~~~

我们已经在 `reward_score 目录 <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score>`_ 中预先实现了一些奖励函数。

- 在 `GSM8k 示例 <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py>`_ 中，我们强制响应在四个 #### 之后输出最终答案，然后使用字符串匹配与真实值进行比较。如果完全正确，得 1 分；如果格式正确，得 0.1 分；如果格式不正确，得 0 分。
- 在 `MATH 示例 <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py>`_ 中，我们遵循 `lm-evaluation-harness 仓库 <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py>`_ 中的实现。

自定义
~~~~~~~~~~

您可以在单独的文件中实现自定义奖励函数，并使用 ``custom_reward_function.path`` 和 ``custom_reward_function.name`` 指定它们。有关它们的集合，请参阅 :ref:`config-explain-page`。

您的奖励函数的参数应为 ``data_source``、``solution_str``、``ground_truth`` 和 ``extra_info``。
例如：

.. code:: python

  def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    return len(solution_str)/100

如果您只测试一个自定义奖励函数，您可以简单地将其命名为 'compute_score'，并将 ``custom_reward_function.name`` 保持为空。

要使用不同的自定义奖励函数进行多次测试，您可以为每次试验修改 ``custom_reward_function.path`` 和 ``custom_reward_function.name``。 
例如，您可以创建一个单独的 `my_reward.py` 文件，并在其中实现多个奖励函数。这样，对于不同的试验，您只需调整 ``custom_reward_function.name``，使得在脚本中进行多次测试更加方便。