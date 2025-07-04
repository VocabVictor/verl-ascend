Trainer 接口
================================

最后更新：2025年6月8日（API 文档字符串为自动生成）。

训练器驱动训练循环。鼓励在出现新的训练范式时引入新的训练器类。

.. autosummary::
   :nosignatures:

   verl.trainer.ppo.ray_trainer.RayPPOTrainer

核心 API
~~~~~~~~~~~~~~~~~

.. autoclass:: verl.trainer.ppo.ray_trainer.RayPPOTrainer
   :members: __init__, init_workers, fit

.. automodule:: verl.utils.tokenizer
   :members: hf_tokenizer

.. automodule:: verl.trainer.ppo.core_algos
   :members: agg_loss, kl_penalty, compute_policy_loss, kl_penalty

.. automodule:: verl.trainer.ppo.reward
   :members: load_reward_manager, compute_reward, compute_reward_async

.. autoclass:: verl.workers.reward_manager.NaiveRewardManager

.. autoclass:: verl.workers.reward_manager.DAPORewardManager