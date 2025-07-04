# 近端策略优化 (PPO)

最后更新：2025年6月19日。

近端策略优化 (PPO) 是一类用于强化学习的策略梯度方法，由OpenAI于2017年提出。PPO在简单性、稳定性和性能之间取得了平衡，使其成为现代强化学习应用中最广泛使用的算法之一，包括大规模语言模型的微调。

传统的策略梯度方法，如REINFORCE或普通策略梯度，存在以下问题：

- 高方差和样本效率低下。
- 由于策略更新过大而导致的不稳定性。

PPO通过使用剪切的替代目标来解决这个问题，避免了过大的更新，同时不需要二阶导数。

有关PPO的更多技术细节，我们建议阅读[OpenAI spinning up教程](https://spinningup.openai.com/en/latest/algorithms/ppo.html)中的介绍，以及论文[近端策略优化算法](https://arxiv.org/abs/1707.06347)。

## 关键组件

- Actor-Critic架构：PPO同时需要一个actor模型（策略）和一个critic模型（价值函数）。这与不需要critic模型的其他算法（如GRPO和RLOO）不同。

# PPO

- 广义优势估计 (Generalized Advantage Estimation, GAE): PPO 使用 GAE 来计算优势值，这有助于减少策略梯度估计中的方差，同时保持低偏差。

- 裁剪替代目标 (Clipped Surrogate Objective): PPO 的核心是通过裁剪替代目标函数来实现的，该函数限制了策略更新。

## 配置

注意，所有包含 `micro_batch_size` 的配置用于设置每次前向或后向传播的最大样本或标记数量，以避免 GPU 内存溢出（OOM），其值不应改变算法或收敛行为。

大多数评论者（critic）配置与演员（actor）的配置类似。请注意，下面的图中省略了评论者模型。

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `data.train_batch_size`: 用于生成一组采样轨迹/回放的全局批量大小。响应/轨迹的数量为 `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`: 采样轨迹集被拆分为多个小批量，每个小批量的大小为 ppo_mini_batch_size，用于 PPO 演员更新。ppo_mini_batch_size 是所有工作节点的全局大小

- `actor_rollout_ref.critic.ppo_mini_batch_size`: 采样轨迹集被拆分为多个小批量，每个小批量的大小为 ppo_mini_batch_size，用于 PPO 评论员更新。ppo_mini_batch_size 是所有工作节点的全局大小

- `actor_rollout_ref.actor.clip_ratio`: PPO 剪切范围。默认为 0.2

- `actor_rollout_ref.actor.ppo_epochs`: 对于演员，PPO 更新一组采样轨迹的轮数

- `critic.ppo_epochs`: 对于评论员，PPO 更新一组采样轨迹的轮数。默认为 `actor_rollout_ref.actor.ppo_epochs`

- `algorithm.gemma`: 折扣因子

- `algorithm.lam`: 在 GAE 估计器中平衡偏差和方差的 lambda 项

- `algorithm.adv_estimator`: 支持 gae、grpo、reinforce_plus_plus、reinforce_plus_plus_baseline、rloo

## 高级扩展

### KL 散度控制

用于防止策略与参考策略过度偏离的选项。提供两种机制：KL 奖励惩罚和 KL 损失。有关更多技术细节，请参见 [用人类反馈训练语言模型以遵循指令](https://arxiv.org/abs/2203.02155)

用于 KL 散度控制的 KL 损失选项：

- `actor_rollout_ref.actor.use_kl_loss`: 在演员中使用 KL 损失。当使用时，我们不在奖励函数中应用 KL。默认值为 False

- `actor_rollout_ref.actor.kl_loss_coef`: KL 损失的系数。默认值为 0.001。

- `actor_rollout_ref.actor.kl_loss_type`: 支持 kl(k1)、abs、mse(k2)、low_var_kl(k3) 和 full。计算演员与参考策略之间的 KL 散度的方法。有关详细分析，请参见这篇博客文章：http://joschu.net/blog/kl-approx.html

在奖励中使用 KL 惩罚的选项：

- `algorithm.use_kl_in_reward`: 是否启用奖励中的 KL 惩罚。默认值为 False。

- `algorithm.kl_penalty`: 支持 kl(k1)、abs、mse(k2)、low_var_kl(k3) 和 full。这定义了计算演员(Actor)与参考策略(Reference Policy)之间的 KL 散度的方式。有关具体选项，请参阅 `core_algos.py` 中的 `kl_penalty`。有关详细分析，请参见这篇博客文章: http://joschu.net/blog/kl-approx.html

- `algorithm.kl_ctrl.kl_coef`: 奖励内 KL 散度惩罚的 (初始) 系数。默认值为 0.001。
- `algorithm.kl_ctrl.type`: 'fixed' 表示固定 KL 控制器(FixedKLController)，'adaptive' 表示自适应 KL 控制器(AdaptiveKLController)。
- `algorithm.kl_ctrl.horizon`: 有关详细信息，请参见自适应 KL 控制器(AdaptiveKLController)的源代码。
- `algorithm.kl_ctrl.target_kl`: 有关详细信息，请参见自适应 KL 控制器(AdaptiveKLController)的源代码。

### 双重剪切 PPO

双重剪切 PPO 通过在优势小于零时对策略比率施加下限来引入一种方法，当乘以一个较大的比率时，不超过指定的下限。

![image](https://github.com/user-attachments/assets/fc232181-d8b0-4307-8dd2-4dc0a4c1c139)

- `actor_rollout_ref.actor.clip_ratio_c`: 双重剪辑 PPO 的值的下限，默认为 3.0

## 参考示例

Qwen2.5 训练日志和命令: [链接](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log)

```bash
bash run_gemma.sh
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  trainer.logger=['console'] \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=2 \
  critic.ppo_micro_batch_size=2
```

参考性能与 verl v0.2：

| 模型                          | 方法            | 分数  | 链接                                                                                           |
|-------------------------------|------------------|-------|------------------------------------------------------------------------------------------------|
| Qwen/Qwen2.5-0.5B-Instruct     | 预训练模型      | 36.4  | [Qwen 博客](https://qwenlm.github.io/blog/qwen2.5-llm/)                                        |
| Qwen/Qwen2.5-0.5B-Instruct     | PPO              | 56.7  | [PPO 命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log) |