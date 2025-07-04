# 最优奖励基线的在线策略强化学习 (OPO)

最后更新：2025年6月2日。

在强化学习中，松散的在线策略约束和次优基线通常会导致训练不稳定，例如大幅度的策略变化和熵崩溃。OPO通过使用精确的在线策略训练和理论上的最优奖励基线进行优势估计，解决了这些挑战。它实现了更低的策略变化和更高的输出熵，鼓励生成更具多样性和更少重复的响应。

OPO使用组采样为每个输入生成多个输出，类似于GRPO。与通常使用组的平均奖励作为基线的基于组的算法不同，OPO采用理论上的最优基线：组的长度加权奖励。它还省略了标准差归一化。通过采用这两个关键组件，OPO使得训练单一策略模型的目标仅为最大化期望奖励。有关更多细节，请参考原始论文 [On-Policy RL with Optimal Reward Baseline](https://arxiv.org/pdf/2505.23585)。

## 关键组件

- 精确的在线策略训练：始终从当前策略生成响应，而不使用任何预生成的数据或离策略数据。
- 最优奖励基线：使用组的长度加权奖励作为归一化奖励的基线。

## 配置

要在框架内配置OPO，请使用以下YAML设置。这些参数对于启用精确的策略训练和激活最佳奖励基线至关重要。

```yaml
algorithm:
  adv_estimator: opo  # Use OPO for optimal reward baseline 
data:
  train_batch_size: 1024
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 1024 # ppo_mini_batch_size should equal to train_batch_size to enable exact on-policy training
    entropy_coeff: 0 # disable entropy regularization
    use_kl_loss: False # disable kl regularization
    kl_loss_coef: 0 
```

## 高级扩展

OPO 还可以扩展到其他算法，如 RLOO 和 Reinforce++。只需调整它们的配置，以启用精确的在政策(on-policy)训练，并将最优的长度加权奖励基线纳入其中，对它们的优势估计函数进行最小的修改。