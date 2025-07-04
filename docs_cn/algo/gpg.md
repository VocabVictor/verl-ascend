# GPG: 群体策略梯度

最后更新：2025年7月3日。

群体策略梯度（GPG）是一种极简的强化学习（RL）方法，它在不依赖监督微调或复杂技巧的情况下增强了大型语言模型的推理能力。GPG 重新审视了传统的策略梯度，并直接优化 RL 目标——没有替代损失，没有 KL 惩罚，没有评论者，也没有参考模型。与 GRPO 相比，GPG 更简单、更高效，并在许多任务上取得了更好的结果。有关更多详细信息，请参阅原始论文 [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://arxiv.org/abs/2504.02546)。

## 关键组件
- 使用修正的优势函数来提高策略梯度的准确性和训练效率。
- 通过消除评论者和参考模型，避免 KL 散度约束，相较于群体相对策略优化（GRPO），显著简化了训练过程。

## 配置
要在框架中配置 GPG，请使用以下 YAML 设置。

```yaml
algorithm:
  adv_estimator: gpg 
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "gpg"
```

## 高级扩展
GPG 是模型推理的一个简单而强大的基线。尽管它在原始形式中避免使用 KL 损失，但您仍然可以使用 KL 损失来进一步提高性能。

```yaml
algorithm:
  adv_estimator: gpg
actor_rollout_ref:
  actor:
    use_kl_loss: True # enable kl regularization
    kl_loss_coef: 0.01
    policy_loss:
      loss_mode: "gpg"
```