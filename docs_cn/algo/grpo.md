# 组相对策略优化 (GRPO)

最近更新时间: 05/31/2025.

在强化学习中，像PPO这样的经典算法依赖于一个"评论家"模型来估计动作的价值，从而指导学习过程。然而，训练这个评论家模型可能会消耗大量资源。

GRPO通过消除对独立评论家模型的需求来简化这个过程。相反，它的操作如下：
- 组抽样: 对于给定问题，模型生成多个可能的解决方案，形成一个输出的"组"。
- 奖励分配: 对每个解决方案进行评估，并根据其正确性或质量分配奖励。
- 基线计算: 组的平均奖励作为基线。
- 策略更新: 模型通过将每个解决方案的奖励与组基线进行比较来更新其参数，加强高于平均水平的解决方案，抑制低于平均水平的解决方案。

这种方法通过避免训练单独的值估计模型来减少计算开销，使学习过程更加高效。更多细节，请参考原始论文[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

## 关键组件

- 无值函数（无评论者）：与PPO不同，GRPO不训练单独的值网络（评论者）
- 组抽样（分组展开）：GRPO不是对每个输入评估一个展开，而是为每个提示从当前策略生成多个完成（响应）。这组完成被称为一个组。
- 相对奖励：在每个组内，完成被评分（例如，基于正确性），奖励相对于组进行归一化。

## 配置

请注意，所有包含 `micro_batch_size` 的配置用于配置每次前向或后向传递的最大样本或标记数，以避免 GPU 内存溢出，其值不应更改算法/收敛行为。

尽管许多配置以 `ppo_` 前缀开头，但它们适用于 verl 中不同的强化学习算法，因为 GRPO 训练循环类似于 PPO（没有评论者）。

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `actor_rollout.ref.rollout.n`: 对于每个提示，采样 n 次。默认为 1。对于 GRPO，请将其设置为大于 1 的值以进行组采样。

- `data.train_batch_size`: 用于生成一组采样轨迹/回放的提示的全局批量大小。响应/轨迹的数量为 `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`: 采样轨迹集被拆分为多个小批量，每个小批量的大小为 ppo_mini_batch_size，用于 PPO 演员更新。ppo_mini_batch_size 是所有工作者的全局大小。

- `actor_rollout_ref.actor.ppo_epochs`: 对于演员在一组采样轨迹上的 GRPO 更新的轮数

- `actor_rollout_ref.actor.clip_ratio`: GRPO 剪切范围。默认为 0.2

- `algorithm.adv_estimator`: 默认值为 gae。请将其设置为 grpo。

- `actor_rollout_ref.actor.loss_agg_mode`: 默认值为 "token-mean"。可选项包括 "token-mean"、"seq-mean-token-sum"、"seq-mean-token-mean"。原始的 GRPO 论文采用样本级损失 (seq-mean-token-mean)，在长链的上下文 (long-CoT) 场景中可能不稳定。verl 提供的所有 GRPO 示例脚本都使用默认配置 "token-mean" 进行损失聚合。

与其在奖励中添加 KL 惩罚，GRPO 通过直接将训练策略与参考策略之间的 KL 散度添加到损失中进行正则化：

- `actor_rollout_ref.actor.use_kl_loss`: 在演员中使用 KL 损失。当使用时，我们不在奖励函数中应用 KL。默认值为 False。请将其设置为 True 以用于 GRPO。

- `actor_rollout_ref.actor.kl_loss_coef`: KL 损失的系数。默认值为 0.001。

- `actor_rollout_ref.actor.kl_loss_type`: 支持 kl(k1)、abs、mse(k2)、low_var_kl(k3) 和 full。用于计算演员与参考策略之间的 KL 散度。有关详细分析，请参见此博客文章: http://joschu.net/blog/kl-approx.html

## 高级扩展

### DrGRPO

[理解 R1-零类训练：一种批判性视角](https://arxiv.org/pdf/2503.20783) 声称 GRPO 中存在优化偏差，这导致了人为延长的响应，尤其是对于错误输出。这种低效源于 GRPO 使用基于组的奖励归一化计算优势的方式。相反，DrGRPO 通过使用全局常数进行归一化来聚合令牌级损失，以消除长度偏差。

配置以下内容以启用 DrGRPO，其他参数与 GRPO 相同：

- `actor_rollout_ref.actor.loss_agg_mode`： "seq-mean-token-sum-norm"，这会关闭序列维度平均
- `actor_rollout_ref.actor.use_kl_loss`：请将其设置为 False 以用于 DrGRPO
- `algorithm.norm_adv_by_std_in_grpo`： False，这会关闭标准差归一化

## 参考示例

Qwen2.5 GRPO 训练日志和命令：[链接](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log)

```bash
bash examples/grpo_trainer/run_qwen3-8b.sh
```

有关更多参考性能的信息，请参见 https://verl.readthedocs.io/en/latest/algo/baseline.html