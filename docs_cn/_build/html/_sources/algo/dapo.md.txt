# 配方: 分离的剪辑和动态采样策略优化 (DAPO)

最近更新日期: 2025年6月19日。

> 开源算法实现 & 实验运行: [童宇轩](https://tongyx361.github.io/)，[盛光明](https://hk.linkedin.com/in/guangming-sheng-b50640211)

🏠 [主页](https://dapo-sia.github.io/) | 📝 [论文@arXiv](https://arxiv.org/abs/2503.14476) | 🤗 [数据集&模型@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 [代码@GitHub](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo) | 🐱 [仓库@GitHub](https://github.com/BytedTsinghua-SIA/DAPO)

我们提出了**解耦剪辑与动态采样策略优化（Decoupled Clip and Dynamic Sampling Policy Optimization，DAPO）**算法。通过公开我们的工作，我们为更广泛的研究社区和社会提供了可扩展强化学习的实际访问权限，使所有人都能从这些进步中受益。我们的系统基于出色的[verl](https://github.com/volcengine/verl)框架。感谢他们的出色工作！将DAPO训练应用于Qwen2.5-32B基础模型，证明在AIME 2024上胜过了先前的最先进DeepSeek-R1-Zero-Qwen-32B，实现了**50%**的准确率，训练步骤减少了**50%**。

![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)

## 快速入门

1. 在 Ray 集群上准备数据集。

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. 从任何机器将作业提交到 Ray 集群。

```bash
cd verl # Repo root
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml" # This sets environment variables for the Ray cluster
bash recipe/dapo/run_dapo_qwen2.5_32b.sh # or other scripts
```

## 复现运行

| 设置                                         | AIME 2024 准确率 | 硬件      | 图像                                                                 | 提交                                                                                          | 环境变量                                                                                                                           | 训练脚本                                                                                                                                               | 训练记录                                                                         |
| -------------------------------------------- | -------------- | --------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| DAPO                                         | 52%            | 16x8xH800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_qwen2.5_32b.sh)             | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO 无动态采样                              | 50%            | 16x8xH800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_wo_ds_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_wo_ds_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO 无令牌级损失和动态采样                 | 44%            | 16x8xH20  | `hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix`                    | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_early_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_early_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |

> [!重要]
>
> **📢 征集贡献！**
>
> 欢迎提交您的复现运行和设置！

## 配置

### 分离剪裁 Epsilons (-> 剪裁-更高)

一个示例配置：

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

`clip_ratio_low`和`clip_ratio_high`指定了DAPO目标中的$\varepsilon_{\text{low}}$和$\varepsilon_{\text{high}}$。

核心相关代码：

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

### 动态采样（带有分组过滤）

一个示例配置：

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # Non-positive values mean no upper limit
```

设置 `filter_groups.enable` 为 `True` 将过滤掉输出的 `metric` 都相同的组，例如，对于 `acc`，输出的准确率都为 1 或 0 的组。

训练器将重复使用 `gen_batch_size` 进行采样，直到有足够数量的符合 `train_batch_size` 的组，或达到由 `max_num_gen_batches` 指定的上限。

核心相关代码：

```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=} < {max_num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # Align the batch
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

### 灵活的损失聚合模式 (-> 标记级损失)

一个示例配置：

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
    # NOTE: "token-mean" is the default behavior
```

将`loss_agg_mode`设置为`token-mean`将意味着在一个小批次中所有序列中的所有标记上的（策略梯度）损失。  

核心相关代码：

```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
else:
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
```

### 过长奖励塑造

一个示例配置：

```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

将 `overlong_buffer.enable` 设置为 `True` 会对长度超长但仍在硬上下文限制内的输出进行惩罚。

具体来说，当输出的长度超过 `max_response_length` `0` 到 `overlong_buffer.len` 个标记时，惩罚会从 `0` 线性增加到 `overlong_buffer.penalty_factor`。

核心相关代码：

```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```

## 常见问题解答 (FAQ)

### 论文中的"过长过滤(Overlong Filtering)"在哪里？

在论文中，包括表现最佳的实验在内，大多数实验都是在没有使用"过长过滤"的情况下运行的，因为它在从最长输出中正确学习方面与"过长奖励塑造(Overlong Reward Shaping)"有一定的重叠。因此，我们在这里没有实现它。

### [在`main`分支中的`recipe/dapo`目录](https://github.com/volcengine/verl/tree/main/recipe/dapo)与[`recipe/dapo`分支](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo)有什么区别？

[`recipe/dapo`分支](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo)用于**原样复制**，因此不会随着新功能的更新而更新。

[`main`分支中的`recipe/dapo`目录](https://github.com/volcengine/verl/tree/main/recipe/dapo)作为一个示例，展示了如何扩展最新的`verl`以实现算法配方，它将随着新功能的更新而维护。

### 为什么我在修改后无法产生类似的结果？

当今的强化学习基础设施仍然存在固有的不稳定性，我们仍在努力改进。

我们强烈建议一次只修改一项内容。

我们在这里列出一些已知问题：

1. 启用CUDA图(`enforce_eager=False`)可能会导致模型性能下降，其原因仍在调查中。