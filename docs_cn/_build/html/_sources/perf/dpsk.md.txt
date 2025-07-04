# 训练 DeepSeek 671b

最后更新：2025年6月13日。

verl 集成了 Megatron，以支持大型 MoE 模型，如 `Qwen3-235B-A22B` 和 `deepseek-ai/DeepSeek-V3`。这是一个持续的社区努力。

在这个过程中，社区添加了以下功能和优化，使 verl 能够支持更大的模型：
- 在 rollout 和训练之间进行每个张量的权重重分配
- 通过 megatron 启用上下文并行和专家并行
- 为 megatron 提供动态批量大小（序列平衡）
- 减少与 ray 相关的序列化开销
- 优化器卸载、重新计算和高效内核
- 各种调试指标和工具

而 megatron 后端现在支持更广泛的模型列表：
- DeepSeek-V3
- Moonlight
- Qwen3
- Qwen2.5-VL（即将合并）
- Qwen2
- Mixtral

## 入门指南

### DeepSeek 671b

推荐使用的镜像，带有预构建的 megatron 依赖项为 `whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3`，该镜像是使用 [docker/Dockerfile.vllm.sglang.megatron.deepseek](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.vllm.sglang.megatron.deepseek) 中的 Dockerfile 构建的。

对于检查点加载，我们依赖于 megatron dist-ckpt 进行重新分片。DeepSeek-V3 的转换 dist-ckpt 可从 [huggingface BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt](https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main) 获取。

要在 DAPO 数据集上运行端到端训练，请运行 [recipe/dapo/test_dapo_dspk_671b_megatron.sh](https://github.com/volcengine/verl/blob/main/recipe/dapo/test_dapo_dspk_671b_megatron.sh)。该脚本在 512 个 H20(96GB) GPU 上运行，配置如下：
- vllm rollout，TP=32，bfloat16
- megatron 训练，注意力 DP，MoE EP=32，PP=16，bfloat16

在 RL 训练期间，MTP 被禁用。

### Qwen3 236b

对于 Qwen3-236b，请参考 [examples/grpo_trainer/run_qwen3-236b_megatron.sh](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-236b_megatron.sh)，该脚本在 128 个 H20(96GB) GPU 上运行。

## 即将到来的优化

社区继续进一步优化大型 MoE 模型，正在进行的工作包括：
- 进一步优化内存消耗，并提供各种机器类型的推荐/调优配置
- 优化长上下文 RL 训练性能
- 通过 SGLang x Megatron 提升性能

我们邀请社区共同尝试和改进verl。请通过 [slack](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)/[wechat](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG)/[Github issues](https://github.com/volcengine/verl/issues/708) 与我们联系！

## 致谢
@vermouth1992 @ISEEKYAN @ETOgaosion @yzlnew @ShareLer @BearBiscuit05 @ccclyu @ann-qin-lu @SwordFaith @zzong2006 @zhaochenyang20 @ocss884 @eric-haibin-lin