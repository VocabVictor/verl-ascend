<div align="center">
 👋 大家好! 
    verl 是由<b>字节跳动种子团队</b>发起并由 verl 社区维护的强化学习训练库。
    <br>
    <br>
</div>

<div align="center">

[<img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" height="20"/>](https://deepwiki.com/volcengine/verl)
[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

<h1 style="text-align: center;">verl: 火山引擎大语言模型强化学习框架</h1>

verl 是一个为大语言模型(LLMs)设计的灵活、高效且生产就绪的强化学习训练库。

verl 是 **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** 论文的开源版本。

verl 具有灵活易用的特点：

- **多样化RL算法的轻松扩展**: 混合控制器编程模型支持复杂后训练数据流的灵活表示和高效执行。仅需几行代码即可构建 GRPO、PPO 等RL数据流。

- **模块化API与现有LLM基础设施的无缝集成**: 解耦计算和数据依赖，实现与现有LLM框架的无缝集成，如 FSDP、Megatron-LM、vLLM、SGLang 等。

- **灵活的设备映射**: 支持将模型灵活部署到不同的GPU集合上，以实现高效的资源利用和跨不同集群规模的可扩展性。

- 与流行的 HuggingFace 模型的即插即用集成

verl 具有高性能特点：

- **最先进的吞吐量**: 集成了最先进的LLM训练和推理引擎，实现了最高的RL吞吐量。

- **3D-HybridEngine高效的Actor模型重分片**: 消除内存冗余，显著减少训练和生成阶段转换期间的通信开销。

## 新闻

- [2025/06] 基于 Megatron 后端的 verl 支持大型 MoE 模型，如 [DeepSeek-671b 和 Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html)。
- [2025/06] verl 团队将在6月7日的 [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) 提供最新项目更新。欢迎在北京与我们的开发团队见面！
- [2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957)，被 ICML 2025 接收，现已在 verl 中支持！PF-PPO 通过过滤潜在噪声奖励信号并通过重放缓冲区重用高质量经验来增强策略学习效率和鲁棒性。
- [2025/04] **verl-ascend** 版本支持华为昇腾NPU，为国产算力生态提供强化学习训练支持。

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/VocabVictor/verl-ascend.git
cd verl-ascend

# 安装依赖
pip install -r requirements.txt

# 安装 verl
pip install -e .
```

### 华为昇腾NPU支持

本版本特别支持华为昇腾NPU，为国产算力生态提供强化学习训练能力：

```bash
# 安装昇腾NPU依赖
pip install -r requirements-npu.txt
```

### 基础使用示例

```python
# PPO训练示例
python examples/ppo_trainer/run_qwen2-7b_rm.sh

# GRPO训练示例  
python examples/grpo_trainer/run_qwen2-7b.sh

# 多模态训练示例
python examples/grpo_trainer/run_qwen2_5_vl-7b.sh
```

## 主要特性

### 🚀 多种强化学习算法支持
- **PPO (Proximal Policy Optimization)**: 经典的策略优化算法
- **GRPO (Group Relative Policy Optimization)**: 群体相对策略优化
- **DPO (Direct Preference Optimization)**: 直接偏好优化
- **SPIN (Self-Play Fine-tuning)**: 自对弈微调
- **SPPO (Self-Play Preference Optimization)**: 自对弈偏好优化

### 🔧 灵活的后端支持
- **训练后端**: FSDP、Megatron-LM
- **推理后端**: vLLM、SGLang、HuggingFace Transformers
- **硬件支持**: NVIDIA GPU、华为昇腾NPU

### 📊 高级功能
- **多轮对话训练**: 支持复杂的多轮对话场景
- **工具集成**: 搜索工具、代码执行工具、数学工具
- **多模态支持**: 视觉-语言模型训练
- **分布式训练**: 支持多机多卡训练

## 算法介绍

### PPO (Proximal Policy Optimization)
经典的强化学习算法，通过限制策略更新幅度来保证训练稳定性。

### GRPO (Group Relative Policy Optimization)  
一种新的策略优化方法，通过群体相对比较来提升训练效果。

### DPO (Direct Preference Optimization)
直接优化人类偏好，无需显式奖励模型的简化训练方法。

## 示例和教程

### 数学推理任务
```bash
# GSM8K数学推理训练
cd examples/grpo_trainer
bash run_qwen2-7b_math.sh
```

### 多模态训练
```bash
# 视觉-语言模型训练
cd examples/grpo_trainer  
bash run_qwen2_5_vl-7b.sh
```

### 多轮对话
```bash
# 多轮对话训练
cd examples/sglang_multiturn
bash run_qwen2.5-3b_gsm8k_multiturn.sh
```

## 文档

完整文档请访问：[https://vocabvictor.github.io/verl-ascend/](https://vocabvictor.github.io/verl-ascend/)

包含内容：
- 安装指南
- 快速开始教程  
- 算法详解
- API参考
- 高级用法
- 性能调优

## 贡献

欢迎为 verl-ascend 项目贡献代码！请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

## 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系我们

- GitHub Issues: [https://github.com/VocabVictor/verl-ascend/issues](https://github.com/VocabVictor/verl-ascend/issues)
- 原项目: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)

## 致谢

感谢字节跳动种子团队开源原始 verl 项目，本项目基于原项目进行华为昇腾NPU适配和功能扩展。