# 配方: 熵机制

最后更新日期: 2025年6月27日。

<div align="center">

# 大型语言模型推理的强化学习熵机制

[![论文](https://img.shields.io/badge/论文-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2505.22617)  [![Github](https://img.shields.io/badge/PRIME-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL) [![alphaXiv](https://img.shields.io/badge/讨论-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue
)](https://www.alphaxiv.org/abs/2505.22617) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/stingning/status/1928088554166505667) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/charlesfornlp/status/1928089451080585283) [![Twitter-ak](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/_akhaliq/status/1928077929105268861)

```html
<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#🎉新闻" style="text-decoration: none; font-weight: bold;">🎉 新闻</a> •
    <a href="#✨入门指南" style="text-decoration: none; font-weight: bold;">✨ 入门指南</a> •
    <a href="#📖介绍" style="text-decoration: none; font-weight: bold;">📖 介绍</a>
  </p>
  <p>
    <a href="#🎈引用" style="text-decoration: none; font-weight: bold;">🎈 引用</a> •
    <a href="#🌻致谢" style="text-decoration: none; font-weight: bold;">🌻 致谢</a> •
    <a href="#📬联系方式" style="text-decoration: none; font-weight: bold;">📬 联系方式</a> •
    <a href="#📈星星历史" style="text-decoration: none; font-weight: bold;">📈 星星历史</a>
  </p>
</div>
```

</div>

## 🎉新闻

- **[2025/05/29]** 🎉 在[Huggingface每日论文](https://huggingface.co/papers?date=2025-05-29)上荣登当日**第1**名。
- **[2025/05/29]** 在arXiv上发布了我们的论文。请查看[这里](https://arxiv.org/pdf/2505.22617)。我们提供了关于LLM中RL熵机制的见解，并提出了两种简单而有效的策略来缓解熵崩溃。

## ✨开始使用

在准备好训练数据之后，对于在单个节点上训练 Qwen2.5-7B，以 KL-Cov 方法为例，您可以简单地运行：

```
cd verl
conda activate your_env
bash recipe/dapo/7b_kl_cov.sh
```

在多节点上训练 Qwen2.5-32B 模型时，您可以运行以下命令：

```
cd verl
conda activate your_env
bash recipe/dapo/32b_kl_cov.sh
```

## 📖介绍

```markdown
<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/e2a.jpg?raw=true" alt="问题" style="width: 96%; height: auto;">
</div>

本文讨论了在为大型语言模型（LLMs）扩展强化学习（RL）时出现的熵崩溃问题，即在训练过程中策略熵急剧下降，导致过度自信和性能饱和。我们通过实证建立了熵（$H$）与性能（$R$）之间的关系：$R=−aexp(H)+b$，表明性能受到熵耗尽的限制。

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/cov.jpg?raw=true" alt="问题" style="width: 96%; height: auto;">
</div>
```

从理论上讲，我们发现熵变化是由动作概率和logit更新之间的协方差驱动的，这与策略梯度方法中的优势相关。高概率、高优势的动作会减少熵，而罕见的、高优势的动作会增加熵。从经验上看，协方差项保持为正，解释了熵的单调下降。为了缓解这一问题，我们提出了Clip-Cov和KL-Cov，这些方法限制了高协方差标记的更新。这些方法有效地防止了熵的崩溃，并提高了性能。

## 📃评估

```markdown
<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/performance_fig.jpg?raw=true" alt="问题" style="width: 96%; height: auto;">
</div>
```

我们的方法能够在整个训练过程中保持相对较高的熵水平。例如，当基线的熵达到平稳状态且无法再增加时，KL-Cov方法仍然能够维持高出10倍以上的熵水平。同时，策略模型的响应长度稳步增加，其在测试集上的性能始终优于基线模型。这表明我们的模型能够在训练过程中更自由地探索，通过强化学习学习到更好的策略。
| **方法**           | **AIME24** | **AIME25** |  **AMC** | **MATH-500** | **OMNI-MATH** | **OlympiadBench** | **Minerva** | **平均** |
| ----------------- | ---------: | ---------: | -------: | -----------: | ------------: | ----------------: | ----------: | -------: |
| *Qwen2.5-7B*      |            |            |          |              |               |                   |             |          |
| GRPO              |       21.2 |        9.6 |     58.7 |         78.8 |          27.9 |              40.7 |        36.7 |     38.6 |
| 带Clip-higher      |       18.1 |       11.5 |     56.6 |         79.2 |          29.8 |              43.3 |        40.4 |     38.8 |
| 带**`CLIP-Cov`**   |       22.1 |   **15.8** |     58.2 |         80.4 |      **30.5** |          **44.1** |    **41.1** |     40.4 |
| 带**`KL-Cov`**     |   **22.6** |       12.9 | **61.4** |     **80.8** |          29.1 |              42.6 |        38.2 | **40.6** |
| *Qwen2.5-32B*     |            |            |          |              |               |                   |             |          |
| GRPO              |       21.8 |       16.2 |     69.7 |         84.2 |          35.2 |              43.6 |        45.5 |     45.8 |
| 带Clip-higher      |       35.6 |       22.3 |     69.5 |         77.2 |          35.1 |              42.5 |        43.0 |     47.2 |
| 带**`CLIP-Cov`**   |       32.3 |       22.7 |     67.2 |     **87.0** |      **42.0** |          **57.2** |        46.0 |     50.3 |
| 带**`KL-Cov`**     |   **36.8** |   **30.8** | **74.5** |         84.6 |          39.1 |              49.0 |    **46.3** | **52.2** |

## 🎈引用
如果您发现本论文或存储库对您有帮助，请引用我们。 

我们的两种方法在所有基准测试中都取得了非平凡的改进。与GRPO相比，我们的方法在7B模型上平均优于它2.0%，在32B模型上优于它6.4%。此外，我们观察到我们的方法在更大的Qwen2.5-32B上获得了更实质性的增益。具体来说，与GRPO相比，我们的方法在最具挑战性的基准测试AIME24和AIME25上分别实现了15.0%和14.6%的改进。

```bibtex
@article{cui2025entropy,
  title={The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models},
  author={Cui, Ganqu and Zhang, Yuchen and Chen, Jiacheng and Yuan, Lifan and Wang, Zhi and Zuo, Yuxin and Li, Haozhan and Fan, Yuchen and Chen, Huayu and Chen, Weize and others},
  journal={arXiv preprint arXiv:2505.22617},
  year={2025}
}
```
## 🌻Acknowledgement
We implement our reinforcement learning algorithm extending from [verl](https://github.com/volcengine/verl). We utilize [vLLM](https://github.com/vllm-project/vllm) for inference. Our models are trained primarily on [Qwen2.5 family](https://github.com/QwenLM/Qwen2.5). Our training data is built from [DAPO-MATH](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). Thanks for their great contributions!

## 📬 联系方式

如果您有任何问题、讨论或合作机会，欢迎联系：
- 崔淦渠(Ganqu Cui)：cuiganqu@pjlab.org.cn
- 张宇辰(Yuchen Zhang)：yuchen.zhang2003@gmail.com
- 陈家诚(Jiacheng Chen)：jackchan9345@gmail.com
- 丁宁(Ning Ding)：ningding.cs@gmail.com