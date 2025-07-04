# 配方：自我游戏偏好优化 (SPPO)

最后更新：2025年5月28日。

verl 提供了论文 [自我游戏偏好优化用于语言模型对齐](https://arxiv.org/abs/2405.00675) 的社区配方实现。SPPO 可以显著提升大型语言模型 (LLM) 的性能，而无需强大的外部信号，例如来自 GPT-4 的响应或偏好。它的表现优于采用迭代直接偏好优化 (DPO) 等方法训练的模型。SPPO 在理论上是有依据的，确保 LLM 能够在一般的、可能不传递的偏好下收敛到冯·诺依曼赢家（即纳什均衡），并通过对多个数据集的广泛评估进行了实证验证。

论文作者：[吴越](https://yuewu.us/)\*、[孙志清](https://www.cs.cmu.edu/~zhiqings/)\*、[袁会卓](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*、[季凯轩](https://scholar.google.com/citations?user=FOoKDukAAAAJ)、[杨易铭](https://www.cs.cmu.edu/~yiming/)、[顾全全](https://web.cs.ucla.edu/~qgu/)

verl 实现作者：[杨宇浩](https://github.com/yhyang201)、[赵晨阳](https://github.com/zhaochenyang20)

[[网页](https://uclaml.github.io/SPPO/)] [[Huggingface](https://huggingface.co/papers/2405.00675)] [[论文](https://arxiv.org/abs/2405.00675)] [[原始实现](https://github.com/uclaml/SPPO)]

## 重现实验

我们在 MATH 数据集上评估 SPPO 的性能。从 Qwen2.5-7B-Instruct 的初始得分 46.6 开始，经过 20 个训练周期后，我们达到了 65.6 的得分，使我们的模型大约位于 [MATH 排行榜](https://paperswithcode.com/sota/math-word-problem-solving-on-math) 的前 20 名。需要注意的是，verl 的内部评估指标可能与 Qwen2.5-7B-Instruct 的官方评估方法并不完全一致。因此，为了保持一致性和公平比较，我们仅报告基于 verl 评估框架的结果。

```
git clone git@github.com:volcengine/verl.git
cd verl
python3 -m uv pip install -e ".[sglang]"

```bash
export WANDB_API_KEY=<您的_WANDB_API_KEY>
```

```markdown
python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3
bash recipe/sppo/run_qwen2.5-7b_rm.sh
```

请注意，安装有时会失败，无法安装flash-attn。如果发生这种情况，您可以通过运行以下命令手动安装：
```

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

## 致谢

我们衷心感谢以下人士的贡献和指导：

- [Yue Wu](https://yuewu.us/)
- [Chendong Wang](https://cdwang96.github.io/)
- [Yifan Zhang](https://github.com/yifanzhang-pro)
- [Yongan Xiang](https://github.com/BearBiscuit05)
- [Junrong Lin](https://github.com/ocss884)
- [Yuxuan Tong](https://github.com/tongyx361)
- [Guangming Shen](https://github.com/PeterSH6)
- [Biao He](https://www.linkedin.com/in/biao-he/)
- [Qingquan Song](https://qingquansong.github.io/)
- [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)