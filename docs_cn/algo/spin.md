# 食谱：自我对弈微调 (SPIN)

最后更新：2025年5月31日。

`verl` 提供了一种灵感来源于论文 **"自我对弈微调将弱语言模型转变为强语言模型"** (SPIN) 的食谱。SPIN 是一种语言模型微调算法，通过受博弈论启发的自我对弈机制实现迭代自我改进。

**核心思想：** 模型通过与自身对弈进行学习，减少对外部偏好数据集或更强教师模型的依赖：

1.  **合成数据生成：** 当前模型生成响应，从之前的迭代中创建自己的训练数据。
2.  **双人游戏设置：** 一个涉及两个玩家的游戏，由单个 LLM 执行。
3.  **迭代训练：** 模型通过不断优化其策略逐步改进，每次迭代的模型成为下一次迭代的对手。

论文作者：[Zixiang Chen](https://github.com/uclaml/SPIN)\*、[Yihe Deng](https://github.com/uclaml/SPIN)\*、[Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*、[Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ)、[Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

[[网页](https://uclaml.github.io/SPIN/)] [[Huggingface](https://huggingface.co/papers/2401.01335)] [[论文](https://arxiv.org/abs/2401.01335)] [[原始实现](https://github.com/uclaml/SPIN)]

verl 实现作者: [Chendong Wang](https://cdwang96.github.io/), [Chenyang Zhao](https://github.com/zhaochenyang20)

---

## 关键功能 (compute_online_dpo_loss) 和相关工作
SPIN (Chen et al., 2024) 提出了一个迭代自我博弈机制来微调语言模型。在每次迭代中，SPIN 的训练目标在使用逻辑损失函数时，相当于直接偏好优化 (Direct Preference Optimization, DPO) 损失 (Rafailov et al., 2023)。

这个 `verl` 配方通过迭代使用 DPO 损失实现了 SPIN 的核心概念 (Xu et al., 2023; Xiong et al., 2023; Snorkel AI, 2024)。这意味着在每次迭代中，我们使用 DPO 损失进行偏好优化来微调 LLM。值得注意的是，Xu et al. (2023) 探索了使用成对的 cringe 损失进行迭代偏好优化，而 Xiong et al. (2023) 讨论了如何在 KL 约束下通过迭代训练将理论与实践结合起来。迭代偏好学习的概念也在在线 DPO (Guo et al., 2024) 中得到了探讨，该方法专注于来自在线 AI 反馈的直接对齐。在在线 DPO 中，偏好数据在训练过程中动态更新，使模型能够从自身生成的数据中学习。

具体来说，我们开发了 **`compute_online_dpo_loss`** 函数，并在此基础上构建了这个 SPIN 配方。通过结合在线偏好生成，这种方法使得语言模型能够不断精炼，而无需依赖固定的外部偏好数据集。

**参考文献：**
* [自我对弈微调将弱语言模型转变为强语言模型](https://arxiv.org/abs/2401.01335) (Chen et al., 2024) 
* [直接偏好优化：你的语言模型实际上是一个奖励模型](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) 
* [有些事情比其他事情更让人 cringe：使用成对 cringe 损失进行偏好优化](https://arxiv.org/abs/2312.16682) (Xu et al., 2023) 
* [从人类反馈中进行迭代偏好学习：在 KL 约束下将理论与实践相结合的 rlhf](https://arxiv.org/abs/2312.11456) (Xiong et al., 2023)
* [Snorkel-Mistral-PairRM-DPO](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO) (Snorkel AI, 2024)
* [来自在线 AI 反馈的直接语言模型对齐](https://arxiv.org/abs/2402.04792) (Guo et al., 2024)

## 我们的在线 DPO 实现

我们的 `compute_online_dpo_loss` 函数适配了 `verl` 的现有 PPO 基础设施（基于 `verl` v0.3.0.post1），用于这一迭代在线 DPO。我们实现的关键方面包括：

* **无评论者（No Critic）：** 与 PPO 不同，我们省略了价值函数评论者。
* **动态参考模型（Dynamic Reference Model）：** 使用显式参考策略（`ref_policy_wg`）来计算 DPO 损失。该参考模型的权重可以根据演员（`ref_update_freq`）定期更新，从而提供动态基线。
* **在线偏好生成（Online Preference Generation）：** `compute_onlineDPO_pref` 函数（在 `core_algos.py` 中）根据奖励来源动态创建选择/拒绝对（例如，基于规则的数学问题排名）。
* **DPO 损失集成（DPO Loss Integration）：** 我们在演员更新（`dp_actor.py`）中用我们的 `compute_online_dpo_loss`（在 `core_algos.py` 中）替换了 PPO 的策略损失，直接利用生成的偏好优化策略。
* **迭代训练协调（Iterative Training Orchestration）：** `SpinTrainer`（在 `spin_trainer.py` 中）管理整个自我对弈循环：生成、偏好标记、可选的参考模型更新和策略更新，使得持续自我改进与 SPIN 的原则保持一致。

---
## 算法

该配方实现了一种适应于 `verl` 强化学习框架的在线算法，为微调语言模型提供了一个替代 PPO 的方案。

**在线循环：** 与 PPO 中最大化标量奖励信号不同，这种方法直接优化策略模型，以使其与在训练过程中生成的偏好数据 *在线* 对齐：

1.  **生成：** 当前模型为批次中的每个提示生成多个响应。
2.  **偏好标记：** 一个函数评估这些生成的响应，以确定哪个是偏好的（选择的）以及哪个是不偏好的（拒绝的）。这可以通过奖励函数或基于特定规则的隐式排名来完成。（在这个配方中，我们使用基于规则的数学问题排名）。
3.  **更新：** 这个偏好元组（`prompt`，`chosen_response`，`rejected_response`）用于通过 `compute_online_dpo_loss` 更新演员模型，并与参考模型进行比较。

**与SPIN的连接：**  
在步骤2中，在线生成循环将通过使用某种偏好标记方法（基于规则的数学问题排名，通过选择更好的选项）动态改变目标数据分布，而不仅仅是使用固定的目标数据分布。这探索了SPIN论文第7节中提到的“动态改变目标数据分布”的方向，旨在潜在地提升大型语言模型（LLM）的性能，超越固定的人类标注数据的上限。

---

## 重现实验 (示例设置)

以下步骤概述了如何设置环境并运行 SPIN 配方，基于提供的测试日志使用 GSM8K 和 Qwen2.5-3B-Instruct。

1.  **设置环境 (使用 Docker 的示例)：**
    ```bash
    # 启动一个具有 GPU 访问权限和共享内存的容器
    docker run -it --name spin_test --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v /path/to/host/.cache:/root/.cache \
        -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
        lmsysorg/sglang:latest \
        /bin/bash

    # 在容器内或您的主机上：
    # 确保 /tmp 可写
    mkdir -p /tmp
    chmod 1777 /tmp

    # 安装 Python 3.10（如果尚未安装）和 venv
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv tmux
    python3 -m ensurepip --upgrade

    # 创建并激活虚拟环境
    python3 -m venv ~/.python/spin_env
    source ~/.python/spin_env/bin/activate

    # 安装 uv（快速包安装器）
    python3 -m pip install uv
    ```

2.  **安装 verl 及其依赖项：**
    ```bash
    # 克隆 verl 仓库并切换到 spin 分支
    cd ~
    git clone git@github.com:volcengine/verl.git && cd verl

    # 安装 flash-attn（处理潜在的构建问题）
    python3 -m uv pip install wheel packaging
    python3 -m uv pip install flash-attn --no-build-isolation --no-deps

    # 安装带有 sglang 附加功能的 verl
    python3 -m uv pip install -e ".[sglang]"
    ```
    *注意：如果 `flash-attn` 安装失败，请再次尝试手动步骤或查阅其文档。*

3.  **登录并下载数据/模型：**
    ```bash
    # 登录 Weights & Biases（可选，用于记录）
    export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
    # wandb login

    # 下载 GSM8K 数据集
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k # 调整后的路径

    # 下载基础模型（示例：Qwen2.5-3B-Instruct）
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct
    ```

4.  **配置:**
    * 修改配置文件（例如，`config/spin_trainer.yaml` 或运行脚本中指定的文件），确保路径正确指向您下载的模型、数据、所需的超参数（`dpo_beta`、学习率等）以及分布式训练设置（节点、每个节点的GPU数量）。
    * 注意 `actor_rollout_ref.model_path`、`data` 路径、`reward_model` 配置（如果使用的话）和 `trainer.ref_update_freq`。

5.  **运行训练:**
    ```bash
    # 设置可见的CUDA设备（根据您的硬件和配置进行调整）
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # 启动训练脚本（例如，test.sh 或自定义脚本）
    # 确保 test.sh 指向正确的配置和主脚本
    bash recipe/spin/run_spin.sh
    ```

---

## 配置

* 主要配置通常通过在启动脚本中指定的 YAML 文件进行管理（例如，`config/spin_trainer.yaml`）。
* 关键配置部分：
    * `data`: 训练/验证提示文件的路径、批处理大小、序列长度。
    * `actor_rollout_ref`: 基础模型的路径（用于演员和初始参考）、FSDP 设置、优化参数（学习率、调度器）。
    * `reward_model`: 用于在线偏好标注的奖励模型配置（路径、批处理大小等）。如果使用更简单的奖励函数，可以省略。
    * `algorithm`: DPO 特定的超参数，如 `dpo_beta`、`dpo_loss_type`。
    * `trainer`: 分布式训练设置（节点、每个节点的 GPU）、日志记录（WandB）、检查点频率，以及 `ref_update_freq`（设置为 > 0 以启用来自演员的周期性参考模型更新）。

---

## 关键文件

* `main_spin.py`：使用 Hydra 加载配置并启动 `SpinTrainer` 的主要入口点。
* `spin_trainer.py`：定义 `SpinTrainer` 类，协调在线 DPO (在线分布式策略优化) 训练循环。
* `fsdp_workers.py`：实现 Ray 工作节点（Actor，引用），可能使用 FSDP (全局分布式数据并行)。
* `dp_actor.py`：包含 Actor 类，包括 DPO 策略更新逻辑。
* `core_algos.py`：包括 `compute_online_dpo_loss` 和 `compute_onlineDPO_pref` 的辅助函数。
* `config/spin_trainer.yaml`（或类似文件）：食谱的主要 Hydra 配置文件。
* `run_spin.sh`（或类似文件）：用于启动训练运行的示例 bash 脚本。
* `README.md`：本文件。

---

## 致谢

我们衷心感谢`verl`社区及顾问的贡献和指导，包括（改编自SPPO）：

* [Zixiang Chen](https://sites.google.com/view/zxchen)
* [Yuhao Yang](https://github.com/yhyang201)
* [Yifan Zhang](https://github.com/yifanzhang-pro)
* [Yongan Xiang](https://github.com/BearBiscuit05)
* [Junrong Lin](https://github.com/ocss884)
* [Yuxuan Tong](https://github.com/tongyx361)
* [Guangming Shen](https://github.com/PeterSH6)
* [Biao He](https://www.linkedin.com/in/biao-he/)
* [Qingquan Song](https://qingquansong.github.io/)
* [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/)
* [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)