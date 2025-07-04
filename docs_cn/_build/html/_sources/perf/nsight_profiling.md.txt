# NVIDIA Nsight Systems 在 verl 中的性能分析

最后更新：2025年6月20日。

本指南解释了如何使用 NVIDIA Nsight Systems 对 verl 训练运行进行性能分析。

## 配置

在 verl 中的性能分析可以通过训练器配置文件（ppo_trainer.yaml 或其他文件如 dapo_trainer.yaml）中的多个参数进行配置：

### 先决条件

Nsight Systems 版本很重要，请参考 `docker/Dockerfile.vllm.sglang.megatron` 以获取我们使用的版本。

### 全局分析控制

verl 有一个单一的控制进程和多个工作进程。控制进程和工作进程都可以被分析。由于控制进程可以在集群中的任何节点上执行，因此在日志中会打印一条消息，以指示控制进程的节点主机名和进程 ID。

在 `trainer` 中，三个新的配置项控制分析器的行为：

* **`trainer.profile_steps`**。在此列表中指定应进行分析的步骤编号。例如：[1, 2, 5] 将分析步骤 1、2 和 5。而 ``null`` 表示不进行分析。

* **`controller_nsight_options`**。此配置组用于单个控制器。此配置组中的所有字段将在 Ray 启动控制进程时发送到 Nsight Systems。`ppo_trainer.yaml` 提供了一个可行的示例。用户可以参考 [Nsight Systems 手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 和 [Ray 用户指南](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html) 以获取更多详细信息。

* **`worker_nsight_options`**。该配置组用于工作进程。类似地，当 Ray 启动控制器进程时，此配置组中的所有字段将被发送到 Nsight Systems。捕获范围用于控制分析器何时开始和停止。因此 `capture-range: "cudaProfilerApi"` 是固定的，不会更改。用户可以通过一些精确的计算来更改 `capture-range-end`，或者将其保持为 `null`。

### 工作进程分析

Verl 管理多个强化学习角色，_Actor_、_Ref_、_Rollout_、_Critic_、_Reward_，这些角色在不同的工作类中实现。这些工作进程可以组合成一个 Ray Actor，运行在一个进程组中。每个强化学习角色都有自己的分析配置组 `profiler`，该组由三个字段组成：

* **`all_ranks` 和 `ranks`**。当 `all_ranks` 设置为 `True` 时，将对所有排名进行分析；当设置为 `False` 时，将对 `ranks` 进行分析。默认情况下，verl 在每个进程排名中将整个训练过程分析为一个单独的 `worker_process_<PID>.<step>.nsys-rep` 文件。请注意，`<step>` 是从 `1` 开始连续计数的，而不是 `trainer.profile_steps` 本身。

* **`discrete`**. 当设置为 `False` 时，所有角色在一个训练步骤中的操作将被转储到一个数据库中。当设置为 `True` 时，通过 `DistProfiler.annotate` 注释的操作将被转储到一个离散数据库中。在这种情况下，每个角色的操作占用一个 `<step>`。

* **`actor_rollout_ref`**. 此 Worker 可以配置为最多包含 3 个角色并一起执行。最终的 `profiler` 配置是这三个角色配置的联合。

* **Verl 组合模式**. Verl 可以将两个 Worker 子类组合为一个 Worker Actor。在这种情况下，用户应确保组合的 Workers 具有一致的 `discrete`。Nsight Systems profiler 无论如何都会使用 `torch.cuda.profiler.start()` 和 `stop()` 配对来转储一个 `<step>` 数据库。

### 在哪里找到分析数据

默认情况下，`*.nsys-rep` 文件保存在每个节点的目录 `/tmp/ray/session_latest/logs/nsight/` 中。根据 Ray 手册，默认目录是不可更改的。["然而，Ray 保留了默认配置的 `--output` 选项"](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html)。

一些用户可能会觉得这不方便，但可以理解的是，Ray 可能会启动数百个进程，如果我们将文件保存在一个中央位置，将会对网络文件系统造成很大的压力。

## 使用示例

要为特定组件和步骤启用性能分析，请像这样修改您的 `ppo_trainer.yaml`：

### 禁用性能分析器
```yaml
    trainer:
        profile_steps: null # 禁用性能分析
```

### 启用性能分析器并为一个训练步骤设置一个数据库
```yaml
    trainer:
        profile_steps: [1, 2, 5]
    actor_rollout_ref:
        actor:
            profiler:
                discrete: False
                all_ranks: False
                ranks: [0, 1]
        rollout:
            profiler:
                discrete: False
                all_ranks: False
                ranks: [0, 1]
        ref:
            profiler:
                discrete: False
                all_ranks: False
                ranks: [0, 1]
    critic:
        profiler:
            discrete: False
            all_ranks: False
            ranks: [0, 1]
```

### 启用分析器和多个数据库以进行一次训练步骤
```yaml
    trainer:
        profile_steps: [1, 2, 5]
    actor_rollout_ref:
        actor:
            profiler:
                discrete: True
                all_ranks: False
                ranks: [0, 1]
        rollout:
            profiler:
                discrete: True
                all_ranks: False
                ranks: [0, 1]
        ref:
            profiler:
                discrete: True
                all_ranks: False
                ranks: [0, 1]
    critic:
        profiler:
            discrete: True
            all_ranks: False
            ranks: [0, 1]
```

## 分析输出

当启用分析时，verl 将为指定的组件和步骤生成 Nsight Systems 分析文件。分析文件将包括：

- CUDA 内核执行
- 内存操作
- CPU-GPU 同步
- 关键操作的 NVTX 标记

Nsight Systems 支持多报告视图，可以同时打开多个数据库。在此模式下，不同的进程和步骤可以在同一时间轴上对齐，以便进行更好的分析。