多节点训练
==================

最后更新：2025年6月10日。

.. _wuxibin89: https://github.com/wuxibin89

作者： `Xibin Wu <https://github.com/wuxibin89>`_, `Yusheng Su <https://yushengsu-thu.github.io/>`_.

手册
------

设置多节点 Ray 集群
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 使用 ``ray start --head --dashboard-host=0.0.0.0`` 启动主节点，您需要关注两个地址：

- GCS 地址: ``ray start --address=<address>``, 工作节点应连接到此地址。
- 仪表板地址: ``<address>:8265``, 您应向该地址提交作业到集群。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/head.png?raw=true

2. 使用 ``ray start --address=<address>`` 启动工作节点，您将获得上述内容。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/worker.png?raw=true

3. 现在您应该可以通过 ``ray status`` 查看集群中有 2 个节点。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/status.png?raw=true

4. 此外，您可以通过上面获得的地址在浏览器中访问仪表板。

*可能需要配置防火墙规则以访问仪表板，如果遇到任何问题，请联系您的网络管理员。*

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/overview.png?raw=true

提交作业到 Ray 集群
~~~~~~~~~~~~~~~~~~~~~~~~~
1. 使用您上面获得的仪表板地址将 Ray 作业提交到集群。

.. code-block:: bash

```bash
ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        --no-wait \
        -- \
        python3 -m verl.trainer.main_ppo \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        ...
```

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/submit.png?raw=true

2. 然后您可以使用以下命令检查作业状态：

- ray job list：列出提交到集群的所有作业。
- ray job logs <Submission ID>：查询作业的日志。
- ray job status <Submission ID>：查询作业的状态。
- ray job stop <Submission ID>：请求停止该作业。

3. 您还可以在 ``/tmp/ray/session_latest/logs/`` 中访问驱动程序/任务/演员的日志，驱动程序日志为 ``job-driver-raysubmit_<Submission ID>.log``。

4. 我们强烈建议您在多节点训练中通过仪表板查看作业详情，因为这提供了一种更结构化的方式来查看作业信息。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job.png?raw=true
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job_detail.png?raw=true

Slurm
-----
待定

dstack
------
`dstackai/dstack <https://github.com/dstackai/dstack>`_ 是一个开源的容器编排工具，简化了跨云服务提供商和本地环境的分布式训练，无需使用 K8S 或 Slurm。

先决条件
~~~~~~~~~~~~
一旦 dstack 被 `安装 <https://dstack.ai/docs/installation>`_，使用 ``dstack init`` 将目录初始化为一个仓库。

.. code-block:: bash

```bash
mkdir myproject && cd myproject
    dstack init

**创建一个舰队**

在提交分布式训练任务之前，创建一个 `dstack` `舰队 <https://dstack.ai/docs/concepts/fleets>`_。

运行 Ray 集群任务
~~~~~~~~~~~~~~~~~~~~~~

一旦舰队创建完成，定义一个 Ray 集群任务，例如在 ``ray-cluster.dstack.yml`` 中：
```

.. code-block:: yaml

```yaml
type: task
    name: ray-verl-cluster
```

nodes: 2

```yaml
env:
        - WANDB_API_KEY
        - PYTHONUNBUFFERED=1
        - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    image: whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.2
    commands:
        - git clone https://github.com/volcengine/verl
        - cd verl
        - pip install --no-deps -e .
        - pip install hf_transfer hf_xet
        - |
        if [ $DSTACK_NODE_RANK = 0 ]; then
            python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
            python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct')" 
            ray start --head --port=6379;
        else
            ray start --address=$DSTACK_MASTER_NODE_IP:6379
        fi

    # 暴露 Ray 仪表板端口
    ports:
        - 8265

    resources:
        gpu: 80GB:8
        shm_size: 128GB

    # 在实例上保存检查点
    volumes:
        - /checkpoints:/checkpoints

现在，如果您通过 `dstack apply` 运行此任务，它将自动将 Ray 的仪表板端口转发到 `localhost:8265`。
```

.. code-block:: bash

dstack apply -f ray-cluster.dstack.yml

只要 `dstack apply` 处于连接状态，您就可以使用 `localhost:8265` 提交 Ray 作业进行执行。

提交 Ray 作业
~~~~~~~~~~~~~~~

在您可以提交 Ray 作业之前，请确保在本地安装 `ray`：
   
.. code-block:: shell

    pip install ray

现在您可以将训练作业提交到可用的 Ray 集群，地址为 ``localhost:8265``：
   
.. code-block:: shell

```bash
$ RAY_ADDRESS=http://localhost:8265
    $ ray job submit \
        -- python3 -m verl.trainer.main_ppo \
        data.train_files=/root/data/gsm8k/train.parquet \
        data.val_files=/root/data/gsm8k/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2.5-7B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.project_name=ppo_training \
        trainer.experiment_name=qwen-2.5-7B \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.default_local_dir=/checkpoints \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log \
        trainer.resume_mode=disable
```

有关 `dstack` 工作原理的更多细节，请查看其 `文档 <https://dstack.ai/docs>`_。

如何进行调试？
---------------------

Ray 分布式调试器 VSCode 扩展（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 从 Ray 2.39 开始，Anyscale 引入了 `Ray 分布式调试器 <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode 扩展。请按照扩展的安装说明进行操作，然后使用您之前获得的仪表板 URL 添加您的集群。

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray 分布式调试器 VSCode 扩展截图

2. 前提条件。

确保安装以下内容（有关更多详细信息，请参阅扩展的 README）：

   - Visual Studio Code  
   - `ray[default]` >= 2.9.1  
   - `debugpy` >= 1.8.0  

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/c7098b755ff689859837773a916c857.png?raw=true
      :alt: 带有 Ray 前提条件的 VSCode

3. 环境变量。

   要启用事后调试，请设置：

   .. code-block:: bash

```
      export RAY_DEBUG_POST_MORTEM=1
```

.. admonition:: 注意
      :class: important

      在启动 Ray 之前，请确保移除任何遗留标志：

      - `RAY_DEBUG=legacy`  
      - `--ray-debugger-external`

4. 配置断点 在您的代码中设置 `breakpoint()`，并提交作业到集群。然后扩展将显示断点信息。

   1. 在您的远程函数中插入 `breakpoint()` 调用。  
   2. 将您的作业提交到集群。  

   扩展将检测活动断点并在 VSCode 中显示它们。

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: 在 VSCode 中检测到的断点

   **注意：** 断点仅在使用 `@ray.remote` 装饰的函数内部支持。

5. 启动调试器。

   直接从命令行运行您的作业（不要使用 `launch.json`）：

   .. code-block:: bash

```plaintext
      python job.py
```

6. 附加到断点。

一旦进程命中第一个 `breakpoint()`，请点击 VSCode 侧边栏中的 Ray 分布式调试器图标以附加调试器。

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: 将 VSCode 调试器附加到 Ray 进程

7. 使用多个 breakpoint() 进行调试。

对于每个后续任务，首先断开当前的调试会话，然后再次点击扩展图标以附加到下一个断点。

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/6e83c910a62c82fecb89c6619e001cd.png?raw=true
      :alt: 断开并重新连接调试器

遗留 Ray 调试器
~~~~~~~~~~~~~~~~~~~
1. Ray 具有内置的遗留 `debugger <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`_，允许您调试分布式应用程序。要启用调试器，请使用 ``RAY_DEBUG=legacy`` 和 ``--ray-debugger-external`` 启动 Ray 集群。

.. code-block:: bash

# 启动主节点
    RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
    # 启动工作节点
    RAY_DEBUG=legacy ray start --address='10.124.46.192:6379' --ray-debugger-external

2. 在你的代码中设置断点，并提交作业到集群。然后运行 ``ray debug`` 等待断点：

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/legacy.png?raw=true

多节点训练在AMD集群上
---------------------------------------------------------------------------------------

如果您想在AMD集群上使用slurm进行Docker/Podman容器的多节点训练，可以使用以下脚本。

如果在使用AMD GPU运行verl时遇到任何问题，请联系 `Yusheng Su <https://yushengsu-thu.github.io/>`_。

.. note::
    1. 您需要在以下脚本中使用``podman``或``docker``。我们将稍后发布apptainer脚本。
    2. 如果您想使用``podman``，只需在以下脚本中将``docker``替换为``podman``。

该脚本包括以下步骤：

1. SLURM配置
2. 环境设置
3. Docker/Podman容器设置
4. Ray集群初始化
5. 数据预处理
6. 模型设置
7. 启动训练

slurm_script.sh
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    #!/bin/bash

```bash
#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=200G
#SBATCH --time=30-00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=28
#SBATCH --output=../verl_log/slurm-%j.out
#SBATCH --error=../verl_log/slurm-%j.err
#SBATCH --nodelist=gpu-[0,1]


# 加载必要的模块
### 运行此设置
# [集群]: 使用docker
# docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4


##########################################################################
###以下设置应在不同的项目和集群中进行设置###
##########################################################################

### 项目
CONTAINER_NAME="multinode_verl_training"
IMG="verl.rocm"
DOCKERFILE="docker/Dockerfile.rocm"
# echo $PWD
verl_workdir="${HOME}/projects/verl_upstream"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
export HF_HOME=$TRANSFORMERS_CACHE
```

### 集群网络设置
    export NCCL_DEBUG=TRACE
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    # export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    export HSA_NO_SCRATCH_RECLAIM=1
    ##########################################################################

    ### 对于 ROCm 和训练脚本
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


    # 构建并启动 Docker 容器
    srun bash -c "
        # 在任何错误时退出
        set -e 

        # 清理悬空镜像（标签为 <none> 的镜像）
        docker image prune -f

# 需要先拉取 Docker 镜像
        docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
        
        if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMG}"; then
            echo \"正在构建 ${IMG} 镜像...\"
            docker build -f \"${DOCKERFILE}\" -t \"${IMG}\" .
        else
            echo \"${IMG} 镜像已存在，跳过构建\"
        fi

        # 如果存在则移除旧容器
        docker rm \"${CONTAINER_NAME}\" 2>/dev/null || true

        # 检查网络设备
        ibdev2netdev

# 启动 Docker
        docker run --rm -d \
        -e HYDRA_FULL_ERROR=1 \
        -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
        -e ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        -e NCCL_DEBUG=${NCCL_DEBUG} \
        -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES} \
        -e TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY} \
        -e NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
        -e NCCL_IB_HCA=${NCCL_IB_HCA} \
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
        -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
        -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
        -e NCCL_PROTO=${NCCL_PROTO} \
        -e RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE} \
        -e TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
        -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM} \
        -e TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
        -e HF_HOME=${HF_HOME} \
        --network host \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        -v \${HOME}:\${HOME} \
        -v \${HOME}/.ssh:/root/.ssh \
        -w "${verl_workdir}" \
        --shm-size 128G \
        --name \"${CONTAINER_NAME}\" \
        \"${IMG}\" \
        tail -f /dev/null

```bash
echo \"容器设置完成\"
    "
        # （可选）：如果您不想使用根模式并希望将自己指定为用户
        # 请在上述 Docker 启动脚本中添加 `-e HOST_UID=$(id -u)` 和 `-e HOST_GID=$(id -g)`。
```

### 在训练之前启动 Ray 节点

    # 获取节点名称
    nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # 如果我们在主节点 IP 中检测到空格字符，我们将
    # 将其转换为 ipv4 地址。此步骤是可选的。
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
        echo "检测到 IPV6 地址。我们将 IPV4 地址分割为 $head_node_ip"
    fi

    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP 头: $ip_head"

    # 确保在 Ray 初始化之前设置环境变量

    # 打印所有环境变量
    printenv

```bash
echo "正在启动 HEAD 节点 $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
            ray start --head --node-ip-address="$head_node_ip" --port=$port \
            --dashboard-port=8266 \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    # 可选，尽管在某些版本的 Ray < 1.0 中可能有用。
    sleep 10

    # 除了头节点的节点数量
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "调试: 在 node_i = ${node_i} 上启动工作节点"
        if [ -z "$node_i" ]; then
            echo "错误: 工作节点 $i 的节点名称为空"
            continue
        fi
        echo "在 $node_i 上启动工作节点 $i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            docker exec "${CONTAINER_NAME}" \
                ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
        sleep 5
    done
```

# Ray 初始化测试 (查看上述执行是否有错误)
    echo "正在测试 Slurm 节点中的 Ray 初始化..."
    docker exec "${CONTAINER_NAME}" python3 -c '
    import ray
    try:
        ray.init(address="auto")
        print("\n=== Ray 集群状态 ===")
        print(f"节点数量: {len(ray.nodes())}")
        for node in ray.nodes():
            print("节点: {}, 状态: {}".format(node["NodeManagerHostname"], node["Alive"]))
            # print(f"节点: {node}")
        ray.shutdown()
        print("Ray 初始化成功！")
    except Exception as e:
        print(f"Ray 初始化失败: {str(e)}")
    '
    echo "=== Ray 测试完成 ==="
    ######

# 运行数据预处理

    echo "开始数据预处理..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/gsm8k.py" "--local_dir" "../data/gsm8k"

    echo "开始数据预处理..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/math_dataset.py" "--local_dir" "../data/math"

```plaintext
    train_files="../data/gsm8k/train.parquet"
    val_files="../data/gsm8k/test.parquet"
```

# 下载并测试模型
    echo "正在加载模型..."
    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"

    # 在管道测试后设置模型路径
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

    echo "== 数据和模型加载完成 =="

    echo "开始训练..."

    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"

```bash
PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
        python3 -m verl.trainer.main_ppo \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=$MODEL_PATH \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=8 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.0001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_example' \
        trainer.experiment_name='Qwen2.5-32B-Instruct_function_rm' \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.val_before_train=False \
        trainer.nnodes=${SLURM_NNODES} \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15
```

运行多节点训练，使用上述 slurm_script.sh
~~~~~~~~~~~~~~~~~~~~
只需提交你的 slurm_script.sh 文件。

.. code-block:: bash

```plaintext
sbatch slurm_script.sh
```