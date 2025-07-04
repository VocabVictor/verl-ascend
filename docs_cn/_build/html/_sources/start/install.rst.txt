安装
====

需求
------------

- **Python**: 版本 >= 3.9
- **CUDA**: 版本 >= 12.1

verl 支持多种后端。目前可用的配置如下：

- **FSDP** 和 **Megatron-LM**（可选）用于训练。
- **SGLang**、**vLLM** 和 **TGI** 用于回滚生成。

后端引擎的选择
----------------------------

# 安装说明

## 训练 (Training)

我们建议使用 **FSDP** 后端来研究、探索和原型不同的模型、数据集和强化学习（RL）算法。有关使用 FSDP 后端的指南可以在 :doc:`FSDP Workers<../workers/fsdp_workers>` 中找到。

对于追求更好可扩展性的用户，我们建议使用 **Megatron-LM** 后端。目前，我们支持 `Megatron-LM v0.11 <https://github.com/NVIDIA/Megatron-LM/tree/v0.11.0>`_。有关使用 Megatron-LM 后端的指南可以在 :doc:`Megatron-LM Workers<../workers/megatron_workers>` 中找到。

2. 推理：

对于推理，vllm 0.8.3 及更高版本已测试稳定。我们建议开启环境变量 `VLLM_USE_V1=1` 以获得最佳性能。

有关 SGLang 的详细安装和使用说明，请参考 :doc:`SGLang Backend<../workers/sglang_worker>`。SGLang 的推出正在广泛开发中，并提供许多高级功能和优化。我们鼓励用户通过 `SGLang Issue Tracker <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/106>`_ 报告任何问题或提供反馈。

对于 huggingface TGI 集成，通常用于调试和单 GPU 探索。

从 Docker 镜像安装
-------------------------

我们提供预构建的 Docker 镜像以便快速设置。

对于与 Megatron 或 FSDP 结合使用的 vLLM，请使用稳定版本的镜像 ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3``，该镜像支持 DeepSeek-V3 671B 后训练。

对于最新的 vLLM 与 FSDP，请参考 ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``。

对于与 FSDP 结合使用的 SGLang，请使用 ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5``，该镜像由 SGLang RL 组提供。

查看 ``docker/`` 目录下的文件以获取基于 NGC 的镜像，或者如果您想构建自己的镜像。

1. 启动所需的 Docker 镜像并进入其中：

.. code:: bash

    docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag>
    docker start verl
    docker exec -it verl bash


2. 在容器内，安装最新的 verl：

.. code:: bash

    # 安装夜间版本（推荐）
    git clone https://github.com/volcengine/verl && cd verl
    # 选择您想要的推理引擎：vllm 或 sglang
    # pip3 install -e .[vllm]
    # pip3 install -e .[sglang]
    # 或者通过以下方式从 pypi 安装，而不是从 git：
    # pip3 install verl[vllm]
    # pip3 install verl[sglang]

.. 注意::

Docker 镜像 ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3`` 的构建配置如下：

    - **PyTorch**: 2.6.0+cu124
    - **CUDA**: 12.4
    - **cuDNN**: 9.8.0
    - **nvidia-cudnn-cu12**: 9.8.0.87，**对于使用 Megatron FusedAttention 和 MLA 支持非常重要**
    - **Flash Attention**: 2.7.4.post1
    - **Flash Infer**: 0.2.5
    - **vLLM**: 0.8.5
    - **SGLang**: 0.4.6.post5
    - **Megatron-LM**: core_v0.12.1
    - **TransformerEngine**: 2.3
    - **Ray**: 2.44.1

.. 注意::

对于具有 EFA 网络接口的 AWS 实例 (Sagemaker AI Pod)，  
您需要按照 ``docker/Dockerfile.awsefa`` 中所示安装 EFA 驱动程序。

从自定义环境安装
---------------------------------------------

我们建议使用 Docker 镜像以便于操作。然而，如果您的环境与 Docker 镜像不兼容，您也可以在 Python 环境中安装 verl。

前提条件
::::::::::::::

为了使训练和推理引擎能够更好地利用更快的硬件支持，需要安装CUDA/cuDNN及其他依赖项，并且在安装其他软件包时，有些依赖项容易被覆盖，因此我们将它们放在:ref:`安装后`步骤中。

我们需要安装以下先决条件：

- **CUDA**: 版本 >= 12.4
- **cuDNN**: 版本 >= 9.8.0
- **Apex**

建议使用版本高于12.4的CUDA作为Docker镜像，请参考`NVIDIA的官方网站 <https://developer.nvidia.com/cuda-toolkit-archive>`_以获取其他版本的CUDA。

.. code:: bash

    # 将目录更改为您喜欢的任何位置，不建议在源代码目录中
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda-toolkit-12-4
    update-alternatives --set cuda /usr/local/cuda-12.4

cuDNN 可以通过以下命令安装，其他版本的 cuDNN 请参考 `NVIDIA 官方网站 <https://developer.nvidia.com/rdp/cudnn-archive>`_。

.. code:: bash

    # 将目录更改为您喜欢的任何位置，不建议在源代码目录中
    wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn-cuda-12

Megatron-LM 和 FSDP 训练需要 NVIDIA Apex。
您可以通过以下命令安装，但请注意，这个步骤可能会花费很长时间。
建议设置 ``MAX_JOBS`` 环境变量以加速安装过程，但不要设置得太大，否则内存会过载，您的机器可能会卡死。

.. code:: bash

# 更改目录到您喜欢的任何位置，建议不要在 verl 源代码目录中
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


安装依赖
::::::::::::::::::::

.. 注意::

我们建议使用一个全新的 conda 环境来安装 verl 及其依赖项。

    **请注意，推理框架通常会严格限制您的 pytorch 版本，如果不够注意，会直接覆盖您已安装的 pytorch。**

    作为对策，建议首先安装推理框架，并使用它们所需的 pytorch。对于 vLLM，如果您希望使用现有的 pytorch，请遵循他们的官方说明
    `使用现有的 PyTorch 安装 <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source>`_ 。

1. 首先，为了管理环境，我们建议使用 conda：

.. code:: bash

   conda create -n verl python==3.10
   conda activate verl


2. 然后，执行我们在 verl 中提供的 ``install.sh`` 脚本：

.. code:: bash

    # 确保您已激活 verl conda 环境
    # 如果您需要使用 megatron
    bash scripts/install_vllm_sglang_mcore.sh
    # 或者如果您只需要使用 FSDP
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh


如果您在此步骤中遇到错误，请检查脚本并手动按照脚本中的步骤操作。

安装 verl
::::::::::::

要安装最新版本的 verl，最好的方法是从源代码克隆并安装它。然后您可以修改我们的代码以自定义您自己的后训练作业。

.. code:: bash

   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install --no-deps -e .

安装后
:::::::::::::::::

请确保在安装其他软件包时不要覆盖已安装的软件包。

值得检查的软件包包括：

- **torch** 及其系列
- **vLLM**
- **SGLang**
- **pyarrow**
- **tensordict**
- **nvidia-cudnn-cu12**：适用于 Magetron 后端

如果在运行 verl 时遇到软件包版本问题，请更新过时的软件包。

在 AMD GPU 上安装 - ROCM 内核支持
------------------------------------------------------------------

当您在 AMD GPU（MI300）上使用 ROCM 平台时，无法使用先前的快速入门来运行 verl。您应该按照以下步骤构建一个 Docker 并运行它。
如果在使用 AMD GPU 运行 verl 时遇到任何问题，请随时联系我 - `Yusheng Su <https://yushengsu-thu.github.io/>`_。

查找用于 AMD ROCm 的 Docker 文件：`docker/Dockerfile.rocm <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.rocm>`_

.. code-block:: bash

# 在仓库目录中构建 Docker：
    # docker build -f docker/Dockerfile.rocm -t verl-rocm:03.04.2015 .
    # docker images # 你可以找到你构建的 Docker
    FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

    # 设置工作目录
    # WORKDIR $PWD/app

    # 设置环境变量
    ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

    # 安装 vllm
    RUN pip uninstall -y vllm && \
        rm -rf vllm && \
        git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
        cd vllm && \
        MAX_JOBS=$(nproc) python3 setup.py install && \
        cd .. && \
        rm -rf vllm

    # 复制整个项目目录
    COPY . .

    # 安装依赖
    RUN pip install "tensordict<0.6" --no-deps && \
        pip install accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        liger-kernel \
        numpy \
        pandas \
        datasets \
        peft \
        "pyarrow>=15.0.0" \
        pylatexenc \
        "ray[data,train,tune,serve]" \
        torchdata \
        transformers \
        wandb \
        orjson \
        pybind11 && \
        pip install -e . --no-deps

构建镜像
::::::::::::::::::::::::

.. code-block:: bash

```bash
docker build -t verl-rocm .
```

启动容器
::::::::::::::::::::::::::::

.. code-block:: bash

```bash
docker run --rm -it \
      --device /dev/dri \
      --device /dev/kfd \
      -p 8265:8265 \
      --group-add video \
      --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --privileged \
      -v $HOME/.ssh:/root/.ssh \
      -v $HOME:$HOME \
      --shm-size 128G \
      -w $PWD \
      verl-rocm \
      /bin/bash
```

如果您不想使用根模式并希望以用户身份运行，请在上述 Docker 启动脚本中添加 ``-e HOST_UID=$(id -u)`` 和 ``-e HOST_GID=$(id -g)``。

目前，verl 与 AMD GPU 一起支持 FSDP 作为训练引擎，vLLM 和 SGLang 作为推理引擎。我们将在未来支持 Megatron。