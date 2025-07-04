=======================
搜索工具集成
=======================

最后更新：2025年5月30日。

介绍
------------
- 我们在多轮强化学习（Multi-Turn RL）中添加了一个搜索工具调用功能，使模型能够在演员（Actor）执行过程中发起检索请求，并直接使用检索结果进行训练。**我们支持使用本地密集检索器（local dense retriever）作为检索工具，以及与您自己的本地检索引擎集成。**

快速重现
------------------

创建一个新的 Docker 容器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   docker run \
       -it \
       --shm-size 32g \
       --gpus all \
       -v {Huggingface-Cache-Path}:/root/.cache \
       --ipc=host \
       --network=host \
       --privileged \
       --name sglang_{your-name} \
       lmsysorg/sglang:dev \
       /bin/zsh

如果您在退出容器后需要重新启动：

.. code:: bash

   docker start -i sglang_{your-name}

更新 Python 并使用 uv 配置虚拟环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   apt update
   apt install -y python3.10 python3.10-venv

   # 创建虚拟环境
   python3 -m venv ~/.python/verl-multiturn-rollout

   # 激活虚拟环境
   source ~/.python/verl-multiturn-rollout/bin/activate

   # 安装 uv
   python3 -m pip install uv

安装 verl 上游
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd ~
   git clone https://github.com/volcengine/verl.git
   cd verl

   # 安装 verl
   python3 -m uv pip install .
   python3 -m uv pip install -r ./requirements_sglang.txt

# 手动安装 flash-attn
   python3 -m uv pip install wheel
   python3 -m uv pip install packaging
   python3 -m uv pip install flash-attn --no-build-isolation --no-deps

设置本地检索引擎
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您使用的是自己的本地检索服务，可以跳过此步骤。我们选择了在 search-R1 示例中提供的本地密集检索器；详细说明请参见 `searchR1 文档 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__。简而言之：

-  GPU 版本提供更高的准确性和速度；每个 GPU 大约使用 5–7 GB 的内存。
-  CPU 版本可用于简单测试，但检索精度较低，这会降低训练性能。有关详细信息，请参见 search-R1 中的 `检索器文档 <https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md>`__。
-  推荐使用 Conda 安装 faiss-gpu=1.8.0；venv 可能会导致错误。

**注意**：为了同时启动训练过程和本地检索服务，我们启动两个独立的 Python 环境。训练使用的是在 verl-multiturn-rollout 环境中的 uv，而检索器则使用 conda 安装 ``faiss-gpu``。

.. code:: bash

   # 下载 Miniconda 安装脚本
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

   # 以批处理模式安装到 $HOME/miniconda3
   bash ~/miniconda.sh -b -p $HOME/miniconda3

   # 激活 conda（仅在当前 shell 中）
   eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

   # （可选）将 conda 添加到默认 shell 启动
   conda init

   # 重新加载 shell 配置
   source ~/.bashrc

   # 创建并激活 Python 3.10 的检索器环境
   conda create -n retriever python=3.10 -y
   conda activate retriever

   # 安装 PyTorch（带 GPU 支持）及相关库
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

   # 安装其他 Python 包
   pip install transformers datasets pyserini huggingface_hub

# 安装 GPU 版本的 faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   # 安装 API 服务框架
   pip install uvicorn fastapi

下载索引和语料库
~~~~~~~~~~~~~~~~~~

本地检索文件较大，请准备足够的磁盘空间。
下载大约需要 60–70 GB，解压后大约需要 132 GB：

.. code:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path
   cat $save_path/part_* > $save_path/e5_Flat.index
   gzip -d $save_path/wiki-18.jsonl.gz

启动本地平面 e5 检索服务器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 第一次启动将下载模型并加载索引。
2. 除了下载，启动大约需要 1–2 分钟。
3. 启动后，每个 GPU 使用大约 5–7 GB 的内存，剩余内存用于多轮强化学习训练。

.. code:: bash

   conda activate retriever

   index_file=$save_path/e5_Flat.index
   corpus_file=$save_path/wiki-18.jsonl
   retriever_name=e5
   retriever_path=intfloat/e5-base-v2

```bash
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
     --index_path $index_file \
     --corpus_path $corpus_file \
     --topk 3 \
     --retriever_name $retriever_name \
     --retriever_model $retriever_path \
     --faiss_gpu

设置 WANDB_API_KEY
~~~~~~~~~~~~~~~~~~~~

.. code:: bash
```

```
   export WANDB_API_KEY={您的_WANDB_API_KEY}
```

# 定义时间戳函数
   function now() {
       date '+%Y-%m-%d-%H-%M'
   }

**预处理数据集**
~~~~~~~~~~~~~~~~~~~~~~~~~~

   **注意：** 以下数据处理和训练命令必须在 verl-multiturn-rollout 环境中运行。

.. code:: bash

```plaintext
   python3 examples/data_preprocess/preprocess_search_r1_dataset.py
```

在 8 x H20 上进行测试
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 确保定义了 now() 函数
   # 创建日志目录
   mkdir -p logs

   # 设置 GPU 并使用合适的日志路径运行
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

   nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
     trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) \
     > logs/searchR1-like$(now).log 2>&1 &

自定义搜索配置
---------------------------

要启用多轮推理，请在您的配置中设置以下字段：

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: "sglang"
       multi_turn:
         enable: True

您必须在 ``examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`` 中指定 ``retrieval_service_url``，并正确配置并发性。有关并发性的更多详细信息，请参阅 Sandbox Fusion 示例：

.. code:: yaml

   tools:
     - class_name: verl.tools.search_tool.SearchTool
       config:
         retrieval_service_url: http://127.0.0.1:8000/retrieve
         num_workers: 120
         rate_limit: 120
         timeout: 30

检索器的输入/输出格式如下。如果您的服务参数匹配，只需修改 ``retrieval_service_url``。您还可以在 ``search_r1_like_utils.py`` 中进行自定义。

.. code:: python

   输入格式：
   {
     "queries": ["什么是Python？", "告诉我关于神经网络的事。"],
     "topk": 3,
     "return_scores": true
   }

   输出格式（当 return_scores=True 时，返回相似度分数）：
   {
       "result": [
           [   # 每个查询的结果
               {
                   "document": doc, "score": score
               },
               # ... 更多文档
           ],
           # ... 其他查询的结果
       ]
   }

笔记
-----

1. 总训练时间约为27小时；同时，验证数据集非常大（51k），每次验证大约需要6000秒。（因此，默认情况下``val_before_train=False``）