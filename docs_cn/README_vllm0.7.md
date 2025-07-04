# 升级到 vllm >= 0.7

注意：verl+vllm 0.8.3 现在已稳定。请参阅 ``docs/README_vllm0.8.md`` 以获取升级指南。

## 安装

注意：在撰写本文时，verl+vllm 0.7.x 支持 **FSDP**（全模型分布式并行）用于训练，支持 **vLLM**（可变长语言模型）用于推理。

```
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# 安装 verl
git clone https://github.com/volcengine/verl.git  
cd verl  
pip3 install -e .

# 安装最新稳定版本的 vLLM
pip3 install vllm==0.7.3 

# 安装 flash-attn
pip3 install flash-attn --no-build-isolation

```

注意，如果您正在安装较低版本的 vLLM (0.7.0, 0.7.1, 0.7.2)，在上述步骤之后，您需要手动对 vllm (/path/to/site-packages/vllm 安装后) 进行一些小补丁：

- vllm/distributed/parallel_state.py: 删除以下断言：

```
if (world_size
        != tensor_model_parallel_size * pipeline_model_parallel_size):
    raise RuntimeError(
        f"world_size ({world_size}) is not equal to "
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
        f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

```

- vllm/executor/uniproc_executor.py: 将`local_rank = rank`修改为`local_rank = int(os.environ["LOCAL_RANK"])`
- vllm/model_executor/model_loader/weight_utils.py: 在`pt_weights_iterator`中移除`torch.cuda.empty_cache()`

## 特性

### 使用 CUDA 图形

安装完成后，可以使用 FSDP 作为训练后端的示例。默认情况下，`enforce_eager` 被设置为 True，这会禁用 CUDA 图形。要享受 vLLM>=0.7 的 CUDA 图形和睡眠模式，请将以下行添加到 bash 脚本中：

```
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=True \

```

对于类似 examples/ppo_trainer/run_qwen2-7b_seq_balance.sh 这样的典型作业，在 vLLM0.7.0 中，rollout 生成时间为 85 秒。通过启用 cudagraph，生成持续时间进一步缩短至 62 秒。

**注意:** 目前，在 vLLM>=0.7 中，如果 `SamplingParams` 中的 `n` 大于 1，则存在潜在的性能问题，会影响 rollout 生成时间的稳定性（某些迭代会出现生成时间突增），使用 vLLM 的 V0 引擎。

### 使用 vLLM V1 引擎

使用 vLLM V1 引擎可以避免不稳定性问题，并实现额外的性能改进。要使用 V1 引擎，您可以首先卸载先前安装的 vLLM，然后按照以下步骤安装更新版本。

```
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 2275784
sed -i "903a\    data_parallel_size = world_size // pipeline_model_parallel_size // tensor_model_parallel_size" ./vllm/distributed/parallel_state.py
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

然后，您可以通过设置 `export VLLM_USE_V1=1` 来启用 VLLM V1 引擎。在一些基准测试中，V1 引擎表现出比 vLLM V0 引擎快 1.5 倍的速度提升。vLLM V1 引擎的稳定支持可在 verl main 上获得。