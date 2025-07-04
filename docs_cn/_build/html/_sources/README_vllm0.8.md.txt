# 升级至 vLLM >= 0.8

最后更新时间：2025年5月4日。

## 安装

注意：这个 verl+vLLM 0.8+ 版本支持**FSDP**用于训练和**vLLM**用于推出。

```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# 安装 verl
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .

# 安装最新稳定版本的 vLLM
pip3 install vllm==0.8.3

# 安装 flash-attn
pip3 install flash-attn --no-build-isolation

```

我们为 verl+vLLM 0.8.3 提供了预构建的 Docker 镜像。您可以使用以下命令直接导入它：

```bash
docker pull hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0
```

## 特性

vLLM 0.8+默认支持cuda图形和V1引擎。要启用这些功能，请记得在bash脚本中添加以下行：

```bash
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=True \
```

并且**删除**该环境变量（如果存在）：

## 注意事项

当您直接升级 vllm>=0.8 时，一些依赖包可能会发生版本更改。如果您遇到以下问题：

```bash
in <module> from torch.multiprocessing.reductions import ForkingPickler ImportError: cannot import name 'ForkingPickler' from 'torch.multiprocessing.reductions' (/opt/conda/lib/python3.11/site-packages/torch/multiprocessing/reductions.py)
```

你需要使用命令 `pip install tensordict==0.6.2` 将 `tensordict` 升级到版本 0.6.2。