verl 性能调优针对 AMD (ROCm 内核)
=====================================================

最后更新：2025年04月25日。

作者： `Yang Wang <https://github.com/YangWang92/>`_

为 AMD GPU 启用睡眠模式的 vLLM 补丁
--------------------------------------------------------------

默认情况下，verl 需要 vLLM 启用睡眠模式，这允许 vLLM 在推出后将 GPU 内存卸载到 CPU 内存。然而，这一功能仍在 vLLM 社区审查中。

要启用 vLLM 的睡眠模式，您可以首先使用社区补丁代码（来自 `这个拉取请求 <https://github.com/vllm-project/vllm/pull/12695>`_）从相应的拉取请求构建 vLLM 的源代码。在补丁合并到 vLLM 主分支后，您可以直接从最新版本安装 vLLM。

1. 克隆 vLLM 仓库并使用以下命令构建：

.. code-block:: bash

```bash
git clone -b sleep_amd https://github.com/HollowMan6/vllm.git
    cd vllm
    sudo ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so
    VLLM_TARGET_DEVICE=rocm ROCM_PATH=/opt/rocm/ VLLM_GPU_LANG=HIP SETUPTOOLS_SCM_PRETEND_VERSION=0.8.4.dev python3 setup.py develop
```

2. 此外，请确保在您的 Docker 镜像中使用的 ROCm 版本大于或等于 ROCm 6.3.4，我们建议使用 ROCm 6.4.0 以获得更好的性能（请参见 `this comment <https://github.com/vllm-project/vllm/pull/12695#issuecomment-2637839574>`_）。

升级后，您可以通过运行以下测试代码来验证睡眠模式是否已启用（来自 `this comment <https://github.com/vllm-project/vllm/pull/12695#issuecomment-2637839574>`_）。

.. code-block:: python

```python
import torch
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_sleep_mode=True)

def run_inference(prompt):
    outputs = llm.generate(prompt)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")

print("CUDA 内存使用情况（推理后）:")
torch.cuda.empty_cache()
print(f"{torch.cuda.memory_allocated()=}")

run_inference("旧金山是")
llm.sleep()

print("CUDA 内存使用情况（休眠后）:")
torch.cuda.empty_cache()
print(f"{torch.cuda.memory_allocated()=}")
```

	llm.wake_up()

```python
print("CUDA 内存使用情况（唤醒后）：")
torch.cuda.empty_cache()
print(f"{torch.cuda.memory_allocated()=}")
```

```python
run_inference("巴黎是")
```

如果启用了睡眠模式，您应该会看到内存使用量在睡眠后减少。

在应用 vLLM 补丁并完成安装后，您可以在 verl 中启用睡眠模式以减少内存开销。这允许 verl 在展开过程中卸载未使用的 GPU 内存，从而显著降低在长上下文训练或多节点强化学习期间的内存占用。

启用 CUDA 图并绕过与 ROCm 相关的问题
--------------------------------------------------------------

由于在 ROCm 中可能存在与 CUDA 图捕获相关的问题，我们发现 vLLM 的 CUDA 图功能无法在 AMD 平台的 vLLM V1 模式下在多个节点上启用。这导致展开性能显著降低。

我们的调查显示，当尝试使用 CUDA 图捕获大批量数据时，ROCm 可能会触发意外崩溃。一个解决方法是修补 LLM 配置（来自 `this commit <https://github.com/volcengine/verl/blob/v0.3.0.rc0/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L100-L115>`_）。

.. code-block:: python
	
    self.inference_engine = LLM(
        model=model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt=limit_mm_per_prompt,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        load_format=load_format,
        disable_log_stats=config.disable_log_stats,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
        # enable compilation config to bypass oom on rocm
	# change depends on your GPU memory size
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
        seed=config.get('seed', 0),
    )

然后，您可以通过设置以下环境变量来选择启用 CUDA 图（请参见 `此页面 <https://github.com/volcengine/verl/blob/v0.3.0.rc0/docs/README_vllm0.8.md>`_）：

.. code-block:: bash

actor_rollout_ref.rollout.enforce_eager=False \