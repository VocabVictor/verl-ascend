常见问题解答
====================================

最近更新日期：2025年6月25日。

Ray相关
------------

如何在使用分布式Ray进行调试时添加断点？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

请查看Ray官方调试指南：https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html


"无法向raylet注册worker"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此问题的原因是由于一些系统设置，例如，SLURM对节点上的CPU共享设置了一些限制。
当`ray.init()`尝试启动与机器的CPU核心数量相同的worker进程时，
SLURM的一些限制会限制`core-workers`看到`raylet`进程，从而导致问题的发生。

要解决此问题，您可以将配置项``ray_init.num_cpus``设置为系统允许的数字。

分布式训练
------------------------

如何使用 Ray 在多节点后训练时运行？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

您可以按照 Ray 的官方指南启动一个 ray 集群并提交一个 ray 作业：https://docs.ray.io/en/latest/ray-core/starting-ray.html

然后在配置中，将 ``trainer.nnode`` 配置设置为您作业所需的机器数量。

如何在由 Slurm 管理的集群上使用 verl？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray 为用户提供了 `这个 <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ 官方教程，
以在 Slurm 之上启动一个 Ray 集群。我们已经验证了在多节点设置下在 Slurm 集群上运行 :doc:`GSM8K 示例<../examples/gsm8k_example>` 的以下步骤。

1. [可选] 如果您的集群支持`Apptainer或Singularity <https://apptainer.org/docs/user/main/>`_并希望使用它，请将verl的Docker镜像转换为Apptainer镜像。或者，使用集群上可用的软件包管理器设置环境，或者使用其他容器运行时（例如，通过`Slurm的OCI支持 <https://slurm.schedmd.com/containers.html>`_）。

```bash
apptainer pull /your/dest/dir/vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3.sif docker://verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
```

2. 参考:doc:`GSM8K示例<../examples/gsm8k_example>`准备数据集和模型检查点。

3. 使用您集群的信息修改`examples/slurm/ray_on_slurm.slurm <https://github.com/volcengine/verl/blob/main/examples/slurm/ray_on_slurm.slurm>`_。

4. 使用`sbatch`将作业脚本提交到Slurm集群。

请注意，Slurm集群设置可能有所不同。如果遇到任何问题，请参考Ray的`Slurm用户指南 <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ 查看常见注意事项。

如果您更改了 Slurm 资源规范，请确保根据需要更新作业脚本中的环境变量。

安装相关
------------------------

NotImplementedError: TensorDict(张量字典) 不支持使用 `in` 关键字进行成员检查。
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

详细错误信息：

.. code:: bash

    NotImplementedError: TensorDict(张量字典) 不支持使用 `in` 关键字进行成员检查。如果您想检查特定键是否在您的 TensorDict(张量字典) 中，请改用 `key in tensordict.keys()`。

问题原因：在 linux-arm64 平台上没有适用的 tensordict 包版本。确认方法如下：

.. code:: bash

    pip install tensordict==0.6.2

输出示例:

```bash
错误: 无法找到满足要求 tensordict==0.6.2 的版本 (可用版本: 0.0.1a0, 0.0.1b0, 0.0.1rc0, 0.0.2a0, 0.0.2b0, 0.0.3, 0.1.0, 0.1.1, 0.1.2, 0.8.0, 0.8.1, 0.8.2, 0.8.3)
错误: 找不到符合 tensordict==0.6.2 的发行版

解决方案 1:
  从源代码安装 tensordict:

```bash
pip uninstall tensordict
git clone https://github.com/pytorch/tensordict.git
cd tensordict/
git checkout v0.6.2
python setup.py develop
pip install -v -e .
```

解决方案 2:
  临时修改错误发生的代码: 将 tensordict_var 改为 tensordict_var.keys()


非法内存访问
---------------------------------

如果在执行过程中遇到类似 ``CUDA error: an illegal memory access was encountered`` 的错误消息，请查看 vLLM 文档，以获取针对您的 vLLM 版本的疑难解答步骤。
```

检查点
------------------------

如果您想将模型检查点转换为Hugging Face安全张量格式，请参考``verl/model_merger``。

Triton ``compile_module_from_src`` 错误
------------------------------------------------

如果您遇到类似下面堆栈跟踪的 Triton 编译错误，请根据 https://verl.readthedocs.io/en/latest/examples/config.html 设置``use_torch_compile``标志以禁用融合内核的即时编译。

.. code:: bash

```python
File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 338, in run
    return self.fn.run(*args, **kwargs)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/jit.py", line 607, in run
    device = driver.active.get_current_device()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 23, in __getattr__
    self._initialize_obj()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 20, in _initialize_obj
    self._obj = self._init_fn()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 9, in _create_driver
    return actives[0]()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 371, in __init__
    self.utils = CudaUtils()  # TODO: make static
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 80, in __init__
    mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "cuda_utils")
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 57, in compile_module_from_src
    so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/build.py", line 48, in _build
    ret = subprocess.check_call(cc_cmd)
  File "/data/lbh/conda_envs/verl/lib/python3.10/subprocess.py", line 369, in check_call
    raise CalledProcessError(retcode, cmd)
```

**训练批大小、小批大小和微批大小的含义是什么？**

------------------------------------------------------------------------------------------

这幅图展示了不同批大小配置之间的关系。

https://excalidraw.com/#json=pfhkRmiLm1jnnRli9VFhb,Ut4E8peALlgAUpr7E5pPCA

.. image:: https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d

如何生成 Ray 时间线以分析训练作业的性能？
------------------------------------------------------------------------------------------

要生成 Ray 时间线文件，您可以将配置项 ``ray_init.timeline_file`` 设置为一个 json 文件路径。
例如：

.. code:: bash

    ray_init.timeline_file=/tmp/ray_timeline.json
  
文件将在训练作业结束时在指定路径生成。
您可以使用诸如 chrome://tracing 或 Perfetto UI 这样的工具来查看 Ray 时间线文件。

该图显示了从在 1 个节点上使用 4 个 GPU 进行的训练作业生成的 Ray 时间线文件。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray_timeline.png?raw=true

如何仅为wandb设置代理？
------------------------------------------------------------------------------------------

如果您需要代理来访问wandb，您可以在您的训练作业脚本中添加以下配置。与使用全局 https_proxy 环境变量相比，这种方法不会干扰其他 HTTP 请求，比如 ChatCompletionScheduler。

```bash
+trainer.wandb_proxy=http://<您的代理和端口>
```