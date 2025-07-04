实用工具
============

最后更新：2025年5月19日（API 文档字符串为自动生成）。

本节文档记录了 VERL 库中的实用函数和类。

Python 功能实用工具
------------------------------

.. automodule:: verl.utils.py_functional
   :members: append_to_dict

文件系统实用工具
------------------------

.. automodule:: verl.utils.fs
   :members: copy_to_local

跟踪实用工具
---------------------

.. automodule:: verl.utils.tracking
   :members: Tracking

指标实用工具
---------------------

.. automodule:: verl.utils.metric
   :members: reduce_metrics

检查点管理
------------------------

.. automodule:: verl.utils.checkpoint.checkpoint_manager
   :members: find_latest_ckpt_path

.. automodule:: verl.utils.checkpoint.fsdp_checkpoint_manager
   :members: FSDPCheckpointManager

数据集实用工具
---------------------

.. automodule:: verl.utils.dataset.rl_dataset
   :members: RLHFDataset, collate_fn

Torch 功能实用工具
-----------------------------

.. automodule:: verl.utils.torch_functional
   :members: get_constant_schedule_with_warmup, masked_whiten, masked_mean, logprobs_from_logits

序列长度平衡
----------------------------

.. automodule:: verl.utils.seqlen_balancing
   :members: get_reverse_idx, rearrange_micro_batches

乌利西斯工具
--------------------

.. automodule:: verl.utils.ulysses
   :members: gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

FSDP工具
------------------

.. automodule:: verl.utils.fsdp_utils
   :members: get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn, load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer,

调试工具
-------------------

.. automodule:: verl.utils.debug
   :members: log_gpu_memory_usage, GPUMemoryLogger