单控制器接口
============================

最后更新：2025年5月27日（API 文档字符串为自动生成）。

单控制器提供了一个统一的接口，用于管理使用 Ray 或其他后端的分布式工作者，并在它们之间执行函数。它简化了任务调度和结果收集的过程，特别是在处理数据并行性或模型并行性时。

核心 API
~~~~~~~~~~~~~~~~~

.. autoclass:: verl.single_controller.Worker
   :members: __init__, __new__, get_master_addr_port, get_cuda_visible_devices, world_size, rank

.. autoclass:: verl.single_controller.WorkerGroup
   :members: __init__, world_size

.. autoclass:: verl.single_controller.ClassWithInitArgs
   :members: __init__, __call__

.. autoclass:: verl.single_controller.ResourcePool
   :members: __init__, world_size, local_world_size_list, local_rank_list

.. autoclass:: verl.single_controller.ray.RayWorkerGroup
   :members: __init__

.. autofunction:: verl.single_controller.ray.create_colocated_worker_cls