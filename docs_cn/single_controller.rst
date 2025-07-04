```
``verl.single_controller`` 的设计
==============================================

最后更新：2025年5月21日。

**作者：**\  `Wang Zhang <https://github.com/zw0610>`__
```

前言
-------

我们为``verl``的开发者准备了本文件，特别是那些希望理解或贡献于``verl.single_controller``模块的开发者。该文档并不面向最终用户，而是为那些希望理解架构原理和内部机制的贡献者而准备的。

--------------

来源
------

``single_controller``模块源于我收到的一个请求——将一个玩具单进程RLHF脚本适配为一个分布式系统，尽量减少更改，同时保持调试的便利性。

常见的做法——例如使用PyTorch的分布式数据并行(Distributed Data Parallel, DDP)——通常涉及包装``nn.Module``并启动多个进程，这些进程在不同的rank下执行相同的功能。然而，在分布式RLHF的背景下，这种方法存在两个主要限制：
- 难以表示PPO所需的多个有向无环图(DAG)；
- 难以在训练过程中检查中间张量。

为了保持可调试性，我们选择了一种不同的方法——将训练循环分解为明确的阶段，如``generate_sequences``、``compute_advantages``等。

我们选择了`Ray <https://www.ray.io/>`__作为``verl``的初始后端，因为它能够将Python类方法暴露为RPC端点。然而，Ray的默认模型仅支持**一个方法调用，一个RPC**，而训练大型语言模型(LLM)通常需要跨多个进程进行协调。

为了向用户隐藏对单个方法的多Ray演员调用，我们引入了以下组件：

-  ``WorkerGroup`` – 管理一组远程工作者，并为多进程分布式计算提供统一接口；
-  ``ResourcePool`` – 将计算资源绑定到工作进程；
-  ``ClassWithArgs`` – 允许使用指定的初始化参数进行延迟远程实例化。

--------------

运行示例：``generate_sequences``
-----------------------------------------

为了说明设计，我们将演示如何在分布式工作者中注册和调用``ActorRolloutRefWorker``类中的``generate_sequences``方法。

--------------

步骤 1：使用装饰器注册
~~~~~~~~~~~~~~~~~~~~~~~~~

第一步是定义 ``generate_sequences`` 并使用 ``@register`` 装饰它，因为它将在驱动脚本中被调用。

**来源：**
`fsdp_workers.py <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/workers/fsdp_workers.py#L528>`__

.. code:: python

   class ActorRolloutRefWorker(Worker):
       ...
       @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
       def generate_sequences(self, prompts: DataProto):
           prompts = prompts.to(torch.cuda.current_device())
           ...

``@register`` 装饰器为 ``generate_sequences`` 方法添加元数据。目前，它并不改变功能，而是通过一个魔法键 (``MAGIC_ATTR``) 附加属性：

**来源：**
`decorator.py <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L411>`__

.. code:: python

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
       ...
       def decorator(func):
           @wraps(func)
           def inner(*args, **kwargs):
               if materialize_futures:
                   args, kwargs = _materialize_futures(*args, **kwargs)
               return func(*args, **kwargs)

           attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
           setattr(inner, MAGIC_ATTR, attrs)
           return inner
```

返回装饰器

如代码所示，``dispatch_mode``、``execute_mode`` 和 ``blocking`` 的值附加在 ``generate_sequences`` 方法上。

--------------

步骤 2：初始化期间的绑定
~~~~~~~~~~~~~~~~~~~~~~~~~~

这些附加属性在将 ``ActorRolloutRefWorker`` 作为 ``RayClassWithArgs`` 包装后传递到 ``RayWorkerGroup`` 时被提取和利用。

**来源：**
`main_generation.py <https://github.com/volcengine/verl/blob/4ae9a0fdab229f75f080e9478807783ed4c97154/verl/trainer/main_generation.py#L82>`__

.. code:: python

   ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
   resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
   wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)

在 ``RayWorkerGroup`` 的
`初始化 <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/ray/base.py#L184>`__ 过程中，发生了两个关键步骤：

1. 工作实例 (Ray actors) 被创建：
   `RayWorkerGroup._init_with_resource_pool <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/ray/base.py#L211>`__
2. 使用 ``@register`` 装饰的方法被绑定到 ``RayWorkerGroup``：
   `RayWorkerGroup._bind_worker_method <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/ray/base.py#L214>`__

.. figure:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/worker_group_init.png?raw=true
   :alt: 工作组的初始化与绑定

   initialization_and_binding_of_worker_group

绑定过程是 ``verl.single_controller`` 的核心。

**关键函数：**
`WorkerGroup._bind_worker_method <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/worker_group.py#L143>`__

.. code:: python

   def _bind_worker_method(self, user_defined_cls, func_generator):
       ...
       for method_name in dir(user_defined_cls):
           try:
               method = getattr(user_defined_cls, method_name)
               assert callable(method)
           except Exception:
               continue  # 跳过属性
           <<<待续 1>>>

当一个方法具有 ``MAGIC_ATTR`` 时，通过 ``@register`` 设置的属性被提取：

.. code:: python

           <<<继续 1>>>
           if hasattr(method, MAGIC_ATTR):
               attribute = getattr(method, MAGIC_ATTR)
               dispatch_mode = attribute["dispatch_mode"]
               execute_mode = attribute["execute_mode"]
               blocking = attribute["blocking"]

               <<<待续 2>>>

如上图所示，这些属性被输入到 ``func_generator`` 中。然而，``func_generator`` 需要 ``method_name``、``dispatch_fn``、``collect_fn``、``execute_fn`` 和 ``blocking``。我们需要从 `DISPATCH_MODE_FN_REGISTRY <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L387>`__ 中找到与 ``dispatch_mode`` (``DP_COMPUTE_PROTO``) 相关的 ``dispatch_fn`` 和 ``collect_fn``：

.. code:: python3

   DISPATCH_MODE_FN_REGISTRY = {
       Dispatch.ONE_TO_ALL: {
           "dispatch_fn": dispatch_one_to_all,
           "collect_fn": collect_all_to_all,
       },
       ...
       Dispatch.DP_COMPUTE_PROTO: {
           "dispatch_fn": dispatch_dp_compute_data_proto,
           "collect_fn": collect_dp_compute_data_proto,
       },
       ...
   }

同样，``execute_fn`` 是通过 ``execute_mode`` 选择的，并通过以下方式提取：

.. code:: python

               <<<continue 2>>>
               # 获取 execute_fn_name
               execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
               wg_execute_fn_name = execute_mode["execute_fn_name"]

# 从字符串获取 execute_fn
               try:
                   execute_fn = getattr(self, wg_execute_fn_name)
                   assert callable(execute_fn), "execute_fn 必须是可调用的"
               except Exception:
                   print(f"execute_fn {wg_execute_fn_name} 无效")
                   raise
               <<<待续 3>>>

在这个 ``generate_sequences`` 的案例中：-
``dispatch_mode = Dispatch.DP_COMPUTE_PROTO`` -
``dispatch_fn = dispatch_dp_compute_data_proto`` -
``collect_fn = collect_dp_compute_data_proto`` -
``execute_fn = RayWorkerGroup.execute_all``

ONE_TO_ALL 与 DP_COMPUTE_PROTO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dispatch_mode`` 与 ``dispatch_fn`` 和 ``collect_fn`` 相关联。顾名思义，``dispatch_fn`` 处理 ``WorkerGroup`` 中的输入参数，并生成一批（列表）输入参数，每个参数将被传递给附加到 ``WorkerGroup`` 的工作者。

``dispatch_fn`` 的 ``ONE_TO_ALL`` 是
`dispatch_one_to_all <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L119>`__，
它只是将所有输入参数复制到 N 个副本中，其中 N 等于附加到 ``worker_group`` 的工作线程数量：

.. code:: python

   def dispatch_one_to_all(worker_group, *args, **kwargs):
       args = tuple([arg] * worker_group.world_size for arg in args)
       kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
       return args, kwargs

``dispatch_fn`` 的 ``DP_COMPUTE_PROTO`` 是
`dispatch_dp_compute_data_proto <https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L350>`__，
它使用 ``DataProto.chunk`` 将一个大的 ``DataProto`` 拆分为 N 个较小的 ``DataProto``，其中 N 等于 ``worker_group`` 的 world_size（工作线程数量）：

.. code:: python

   def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
       from verl.single_controller.base.worker_group import WorkerGroup

```rst
assert isinstance(worker_group, WorkerGroup)
       # 注意：为 dp 计算 DatapProto 启用自动填充
       splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
           worker_group.world_size,
           *args,
           **kwargs,
       )
       return splitted_args, splitted_kwargs

``collect_fn`` 遵循相同的模式，并处理来自 ``WorkerGroup`` 所有工作者返回值的批次（列表），并将其合并为一个列表，类似于 ``collect_all_to_all``，或者作为一个大型 ``DataProto``，如 ``collect_dp_compute_data_proto`` 所做的那样。

最后，使用 ``func_generator`` 动态生成一个新方法，并将其添加到 ``WorkerGroup`` 实例中：

.. code:: python

               <<<continue 3>>>
               # 将新方法绑定到 RayWorkerGroup
               func = func_generator(
                   self,
                   method_name,
                   dispatch_fn=dispatch_fn,
                   collect_fn=collect_fn,
                   execute_fn=execute_fn,
                   blocking=blocking,
               )
```

```python
try:
                   setattr(self, method_name, func)
                   method_names.append(method_name)
               except Exception as e:
                   raise ValueError(f"无法设置方法名 {method_name}") from e

这使得该方法可以通过 ``WorkerGroup`` 接口调用。
```

--------------

步骤 3：调用链
~~~~~~~~~~~~~~~~~~

以上所有机制确保分布式调用与单进程调用感觉相同。在原始的单进程脚本中，代码如下：

.. code:: python

   rollout = Rollout()
   rollout.generate_sequences(batch)

使用 ``verl``，多进程程序变为：

.. code:: python

   rollout = RayWorkerGroup(resource_pool=[4], RayClassWithArgs(Rollout))
   rollout.generate_sequences(batch)

.. figure:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/call_generate_sequences.png?raw=true
   :alt: generate_sequences的调用链

   call_chain_of_generate_sequences

在这个简单的调用背后：- ``dispatch_fn`` 将输入分配到各个工作节点 - ``execute_fn`` 执行实际的远程调用 - ``collect_fn`` 收集结果

所有这些都被抽象化，使开发者能够以最小的改动编写分布式代码。

--------------

超越强化学习后训练：泛化"verl.single_controller"
-------------------------------------------------------

"verl.single_controller"模块的泛化远远超出了强化学习范畴。它提供了一个清晰的抽象，用于批量处理远程方法调用，并具有自动的输入/输出处理。

通过最小化单进程和多进程脚本之间的差距，"verl.single_controller"打开了在更广泛领域进行分布式计算的大门——不仅限于强化学习后训练。

我们希望这种设计能激发社区中更多的示例和扩展。