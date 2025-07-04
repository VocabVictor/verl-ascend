Ray 调试教程
==================

最后更新：2025年04月23日


.. _wuxibin89: https://github.com/wuxibin89

作者：`Ao Shen <https://aoshen524.github.io/>`_.

如何调试？
---------------------


Ray 分布式调试器 VSCode 扩展（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 从 Ray 2.39 开始，Anyscale 引入了 `Ray 分布式调试器 <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode 扩展。请按照扩展的安装说明进行操作，然后使用之前获得的仪表板 URL 添加您的集群。

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray 分布式调试器 VSCode 扩展截图

2. 前提条件。

确保安装以下内容（有关更多详细信息，请参阅扩展的 README）：

   - Visual Studio Code  
   - `ray[default]` >= 2.9.1  
   - `debugpy` >= 1.8.0  

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/readme.png?raw=true
      :alt: 带有 Ray 前提条件的 VSCode

3. 环境变量

   要启用事后调试，请设置：

   .. code-block:: bash

导出 RAY_DEBUG_POST_MORTEM=1

.. admonition:: 注意
      :class: important

      在启动 Ray 之前，请确保移除任何遗留标志：

      - `RAY_DEBUG=legacy`  
      - `--ray-debugger-external`

4. 配置断点 在您的代码中设置 `breakpoint()`，并将作业提交到集群。然后扩展将显示断点信息。

   1. 在您的远程函数中插入 `breakpoint()` 调用。  
   2. 将您的作业提交到集群。  

   扩展将检测活动断点并在 VSCode 中显示它们。

   **注意：** 断点仅在使用 `@ray.remote` 装饰的函数内部受支持。

5. 启动调试器。

   直接从命令行运行您的作业（不要使用 `launch.json`）：

   .. code-block:: bash

```plaintext
      python job.py
```

6. 附加到断点

一旦进程命中第一个 `breakpoint()`，请点击 VSCode 侧边栏中的 Ray 分布式调试器图标以附加调试器。

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/launch.png?raw=true
      :alt: 将 VSCode 调试器附加到 Ray 进程

7. 使用多个 breakpoint() 进行调试

对于每个后续任务，首先断开当前的调试会话，然后再次点击扩展图标以附加到下一个断点。

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/disconnect.png?raw=true
      :alt: 断开和重新连接调试器

遗留 Ray 调试器
~~~~~~~~~~~~~~~~~~~
1. Ray 具有内置的遗留 `debugger <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`_，允许您调试分布式应用程序。要启用调试器，请使用 ``RAY_DEBUG=legacy`` 和 ``--ray-debugger-external`` 启动 Ray 集群。

.. code-block:: bash

# 启动头节点
    RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
    # 启动工作节点
    RAY_DEBUG=legacy ray start --address='10.124.46.192:6379' --ray-debugger-external

2. 在你的代码中设置断点，并提交作业到集群。然后运行 ``ray debug`` 等待断点：

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/legacy.png?raw=true