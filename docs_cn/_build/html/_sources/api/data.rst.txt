数据接口
=========================

最后更新：2025年5月19日（API 文档字符串是自动生成的）。

DataProto 是数据交换的接口。

`:class:`verl.DataProto` 类包含两个关键成员：

- batch: 一个 :class:`tensordict.TensorDict` 对象，用于实际数据
- meta_info: 一个 :class:`Dict`，包含额外的元信息

TensorDict
~~~~~~~~~~~~

:attr:`DataProto.batch` 是建立在 :class:`tensordict` 之上的，tensordict 是 PyTorch 生态系统中的一个项目。TensorDict 是一个类似字典的张量容器。要实例化一个 TensorDict，您必须指定键值对以及批量大小。

.. code-block:: python

```python
>>> import torch
>>> from tensordict import TensorDict
>>> tensordict = TensorDict({"zeros": torch.zeros(2, 3, 4), "ones": torch.ones(2, 3, 5)}, batch_size=[2,])
>>> tensordict["twos"] = 2 * torch.ones(2, 5, 6)
>>> zeros = tensordict["zeros"]
>>> tensordict
TensorDict(
fields={
    ones: Tensor(shape=torch.Size([2, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False),
    twos: Tensor(shape=torch.Size([2, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
    zeros: Tensor(shape=torch.Size([2, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
batch_size=torch.Size([2]),
device=None,
is_shared=False)

可以沿着其 batch_size 索引一个 tensordict。TensorDict 的内容也可以集体进行操作。
```

.. code-block:: python

>>> tensordict[..., :1]
    TensorDict(
    fields={
        ones: Tensor(shape=torch.Size([1, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        twos: Tensor(shape=torch.Size([1, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
        zeros: Tensor(shape=torch.Size([1, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)
    >>> tensordict = tensordict.to("cuda:0")
    >>> tensordict = tensordict.reshape(6)

有关 :class:`tensordict.TensorDict` 的更多使用信息，请参见官方的 tensordict_ 文档。

.. _tensordict: https://pytorch.org/tensordict/overview.html


核心 API
~~~~~~~~~~~~~~~~~

.. autoclass::  verl.DataProto
   :members: to, select, union, make_iterator, concat