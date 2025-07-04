准备后训练数据
========================================

最后更新：2025年02月09日。

在开始后训练作业之前，我们需要为策略训练准备数据。数据应以parquet格式存储。

我们提供了多个数据预处理脚本，适用于不同的数据集，包括GSM8K、MATH、HelloSwag、Full_hh_rlhf。要准备其他数据集，我们需要遵循以下步骤：数据预处理脚本可以分为两个部分：

1. 第一部分是公共部分，它从huggingface的``datasets``包加载数据集。然后使用``make_map_fn``对数据集进行预处理，并将其存储为parquet格式。

.. code:: python

   import re
   import os
   import datasets

   from verl.utils.hdfs_io import copy, makedirs
   import argparse

   # 提取数据集中每个提示的解决方案
   # def extract_solution(solution_str): 
   # ...


   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--local_dir', default='/opt/tiger/gsm8k')
       parser.add_argument('--hdfs_dir', default=None)

       args = parser.parse_args()

```python
num_few_shot = 5
data_source = 'openai/gsm8k'

dataset = datasets.load_dataset(data_source, 'main')

train_dataset = dataset['train']
test_dataset = dataset['test']

# 构造一个 `def make_map_fn(split)` 用于相应的数据集。
# ...

train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

local_dir = args.local_dir
hdfs_dir = args.hdfs_dir

train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
```

       makedirs(hdfs_dir)

```
       copy(src=local_dir, dst=hdfs_dir)
```

2. 用户需要自行实现 ``make_map_fn()`` 函数（以及 ``extract_solution``），以支持不同的数据集或任务。

我们已经实现了 GSM8k、MATH、Hellaswag 和 Full_hh_rlhf 数据集的数据预处理。我们以 GSM8k 数据集为例：

**GSM8K**

在``make_map_fn``中，每个数据字段应包含以下5个字段：

1. ``data_source``：数据集的名称。用于索引``RewardModule``中的相应奖励函数。
2. ``prompt``：该字段应按照huggingface chat_template的格式构建。``RLHFDataset``中的分词器将应用聊天模板并对提示进行分词。
3. ``ability``：定义任务类别。
4. ``reward_model``：目前，我们在评估过程中仅使用``ground_truth``字段。``ground_truth``是通过``extract_solution``函数计算得出的。**注意**，相应奖励函数的实现应与此提取的``ground_truth``保持一致。
5. ``extra_info``：记录当前提示的一些信息。目前不使用。

.. code:: python

   def extract_solution(solution_str):
       solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # 提取####之后的解决方案
       assert solution is not None
       final_solution = solution.group(0)
       final_solution = final_solution.split('#### ')[1].replace(',', '')
       return final_solution

```python
instruction_following = "让我们一步一步思考，并在 \"####\" 之后输出最终答案。"

   # 为每个数据项添加一行，表示唯一的 ID
   def make_map_fn(split):

       def process_fn(example, idx):
           question = example.pop('question')

           question = question + ' ' + instruction_following

           answer = example.pop('answer')
           solution = extract_solution(answer)
           data = {
               "data_source": data_source,
               "prompt": [{
                   "role": "user",
                   "content": question
               }],
               "ability": "数学",
               "reward_model": {
                   "style": "规则",
                   "ground_truth": solution
               },
               "extra_info": {
                   'split': split,
                   'index': idx
               }
           }
           return data
```

return process_fn