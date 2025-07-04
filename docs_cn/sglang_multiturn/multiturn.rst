多轮发布支持
==========================

最后更新：2025年6月27日。

基本配置
~~~~~~~~~~~~~~~~~~~

要启用多轮发布，请确保在您的发布配置中配置以下字段：

.. code-block:: yaml

actor_rollout_ref: 
        rollout: 
            multi_turn: True
            name: "sglang"

这些配置在回合中激活了 sglang 引擎以支持多轮交互。

自定义工具配置
~~~~~~~~~~~~~~~~~~~~~~~~~

对于自定义环境交互工具，您可以基于 ``verl.tools.base_tool.BaseTool`` 实现自己的工具。然后，在 YAML 文件中指定您的工具配置：

.. code-block:: yaml

```rst
工具:
      - 类名: ""
        配置: {}
        工具模式:

您可以参考 GSM8KTool_example_configuration_，这是工具配置的一个示例。其实现可以在 gsm8k_tool.py_ 中找到。

最后，在您的发布配置中设置 ``tools_config_file``：
```

.. code-block:: yaml

actor_rollout_ref:
        rollout:
            tool_kwargs:
                tools_config_file: <path_to_tool_yaml_file>

这允许在演员（actor）回放步骤中集成自定义工具行为。

如果您希望在回放中进行模拟交互，可以在您的回放配置中设置 ``interaction_config_file``：

.. code-block:: yaml

interaction:
      - class_name: ""
        config: {}

.. code-block:: yaml

actor_rollout_ref:
        rollout:
            interaction_config_file: <path_to_interaction_yaml_file>

多轮对话标记化
~~~~~~~~~~~~~~~~~~~~~~~

对多轮对话的回合进行标记化是一个挑战：在应用聊天模板并标记化完整的消息列表后，很难识别哪些标记属于助手消息。由于标记列表是扁平的，它缺乏与消息角色的直接对齐。

为了解决这个问题，我们采用了一种**基于增量的标记化**策略。每当大型语言模型（LLM）生成一条新消息时，我们会：

1. 对所有先前的消息（`messages[:i]`）应用聊天模板。
2. 再次对包括最新消息的聊天模板应用（`messages[:i+1]`）。
3. 仅对这两个序列化消息字符串之间的*增量*进行标记化。

这确保了只有助手生成的标记被包含在损失掩码中。

.. code-block:: python

# 使用分词器时
   # 通过设置 add_generation_prompt=True 来排除助手提示（例如，"<|im_start|>assistant"）对损失的影响
   prev = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=False)
   curr = tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False, tokenize=False)
   token_ids += tokenizer.encode(curr[len(prev):], add_special_tokens=False)
   loss_mask += [1] * len(token_ids)  # 仅对新的助手令牌进行掩码

.. code-block:: python

# 使用处理器时
   # 通过设置 add_generation_prompt=True 来排除助手提示（例如，"<|im_start|>assistant"）对损失的影响
   prev = processor.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=False)
   prev_model_inputs = processor(text=prev, images=images, videos=videos, return_tensors="pt")[0].tolist()
   curr = processor.apply_chat_template(messages[:i+1], add_generation_prompt=False, tokenize=False)
   curr_model_inputs = processor(text=curr, images=images, videos=videos, return_tensors="pt")[0].tolist()
   token_ids += curr_model_inputs["input_ids"][len(prev_model_inputs["input_ids"]):]
   loss_mask += [1] * len(token_ids)  # 仅对新的助手令牌进行掩码

虽然我们已经验证这在完整消息标记化时产生一致的结果，但未来模型的聊天模板可能会破坏兼容性。为了防止静默的不一致，我们默认在每次回滚结束时比较基于增量的标记化与完整标记化的结果。

如果您看到以下警告，可以在日志中检查不匹配的子字符串：

.. code-block::

在训练和推理过程中检测到不一致的标记化。这可能导致训练期间出现意外行为。请检查您的聊天模板，以确定这是否是故意的。有关更多信息，请参阅 multiturn README.md。

标记化完整性检查模式可以通过 ``actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode`` 参数进行配置，该参数接受以下值：

- ``strict``（默认）：对基于增量的标记化结果和完整标记化结果进行严格比较，对于任何差异发出警告。

- ``ignore_strippable``：忽略空白字符（``\n``、``\t``、``\r``、空格）之间的差异，同时仍然检查有意义的文本不匹配。这在调试聊天模板问题时非常有用，因为在这种情况下，空白的变化是预期和可接受的。

- ``off``：完全禁用标记化完整性检查。仅在您彻底验证标记化差异是预期的且不会影响训练时使用此选项。

示例配置：

.. code-block:: yaml

actor_rollout_ref:
        rollout:
            multi_turn:
                tokenization_sanity_check_mode: "ignore_strippable"  # 可选值: "strict", "ignore_strippable", "off"

特殊情况
^^^^^^^^^^^^^

某些模型（例如，Qwen/QwQ-32B 和 Qwen3 系列）在聊天模板渲染过程中会移除内部推理内容。因此，消息内容在不同回合之间可能会有所不同，这使得基于增量的标记化不准确。

例如，对于以下对话：

.. code-block:: python

```python
messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "2 + 2 等于多少？"},
        {"role": "assistant", "content": "<think>用户询问一个简单的数学问题。</think> 2 + 2 = 4."},
        {"role": "user", "content": "解释一下原因。"},
        {"role": "assistant", "content": "<think>用户想知道答案背后的推理。寻找一个好的解释</think>",
         "tool_calls": [{"id": "tool1", "type": "search", "arguments": {"query": "为什么 2 + 2 = 4？"}}]},
        {"role": "tool", "content": "两个加两个等于四，因为这是一个基本的算术运算。"},
        {"role": "assistant", "content": "<think>工具提供了一个好的解释。</think>两个加两个等于四，因为这是一个基本的算术运算。"}
    ]

1. Qwen/QwQ-32B 在应用聊天模板后将移除所有推理内容，除了最后一条助手消息。
```

.. code-block:: text

# 多轮对话

## 简介

多轮对话系统旨在处理用户与系统之间的多次交互。与单轮对话不同，多轮对话能够理解上下文并保持对话的连贯性。

## 关键概念

- **上下文管理**：多轮对话系统需要有效地管理上下文，以便在对话的不同阶段保持一致性。
- **状态跟踪**：系统需要跟踪用户的意图和状态，以便提供相关的响应。

## 工作流程

1. **输入解析**：系统接收用户输入并解析其意图。
2. **上下文更新**：根据用户的输入更新对话上下文。
3. **响应生成**：生成适当的响应并返回给用户。

## 示例

以下是一个多轮对话的示例：

用户：我想订一张去纽约的机票。

系统：好的，请问您希望什么时候出发？

用户：下周一。

系统：好的，您希望选择哪个航空公司？

## 结论

多轮对话系统通过有效的上下文管理和状态跟踪，能够提供更自然和连贯的用户体验。

.. code-block:: text

# 多轮对话

## 概述

多轮对话系统旨在处理用户与系统之间的多次交互。与单轮对话不同，多轮对话需要保持上下文，以便在对话的不同阶段理解用户的意图。

## 处理多轮对话

为了处理多轮对话，我们采用一种**固定基础对话**的方法，该方法仅包含单个系统消息和用户消息。由于此基础不包括助手消息或推理内容，因此在多个回合中保持一致。

### 示例

以下是一个简单的多轮对话示例：

1. **用户**: 2 + 2 等于多少？
2. **助手**: 2 + 2 = 4。
3. **用户**: 解释一下原因。
4. **助手**: 2 + 2 等于 4，因为这是一个基本的算术运算。

在这个示例中，助手能够根据用户的提问提供准确的答案，并在用户请求解释时给出合理的解释。

## 结论

多轮对话系统的设计需要考虑上下文的保持和用户意图的理解。通过使用固定基础对话的方法，可以有效地管理对话的状态和内容。

.. code-block:: python

```python
BASE_CHAT_HISTORY = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "我是一个用户。"}
    ]
    prev = tokenizer.apply_chat_template(BASE_CHAT_HISTORY, add_generation_prompt=True, tokenize=False)
    curr = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, messages[i]], add_generation_prompt=False, tokenize=False)
    token_ids += tokenizer.encode(curr[len(prev):], add_special_tokens=False)
    loss_mask += [1] * len(token_ids)

该方法在 Qwen3 系列中效果良好。然而，Qwen/QwQ-32B 目前在其聊天模板中存在一个 bug。已经提出了修复方案，但尚未被采纳。在此之前，请使用以下命令下载修复后的模型版本：
```

.. code-block:: bash

```bash
pip install huggingface_hub
    huggingface-cli download Qwen/QwQ-32B --revision refs/pr/81
```

.. _fix: https://huggingface.co/Qwen/QwQ-32B/discussions/81

训练与推理模板之间的差异
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

尽管上述方法修复了增量不匹配问题，但推理时聊天模板中推理内容的移除引入了新的差异：训练使用完整的推理内容，而推理则不使用。

这种不匹配可能以不可预测的方式影响模型性能。为了避免这种情况，我们默认在训练和推理中都使用完整的响应（包括推理）。

然而，这种方法也带来了权衡：

1. 长的推理内容很容易超过模型的上下文窗口，特别是在多轮推理中。
2. 现在推理和生产环境之间存在不匹配——如果在生产中使用默认聊天模板，模型将不会拥有来自过去轮次的推理内容。

我们仍在评估这些问题的影响。如果您遇到上下文长度问题或更喜欢与生产环境匹配的推理（即，排除推理），您可以启用：

``actor_rollout_ref.rollout.multi_turn.use_inference_chat_template = True``

GSM8K 多轮训练性能  
~~~~~~~~~~~~~~~~~~~~~~

查看 GSM8K 任务上多轮 rollout 的训练性能，请点击 HERE_。

.. _HERE: https://wandb.ai/zhaochenyang20/gsm8k_async_rl/runs/1ro1r7om?nw=nwuserzhaochenyang20

.. _GSM8KTool_example_configuration: https://github.com/volcengine/verl/blob/main/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml

.. _gsm8k_tool.py: https://github.com/volcengine/verl/blob/main/verl/tools/gsm8k_tool.py

交互系统  
~~~~~~~~~~~~~~~~~~

有关在 RL 训练期间进行动态对话反馈的信息，请参见：

.. toctree::
   :maxdepth: 1

   interaction_system

搜索工具集成
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   search_tool_example

代码讲解
~~~~~~~~~~~~~~~~~~~~~~~
如果您想更深入地了解代码执行流程，请阅读 https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/verl/multi-turn/code-walk-through