多轮强化学习训练的交互系统
=============================================

最后更新：2025年6月25日。

概述
--------

# 交互系统

verl 交互系统在强化学习训练期间实现动态的多轮对话反馈。该系统允许模型参与迭代问题解决场景，其中交互代理可以根据模型的响应提供纠正反馈、指导或评估。

**多交互支持的新功能**：该系统现在支持在单个训练会话中进行多个命名交互，使得不同样本可以使用不同的交互策略，从而实现复杂的训练场景。这允许课程学习、特定领域的反馈以及在样本级别灵活切换代理。

关键特性：

- **基于异步的架构 (Async-based Architecture)**: 非阻塞交互处理用于分布式训练
- **实例管理 (Instance Management)**: 有状态会话处理，具有唯一实例 ID 以支持并发交互
- **SGLang 集成 (SGLang Integration)**: 与 SGLang 发布系统无缝集成，支持多轮对话
- **配置驱动 (Configuration-driven)**: 通过 YAML 配置文件动态加载代理
- **多交互支持 (Multi-Interaction Support)**: 注册系统支持每个发布的多个命名交互
- **样本级选择 (Sample-Level Selection)**: 每个样本可以通过配置指定使用哪个交互
- **奖励集成 (Reward Integration)**: 与 verl 的奖励系统集成的轮次级评分机制

架构
------------

交互系统遵循基于插件的架构，具有明确的关注点分离：

.. code-block::

交互注册系统
         ↓
    基础交互 (抽象接口)
         ↓
    多个命名交互 (例如，Gsm8kInteraction，自定义交互)
         ↓
    SGLang 部署集成 (interaction_map)
         ↓
    样本级交互选择
         ↓
    异步请求生命周期管理

核心组件
~~~~~~~~~~~~~~~

**交互注册系统**

交互注册系统允许加载和管理多个命名交互：

.. code-block:: python

```python
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
    
    # 从配置中加载多个交互
    interaction_map = initialize_interactions_from_config("config.yaml")
    
    # 通过名称访问特定交互
    gsm8k_interaction = interaction_map["gsm8k"]
    custom_interaction = interaction_map["custom_solver"]
```

**BaseInteraction 接口**

所有交互代理必须实现 ``BaseInteraction`` 抽象类：

.. code-block:: python

```python
from verl.interactions.base import BaseInteraction
from typing import Dict, Any, List, Tuple, Optional

class BaseInteraction:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")
    
    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """初始化交互会话，返回实例ID"""
        
    async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, Dict[str, Any]]:
        """生成响应，返回 (should_terminate, response, score, metadata)"""
        
    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """计算用于强化学习训练的回合级分数"""
        
    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """清理资源"""
```

**请求生命周期**

交互系统通过状态管理与 SGLang 的异步发布集成：

1. ``PENDING`` → 通过 ``start_interaction()`` 初始化交互
2. ``GENERATING`` → 模型生成响应
3. ``INTERACTING`` → 通过 ``generate_response()`` 处理响应
4. ``GENERATING`` → 如果未终止则继续，否则为 ``COMPLETED``

配置
-----

**基本设置**

在您的发布配置中启用交互：

.. code-block:: yaml

actor_rollout_ref:
        rollout:
            multi_turn:
                enable: true
                interaction_config_path: "path/to/interaction_config.yaml"
                max_user_turns: 10
                max_assistant_turns: 10

**交互配置文件**

创建一个交互配置文件（例如，``interaction_config.yaml``）：

**单次交互（旧版格式）**

.. code-block:: yaml

interaction:
      - name: "gsm8k"
        class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}

**多重交互（新格式）**

.. code-block:: yaml

interaction:
      - name: "gsm8k"
        class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}
      - name: "custom_solver"
        class_name: "custom.interactions.CustomInteraction"
        config: 
          solver_type: "advanced"
          timeout: 30
      - name: "code_verifier"
        class_name: "verl.interactions.base.BaseInteraction"
        config: 
          verification_mode: "strict"

**自动名称生成**

如果未提供``name``字段，系统将根据类名自动生成一个名称：

.. code-block:: yaml

interaction:
      - class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}
        # 自动生成名称: "gsm8k"

系统将动态加载所有指定的交互类，并通过名称使其可用。

实现示例: GSM8K
-----------------------------

GSM8K交互演示了一个完整的数学问题解决场景的实现：

.. code-block:: python

```python
from verl.interactions.base import BaseInteraction
from verl.utils.reward_score import gsm8k
from uuid import uuid4

class Gsm8kInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(self, instance_id=None, ground_truth=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        # 提取最后一条用户消息内容
        content = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                content = item.get("content", "")
                break
```

# 确保GSM8K格式（#### 前缀）
            if content.startswith("#### "):
                self._instance_dict[instance_id]["response"] = content
            else:
                self._instance_dict[instance_id]["response"] = "#### " + content

            reward = await self.calculate_score(instance_id)
            if reward == 1.0:
                return True, "您的回答是正确的！", 1.0, {}
            else:
                return False, "您的回答是错误的！您需要反思您的答案并再试一次。", 0.0, {}

        async def calculate_score(self, instance_id, **kwargs):
            return gsm8k.compute_score(
                self._instance_dict[instance_id]["response"],
                self._instance_dict[instance_id]["ground_truth"],
                method="flexible", format_score=0.0, score=1.0,
            )

        async def finalize_interaction(self, instance_id, **kwargs):
            del self._instance_dict[instance_id]

训练集成
--------------------

**训练脚本配置**

在您的训练命令中包含交互配置：

.. code-block:: bash

```bash
python3 -m verl.trainer.main_ppo \\
        --config-path="$CONFIG_PATH" \\
        --config-name='gsm8k_multiturn_grpo_w_interaction' \\
        algorithm.adv_estimator=grpo \\
        data.train_batch_size=512 \\
        data.return_raw_chat=True \\
        actor_rollout_ref.rollout.name=sglang \\
        actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml" \\
        trainer.total_epochs=15
```

**数据要求**

确保您的数据集包含用于交互选择的交互参数，并且具有``name``字段：

.. code-block:: python

# 数据集应在非张量批次中包含 interaction_kwargs
    interaction_kwargs = [
        {"name": "gsm8k", "query": "2+2 等于多少？", "ground_truth": "4"},
        {"name": "custom_solver", "query": "解：x^2 + 5x + 6 = 0", "ground_truth": "x = -2, -3"},
        {"name": "gsm8k", "query": "3+3 等于多少？", "ground_truth": "6"},
    ]

**样本级交互选择**

每个样本可以通过 ``name`` 字段指定使用哪个交互。这使得不同样本可以使用不同的交互策略，从而实现灵活的训练场景：

.. code-block:: python

# 示例：数学问题使用 GSM8K 交互，代码问题使用代码验证器
    data_samples = [
        {
            "prompt": "200 的 15% 是多少？",
            "interaction_kwargs": {
                "name": "gsm8k",
                "query": "200 的 15% 是多少？", 
                "ground_truth": "30"
            }
        },
        {
            "prompt": "编写一个函数检查一个数字是否为质数",
            "interaction_kwargs": {
                "name": "code_verifier",
                "code_type": "python",
                "expected_behavior": "对于质数返回 True"
            }
        }
    ]

**向后兼容性**

如果在 ``interaction_kwargs`` 中未提供 ``name`` 字段，系统将默认使用 ``"gsm8k"`` 以保持向后兼容性。

最佳实践
--------------

**资源管理**

- 在 ``finalize_interaction()`` 中始终实现适当的清理
- 使用唯一的实例 ID 以避免在并发训练中发生冲突
- 处理边缘情况，例如空消息或格式错误的内容

**性能优化**

```rst
- 保持交互逻辑轻量，以避免阻塞训练
- 正确使用 async/await 以维持非阻塞行为
- 考虑在交互实例中缓存耗时的计算
```

**Testing**

全面测试对于交互系统至关重要：

.. code-block:: python

```python
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_interaction_workflow():
    interaction = YourInteraction({})
    
    # 测试完整的工作流程
    instance_id = await interaction.start_interaction(ground_truth="expected_answer")
    
    messages = [{"role": "user", "content": "user_response"}]
    should_terminate, response, reward, metadata = await interaction.generate_response(instance_id, messages)
    
    assert should_terminate in [True, False]
    assert isinstance(reward, float)
    
    await interaction.finalize_interaction(instance_id)

高级用法
--------------

**多交互训练策略**

您可以使用多个交互设计复杂的训练场景：
```

.. code-block:: python

# 示例：不同交互代理的渐进式难度
```python
class MathTrainingPipeline:
    def create_interaction_config(self):
        return {
            "interaction": [
                {
                    "name": "basic_math",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {"difficulty": "easy"}
                },
                {
                    "name": "advanced_math", 
                    "class_name": "custom.interactions.AdvancedMathInteraction",
                    "config": {"difficulty": "hard", "allow_hints": True}
                },
                {
                    "name": "competition_math",
                    "class_name": "custom.interactions.CompetitionMathInteraction", 
                    "config": {"time_limit": 300, "show_steps": False}
                }
            ]
        }

    def create_curriculum_data(self, epoch):
        if epoch < 5:
            return [{"name": "basic_math", ...} for _ in samples]
        elif epoch < 10:
            return [{"name": "advanced_math", ...} for _ in samples]
        else:
            return [{"name": "competition_math", ...} for _ in samples]
```

**自定义评分函数**

您可以集成自定义奖励函数：

.. code-block:: python

```python
async def calculate_score(self, instance_id, **kwargs):
        response = self._instance_dict[instance_id]["response"]
        ground_truth = self._instance_dict[instance_id]["ground_truth"]
        
        # 自定义评估逻辑
        if custom_evaluation_function(response, ground_truth):
            return 1.0
        else:
            return 0.0
```

**多步骤交互**

对于需要多轮反馈的复杂场景：

.. code-block:: python

```python
async def generate_response(self, instance_id, messages, **kwargs):
        instance = self._instance_dict[instance_id]
        instance["attempts"] += 1
        
        # 评估当前响应
        reward = await self.calculate_score(instance_id)
        
        if reward > 0.8:
            return True, "优秀的工作！", reward, {}
        elif instance["attempts"] < 3:
            return False, "不错的尝试，但请努力改进...", reward, {}
        else:
            return True, "已达到最大尝试次数。", reward, {}
```

故障排除
---------------

**常见问题**

1. **实例 ID 冲突**: 确保在并发会话中实例 ID 唯一
2. **内存泄漏**: 始终调用 ``finalize_interaction()`` 来清理资源
3. **阻塞操作**: 保持交互逻辑异步且非阻塞
4. **配置错误**: 验证交互配置路径和类名是否正确
5. **交互名称冲突**: 确保所有交互在配置中具有唯一名称
6. **缺失交互**: 验证 ``interaction_kwargs`` 中的 ``name`` 字段是否与可用交互匹配
7. **向后兼容性**: 从单一交互迁移到多重交互时，为现有数据添加 ``name`` 字段

**Debugging**

启用调试日志以跟踪交互流程：

.. code-block:: bash

```bash
export VERL_LOGGING_LEVEL=调试
```

**性能监控**

监控交互性能对训练吞吐量的影响并进行相应调整。

相关文档
--------------------

- :doc:`multiturn`: 基本的多轮回合配置
- :doc:`sandbox_fusion`: 与 SGLang 的工具集成
- :doc:`search_tool_example`: 搜索工具实现示例