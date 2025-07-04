===============================
沙盒融合工具集成
===============================

最后更新：2025年6月10日。

动机
===========

作为verl的用户，我们希望在Actor回放期间允许模型调用某些工具，将结果纳入训练过程。  
一位来自字节跳动的同事提出了一篇论文，旨在通过代码执行工具增强模型能力。  
我们的目标是支持推理引擎的工具调用能力，使用`sandbox-fusion`作为代码执行系统，为社区提供`retools`的重新实现。

使用Sandbox Fusion + FaaS集成进行奖励计算
=====================================================

- 在当前的数据集和任务中，已经存在类似的工作（例如，Prime），该工作使用本地进程作为运行器来执行模型生成的代码以进行奖励计算。  
- 在此基础上，#1429通过将FaaS集成作为奖励计算的运行器，推动了设计的进展。

目标
=====

- 适应 `sglang` 工具调用协议并为沙箱融合定义工具。
- 与 `async-rollout` 过程集成，确保沙箱融合工具遵循 asyncIO 约定。
- 设计并实现一个基本的速率限制器，以防止诸如 429 错误等问题。

非目标
=========

- 训练效果不在讨论范围内。
- 不考虑可观察性指标。
- 不涉及分布式故障转移和组件容错。

设计细节
==============

工具架构定义
----------------------

- 目前，仅考虑代码执行，因此模型中的 JSON 需要一个 `code` 字段。
- 目前仅支持 Python 代码，因此未定义 `language` 参数。

.. code-block:: python

```python
OpenAIFunctionToolSchema(
       type="function",
       function=OpenAIFunctionSchema(
           name="code_interpreter",
           description="一个用于执行代码的工具。",
           parameters=OpenAIFunctionParametersSchema(
               type="object",
               properties={
                   "code": OpenAIFunctionPropertySchema(
                       type="string",
                       description="要执行的代码。",
                       enum=None,
                   )
               },
               required=["code"],
           ),
           strict=False,
       )
   )

配置参数
--------------------------
```

+----------------------------+--------------------------------------------------------------+
| 参数名称                   | 描述                                                         |
+============================+==============================================================+
| `num_workers`              | 每个数据处理器（DP）请求运行器的工作线程/进程数量。         |
+----------------------------+--------------------------------------------------------------+
| `rate_limit`               | 并发代码执行的全局限制。默认值：10                          |
+----------------------------+--------------------------------------------------------------+
| `default_timeout`          | 每次代码执行的超时时间（以秒为单位）。默认值：30            |
+----------------------------+--------------------------------------------------------------+
| `default_language`         | 默认编程语言。默认值："python"                              |
+----------------------------+--------------------------------------------------------------+
| `enable_global_rate_limit` | 是否启用全局速率限制。默认值：True                        |
+----------------------------+--------------------------------------------------------------+
| `sandbox_fusion_url`       | veFaas沙箱执行服务的URL                                     |
+----------------------------+--------------------------------------------------------------+

速率限制设计
-----------------------

Objective:

- 使用令牌桶模型限制并发请求的数量。

- 确保按顺序提交给代码执行器，以避免因退避而导致的饥饿现象。

设计亮点：

- 使用 Ray 全局 Actor 作为集群级别的单例分布式计数器。

- 使用信号量(Semaphore)进行计数，`acquire` 和 `release` 在不同的线程池中执行，以保持顺序。

- 使用 Ray 的 cloud-pickle 对函数进行序列化，以解耦 `ExecutionWorker`。

.. code-block:: python

```python
@ray.remote(concurrency_groups={"acquire": 1,"release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count

class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)
```

```python
def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
           with ExitStack() as stack:
               stack.callback(self.rate_limit_worker.release.remote)
               ray.get(self.rate_limit_worker.acquire.remote())
               try:
                   return fn(*fn_args, **fn_kwargs)
               except Exception as e:
                   logger.warning(f"执行代码时出错: {e}")

   def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode=PoolMode.ThreadMode):
       if mode == PoolMode.ThreadMode:
           return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(
               enable_global_rate_limit=enable_global_rate_limit,
               rate_limit=rate_limit
           )
       else:
           raise NotImplementedError("进程模式尚未实现")

工具实现
-------------------

- 使用 `instance_id` 来识别跨多个对话轮次的请求。
  
- 使用 `execution_pool` 来实现异步调用。
  
- 在回滚完成后清理状态。
```

.. code-block:: python

```python
class SandboxFusionTool(BaseTool):
       def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
           ...
           self.execution_pool = init_execution_pool(...)
           ...

       async def create(self, instance_id: Optional[str] = None, ...):
           ...

       async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
            code = parameters.get("code", "")
            timeout = parameters.get("timeout", self.default_timeout)
            language = parameters.get("language", self.default_language)
            if not isinstance(code, str):
                code = str(code)

            result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
            self._instance_dict[instance_id]["reward"].append(result.strip())

            return result, result, {}
```

```python
def execute_code(self, instance_id, code, timeout=30, language="python"):
            result_status, metadata  = _process_single_case(0, None, None, self.sandbox_fusion_url, code, timeout, language)
            # 我们应该始终预期这一点，因为我们没有正确的答案
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                return actual_output
            else:
                return "这里没有标准输出"

       async def calc_reward(self, instance_id: str, ...):
           ...

       async def release(self, instance_id: str, ...):
           ...

测试计划
=========

单元测试
---------- 
```

- **test_tools_registration**: 测试工具注册和初始化。
- **test_rollout_req_creation**: 验证 `AsyncRolloutReq` 是否正确构建。
- **test_over_size_case**: 确保在超过 `max_seq_len` 时，回滚提前终止。
- **test_tool_call_basic_case**: 模拟 `sglang` 输出，验证工具调用及结果。
- **test_tool_call_batch_case**: 测试工具调用的批处理。
- **test_basic_multi_process_init**: 验证 Ray 全局 Actor 的行为是否如单例。
- **TestSingleNodeRateLimiterCase**: 验证速率限制器在单节点模式下的工作情况。
- **test_rotten_execution**: 确保速率限制器能够从函数错误中恢复。
- **TestMultiNodeRateLimiterCase**: 验证多节点环境下的行为。

e2e 测试
----------
我们在 `tests/special_e2e` 文件夹中提供了 e2e 测试脚本，名为 `tests/special_e2e/run_gsm8k_fsdp_sgl_multiturn_sf_tool.sh`

通过设置 'trainer.rollout_data_dir'，您可以将回滚数据转储到本地磁盘。以下是从回滚数据中提取的示例：

.. code-block:: python

```json
{
     "input": "
     
     system\n你是一个数学专家。你会得到一个问题，并需要逐步解决它。在调用任何工具之前，逐步推理。在逐步解决问题后，你应该使用 `calc_gsm8k_reward` 工具，至少调用一次，然后在必要时完善你的答案。将你的最终答案放在 `#### <answer>` 的格式中。\n\n# 工具\n\n你可以调用一个或多个函数来协助用户查询。\n\n你可以在 <tools></tools> XML 标签中找到函数签名：\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"执行代码的工具。\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"要执行的代码。\", \"enum\": null}}, \"required\": [\"code\"]}, \"strict\": false}}\n</tools>\n\n对于每个函数调用，返回一个 JSON 对象，包含函数名称和参数，放在 <tool_call></tool_call> XML 标签中：\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n
     
     user\n约翰每年获得相同百分比的奖金。去年他赚了 $100,000，并获得了 $10,000 的奖金。今年他赚了 $200,000。当将他的总工资和奖金加在一起时，约翰今年将赚多少钱？让我们逐步思考，并在 `####` 后输出最终答案。\n
     
     assistant\n",
     "output": "assistant\n<think>\n好的，让我们看看。约翰每年获得相同百分比的奖金。去年，他赚了 $100,000，并获得了 $10,000 的奖金。今年，他赚了 $200,000。问题是问他今年在将总工资和奖金加在一起后将赚多少钱。\n\n首先，我需要计算他获得的奖金百分比。由于奖金每年都是相同的百分比，如果我能根据去年的情况找到这个百分比，我就可以将其应用到今年的收入中，以找到今年的奖金。然后，我会将这个奖金加到他今年的总工资中。\n\n去年，他赚了 $100,000，并获得了 $10,000 的奖金。要找到百分比，我可以将奖金除以他的总收入。所以，10,000 除以 100,000。让我计算一下。\n\n10,000 / 100,000 等于 0.1，也就是 10% 的百分比。所以，他每年的奖金是他工资的 10%。\n\n今年，他赚了 $200,000。所以，今年的奖金将是 200,000 的 10%。让我使用代码解释器计算一下，以确保准确性。\n
     
     <tool_call>\n{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"total_pay_this_year = 200000\\nbonus_percentage = 10 / 100\\nbonus_this_year = total_pay_this_year * bonus_percentage\\ntotal_income_this_year = total_pay_this_year + bonus_this_year\\nprint(total_income_this_year)\", \"executes\": \"True\"}}\n</tool_call>\n
     
     tool\n220000.0\n\n
     
     assistant\n<think>\n所以代码显示，约翰今年的总收入将是 $220,000。这与去年的 $100,000 工资的 10% 奖金一致，即 $10,000。因此，今年的奖金 $20,000 加上他 $200,000 的总工资，结果是总收入 $220,000。\n\n我需要确认没有其他因素在起作用。问题说明每年都是固定百分比的奖金，因此没有提到变化的百分比或其他变量。因此，将 10% 的简单百分比应用于今年的部分工资金额将得到 $20,000。将其加到他的总工资中，得到 $220,000。计算是正确的，并符合问题的参数。\n</think>\n\n#### 220000.0",
     "score": 0,
     "step": 1
}
```

这是可读格式版本：

.. code-block:: python

[系统]
   
   你是一个数学专家。你会收到一个问题，并需要逐步解决它。在调用任何工具之前，逐步推理。在逐步解决问题后，你应该使用 `calc_gsm8k_reward` 工具，至少调用一次，然后在必要时完善你的答案。将你的最终答案放在 `#### <答案>` 的格式中。
   
   # 工具
   
   你可以调用一个或多个函数来协助用户查询。
   
   你在 <tools></tools> XML 标签内提供了函数签名：
   <tools>
   {"type": "function", "function": {"name": "code_interpreter", "description": "执行代码的工具。", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "要执行的代码。", "enum": null}}, "required": ["code"]}, "strict": false}}
   </tools>
   
   对于每个函数调用，返回一个 JSON 对象，包含函数名称和参数，位于 <tool_call></tool_call> XML 标签内：
   <tool_call>
   {"name": <function-name>, "arguments": <args-json-object>}
   </tool_call>
   
   [用户]
   
   约翰每年获得相同百分比的奖金。去年他赚了 $100,000，并获得了 $10,000 的奖金。今年他赚了 $200,000。当将他的总薪水和奖金加在一起时，约翰今年将赚多少钱？让我们逐步思考，并在 `####` 后输出最终答案。
   
   [思考]
   
   好的，让我们看看。约翰每年获得相同百分比的奖金。去年，他赚了 $100,000，并获得了 $10,000 的奖金。今年，他赚了 $200,000。问题是询问他今年在将总薪水和奖金加在一起时会赚多少钱。
   
   首先，我需要计算出他获得的奖金百分比。由于奖金每年都是相同的百分比，如果我能根据去年的细节找到这个百分比，我就可以将其应用到今年的收入上，以找出今年的奖金。然后，我会将这个奖金加到他今年的总薪水上。
   
   去年，他赚了 $100,000，并获得了 $10,000 的奖金。为了找到百分比，我可以将奖金除以他的总收入。所以，$10,000 除以 $100,000。让我计算一下。
   
   $10,000 / $100,000 等于 0.1，作为百分比是 10%。所以，他每年的奖金是他薪水的 10%。
   
   今年，他赚了 $200,000。因此，今年的奖金将是 $200,000 的 10%。让我使用代码解释器计算一下，以确保准确性。
   <tool_call>
   {"name": "code_interpreter", "arguments": {"code": "total_pay_this_year = 200000\nbonus_percentage = 10 / 100\nbonus_this_year = total_pay_this_year * bonus_percentage\ntotal_income_this_year = total_pay_this_year + bonus_this_year\nprint(total_income_this_year)", "executes": "True"}}
   </tool_call>
   
   [工具]
   
   220000.0
   
   [思考]
   
   所以代码显示，约翰今年的总收入将是 $220,000。这与去年的 $100,000 薪水的 10% 奖金 $10,000 一致。因此，今年的奖金 $20,000 加上他 $200,000 的总薪水，结果是 $220,000。
   
   我需要确认没有其他因素在起作用。问题说明每年都是固定百分比的奖金，因此没有提到变化的百分比或其他变量。因此，对今年部分薪水应用 10% 的简单百分比将得到 $20,000。将其加到他的总薪水上，得到 $220,000。计算是正确的，并且与问题的参数一致。
   
   #### 220000.0