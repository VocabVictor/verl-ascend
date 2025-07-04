使用 FSDP 后端添加模型
==================================

最后更新日期：2025年2月9日。

模型
--------------------------

原则上，我们的FSDP后端可以支持任何HF（Hugging Face）模型，并且我们可以使用`hf_weight_loader.py`在`third_party/vllm`下将actor模型的权重与vLLM同步。然而，``hf_weight_loader``在同步过程中将收集模型的完整state_dict，这可能会导致OOM（Out of Memory）。我们建议使用``dtensor_weight_loader``，它会逐层收集完整的模型参数，以减少内存峰值使用量。我们已经在`third_party/vllm`下的`dtensor_weight_loader.py`中为以下模型支持了dtensor weight loader：

- ``GPT2LMHeadModel``
- ``LlamaForCausalLM``
- ``LLaMAForCausalLM``
- ``MistralForCausalLM``
- ``InternLMForCausalLM``
- ``AquilaModel``
- ``AquilaForCausalLM``
- ``Phi3ForCausalLM``
- ``GemmaForCausalLM``
- ``Gemma2ForCausalLM``
- ``GPTBigCodeForCausalLM``
- ``Starcoder2ForCausalLM``
- ``Qwen2ForCausalLM``
- ``DeepseekV2ForCausalLM``

要实现一个在vLLM中受支持的模型的``dtensor_weight_loader``，请按照下面的gemma模型指南操作：

1. 将vllm模型类中的``load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])``复制到``dtensor_weight_loaders.py``中。
2. 修改参数为``(actor_weights: Dict, vllm_model: nn.Module)``
3. 将``self``替换为``vllm_model``
4. 在每个``param = params_dict[name]``之前添加``local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)``，并使用``local_loaded_weight``修改后续的权重加载。
5. 将实现的dtensor权重加载器注册到``__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__``中。

.. code-block:: diff

```python
def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    loaded_params = set()
    for name, loaded_weight in actor_weights.items():
        for (param_name, shard_name, shard_id) in stacked_params_mapping:
            if shard_name not in name:
                continue
            name = name.replace(shard_name, param_name)
            # 跳过为 GPTQ 模型加载额外偏置。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # 在 vllm 中不使用 lm_head，因为它与 embed_token 绑定。
            # 为防止错误，跳过加载 lm_head.weight。
            if "lm_head.weight" in name:
                continue
            # 跳过为 GPTQ 模型加载额外偏置。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))
        loaded_params.add(name)
    unloaded_params = params_dict.keys() - loaded_params
    if unloaded_params:
        raise RuntimeError(
            "从检查点中未初始化某些权重："
            f"{unloaded_params}")
```