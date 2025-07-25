# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Optional

import torch
from torch import nn

from verl.llm import MODEL_ARCH_MAPPING, HfConfigFactory, ModelKeys
from verl.utils.logger import get_logger
from .utils import ActivationMixin, VerlAdapter, VerlConfig, VerlOutput

logger = get_logger()


@dataclass
class LLaMAProConfig(VerlConfig):
    """
    The configuration class for the LLaMAPro module.

    See https://arxiv.org/abs/2401.02415

    Args:
        model_type(`str`): LLaMAPro only support parts of the LLM models because of the variables need to be manually
            modified.
        num_new_blocks(`int`): How many new blocks need to be added
        num_groups(`int`): The groups of new blocks are split to. Default equals to `num_new_blocks` which means each
            single layer will be inserted into every `num_hidden_layers/num_new_blocks` original layers.
    """
    model_type: str = field(
        default=None, metadata={
            'choices': list(MODEL_ARCH_MAPPING.keys()),
        })

    num_new_blocks: int = None

    num_groups: Optional[int] = None

    def __post_init__(self):
        from .mapping import VerlTuners
        self.verl_type = VerlTuners.LLAMAPRO


class LLaMAPro(VerlAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: LLaMAProConfig, adapter_name: str) -> VerlOutput:
        """Prepare a model with `LLaMAProConfig`"""
        num_hidden_layers = HfConfigFactory.get_config_attr(model.config, 'num_hidden_layers')
        if num_hidden_layers is None:
            num_hidden_layers = HfConfigFactory.get_config_attr(model.config, 'num_layers')
        assert num_hidden_layers is not None, 'Cannot find num of layers config'
        assert num_hidden_layers % config.num_new_blocks == 0, f'Model layers {num_hidden_layers} ' \
                                                               f'should be divided by {config.num_new_blocks}'
        if config.num_groups is None:
            config.num_groups = config.num_new_blocks

        # the except block will change the model_type, this will cause `model not found` error
        # when using internvl
        origin_model_type = config.model_type
        model_type = origin_model_type
        num_stride = num_hidden_layers // config.num_groups
        try:
            module_list = LLaMAPro._find_module_list(config, model)
        except AssertionError as e:
            model_type = LLaMAPro.search_correct_model_type(model)
            if model_type is None:
                language_model_name = VerlAdapter.get_model_key_mapping(config.model_type, config).language_model
                if language_model_name:
                    if isinstance(language_model_name, str):
                        language_model_name = [language_model_name]
                    language_model = model.get_submodule(language_model_name[0])
                    model_type = LLaMAPro.search_correct_model_type(language_model)
                    if model_type:
                        model = language_model

            if model_type:
                config.model_type = model_type
                module_list = LLaMAPro._find_module_list(config, model)
            else:
                raise e

        new_module_list = nn.ModuleList()
        new_module_idx = []
        for idx, module in enumerate(module_list):
            new_module_list.append(module)
            if (idx + 1) % num_stride == 0:
                new_module = deepcopy(module)
                ActivationMixin.mark_all_sub_modules_as_plugin(new_module)
                new_module_list.append(new_module)
                new_module_idx.append(idx + 1 + len(new_module_idx))

        LLaMAPro._update_module_weight(config, new_module_list, new_module_idx)
        LLaMAPro._update_module_attr(config, new_module_list)
        model.config.num_hidden_layers = len(new_module_list)
        LLaMAPro._set_module_list(config, model, new_module_list)

        def activate_module(activate: bool):
            if activate:
                LLaMAPro._update_module_attr(config, new_module_list)
                LLaMAPro._set_module_list(config, model, new_module_list)
            else:
                LLaMAPro._update_module_attr(config, module_list)
                LLaMAPro._set_module_list(config, model, module_list)

        def state_dict_callback(state_dict, adapter_name, **kwargs):
            model_key_mapping = LLaMAPro.get_model_key_mapping(model_type, config)
            new_module_list = [model_key_mapping.module_list + f'.{i}' for i in new_module_idx]
            return {
                key: value
                for key, value in state_dict.items() if any([m_part in key for m_part in new_module_list])
            }

        def mark_trainable_callback(model):
            model_key_mapping = LLaMAPro.get_model_key_mapping(model_type, config)
            new_module_list = [model_key_mapping.module_list + f'.{i}' for i in new_module_idx]
            for name, parameter in model.named_parameters():
                parameter: nn.Parameter
                if any([m_part in name for m_part in new_module_list]):
                    parameter.requires_grad = True

        config.model_type = origin_model_type
        model.activate_module = activate_module
        return VerlOutput(
            config=config, state_dict_callback=state_dict_callback, mark_trainable_callback=mark_trainable_callback)

    @staticmethod
    def _update_module_attr(config: LLaMAProConfig, module_list):
        model_type = config.model_type
        model_key_mapping = LLaMAPro.get_model_key_mapping(model_type, config)
        attention = model_key_mapping.attention
        attention = attention.split('{}.')[1]
        if model_type == 'phi3-small':
            raise ValueError('phi3-small does not support llamapro currently')
        if model_type in ('llama', 'mistral', 'qwen2', 'yi', 'gemma', 'deepseek', 'openbuddy', 'xverse', 'orion',
                          'bluelm', 'ziya', 'skywork', 'deepseek-v2', 'minicpm', 'phi3', 'internlm2'):
            for idx, module in enumerate(module_list):
                try:
                    getattr(module, attention).layer_idx = idx
                except AttributeError:
                    getattr(module, 'cross_attn').layer_idx = idx
        elif model_type in ('chatglm', 'glm4'):
            for idx, module in enumerate(module_list):
                getattr(module, attention).layer_number = idx
        elif model_type in ('phi2', ):
            for idx, module in enumerate(module_list):
                getattr(module, attention).block_idx = idx
        else:
            for idx, module in enumerate(module_list):
                attrs = [
                    attr for attr in dir(getattr(module_list[0], attention))
                    if attr in ('layer_idx', 'layer_number', 'block_idx')
                ]
                assert len(attrs) <= 1
                if attrs:
                    setattr(getattr(module, attention), attrs[0], idx)
                else:
                    logger.warn(f'model_type: {model_type} seems has no layer_idx, if you encountered anything wrong,'
                                f'please give us a feedback.')

    @classmethod
    def get_model_key_mapping(cls, model_type, config) -> ModelKeys:

        model_key_mapping = VerlAdapter.get_model_key_mapping(model_type, config)
        assert model_key_mapping.o_proj is not None and model_key_mapping.down_proj is not None, \
            'LLaMAPro only support models with o_proj and down_proj components.'
        return model_key_mapping

    @classmethod
    def search_correct_model_type(cls, module: nn.Module):
        for arch_name, arch_type in MODEL_ARCH_MAPPING.items():
            arch_type: ModelKeys
            if getattr(arch_type, 'module_list') is None:
                # Need to be a LLM arch
                continue

            matched = True
            for f in fields(arch_type):
                arch_str = getattr(arch_type, f.name)
                if f.name == 'arch_name' or arch_str is None:
                    continue

                arch_str = arch_str.replace('{}', '0')
                try:
                    sub_module = module.get_submodule(arch_str)
                    if sub_module is None:
                        matched = False
                except AttributeError:
                    matched = False

                if not matched:
                    break

            if matched:
                return arch_name

    @staticmethod
    def _update_module_weight(config: LLaMAProConfig, module_list, new_module_idx):
        model_key_mapping = LLaMAPro.get_model_key_mapping(config.model_type, config)
        o_proj = model_key_mapping.o_proj.split('{}.')[1]
        down_proj = model_key_mapping.down_proj.split('{}.')[1]

        for idx, module in enumerate(module_list):
            if idx not in new_module_idx:
                continue
            _o_proj: nn.Linear = module.get_submodule(o_proj)
            _down_proj: nn.Linear = module.get_submodule(down_proj)
            _o_proj.weight.data = torch.zeros_like(_o_proj.weight.data)
            _down_proj.weight.data = torch.zeros_like(_down_proj.weight.data)
            if hasattr(_o_proj, 'bias') and _o_proj.bias is not None:
                _o_proj.bias.data = torch.zeros_like(_o_proj.bias)
            if hasattr(_down_proj, 'bias') and _down_proj.bias is not None:
                _down_proj.bias.data = torch.zeros_like(_down_proj.bias)

    @staticmethod
    def _set_module_list(config, module: nn.Module, module_list: nn.ModuleList):
        model_key_mapping = LLaMAPro.get_model_key_mapping(config.model_type, config)
        idx = model_key_mapping.module_list.rfind('.')
        parent = module.get_submodule(model_key_mapping.module_list[:idx])
        setattr(parent, model_key_mapping.module_list[idx + 1:], module_list)

    @staticmethod
    def _find_module_list(config, module: nn.Module) -> nn.ModuleList:
        model_key_mapping = LLaMAPro.get_model_key_mapping(config.model_type, config)
        return module.get_submodule(model_key_mapping.module_list)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        module.activate_module(activate)

    @staticmethod
    def has_additional_modules():
        return True
