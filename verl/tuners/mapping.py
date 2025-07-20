# Copyright (c) Alibaba, Inc. and its affiliates.

from .adapter import Adapter, AdapterConfig
from .llamapro import LLaMAPro, LLaMAProConfig
from .longlora.longlora import LongLoRA, LongLoRAConfig
from .lora import LoRA, LoRAConfig
from .neftune import NEFTune, NEFTuneConfig
from .part import Part, PartConfig
from .prompt import Prompt, PromptConfig
from .reft import Reft, ReftConfig
from .restuning import ResTuning, ResTuningConfig
from .scetuning.scetuning import SCETuning, SCETuningConfig
from .side import Side, SideConfig


class VerlTuners:
    ADAPTER = 'ADAPTER'
    PROMPT = 'PROMPT'
    LORA = 'LORA'
    SIDE = 'SIDE'
    RESTUNING = 'RESTUNING'
    LONGLORA = 'longlora'
    NEFTUNE = 'neftune'
    LLAMAPRO = 'LLAMAPRO'
    SCETUNING = 'SCETuning'
    PART = 'part'
    REFT = 'reft'


VERL_MAPPING = {
    VerlTuners.ADAPTER: (AdapterConfig, Adapter),
    VerlTuners.PROMPT: (PromptConfig, Prompt),
    VerlTuners.LORA: (LoRAConfig, LoRA),
    VerlTuners.SIDE: (SideConfig, Side),
    VerlTuners.RESTUNING: (ResTuningConfig, ResTuning),
    VerlTuners.LONGLORA: (LongLoRAConfig, LongLoRA),
    VerlTuners.NEFTUNE: (NEFTuneConfig, NEFTune),
    VerlTuners.SCETUNING: (SCETuningConfig, SCETuning),
    VerlTuners.LLAMAPRO: (LLaMAProConfig, LLaMAPro),
    VerlTuners.PART: (PartConfig, Part),
    VerlTuners.REFT: (ReftConfig, Reft),
}
