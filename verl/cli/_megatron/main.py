# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

from verl.utils import get_logger
from ..main import cli_main as verl_cli_main

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'verl.cli._megatron.pt',
    'sft': 'verl.cli._megatron.sft',
    'rlhf': 'verl.cli._megatron.rlhf',
}


def cli_main():
    return verl_cli_main(ROUTE_MAPPING)


if __name__ == '__main__':
    cli_main()
