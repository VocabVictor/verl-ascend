# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional

from verl.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'verl.cli.pt',
    'sft': 'verl.cli.sft',
    'infer': 'verl.cli.infer',
    'merge-lora': 'verl.cli.merge_lora',
    'web-ui': 'verl.cli.web_ui',
    'deploy': 'verl.cli.deploy',
    'rollout': 'verl.cli.rollout',
    'rlhf': 'verl.cli.rlhf',
    'sample': 'verl.cli.sample',
    'export': 'verl.cli.export',
    'eval': 'verl.cli.eval',
    'app': 'verl.cli.app',
}


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def _compat_web_ui(argv):
    # [compat]
    method_name = argv[0]
    if method_name in {'web-ui', 'web_ui'} and ('--model' in argv or '--adapters' in argv or '--ckpt_dir' in argv):
        argv[0] = 'app'
        logger.warning('Please use `verl app`.')


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    
    # Handle no arguments or help
    if not argv or argv[0] in ['-h', '--help', 'help']:
        print("Usage: verl <command> [options]")
        print("\nAvailable commands:")
        for cmd in sorted(route_mapping.keys()):
            print(f"  {cmd}")
        print("\nFor help with a specific command, use: verl <command> --help")
        sys.exit(0)
    
    _compat_web_ui(argv)
    method_name = argv[0].replace('_', '-')
    
    if method_name not in route_mapping:
        print(f"Error: Unknown command '{method_name}'")
        print(f"Available commands: {', '.join(sorted(route_mapping.keys()))}")
        sys.exit(1)
    
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    torchrun_args = get_torchrun_args()
    python_cmd = sys.executable
    if torchrun_args is None or method_name not in {'pt', 'sft', 'rlhf', 'infer'}:
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
