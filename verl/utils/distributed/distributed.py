# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for distributed training."""

import os
import socket
from typing import Tuple

import torch.distributed

from .device import get_nccl_backend, get_torch_device


def find_free_port(start_port: int = 29500) -> int:
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + 100}")


def get_dist_setting() -> Tuple[int, int, int, int]:
    """Get distributed training settings with automatic detection.
    
    Returns:
        Tuple of (rank, local_rank, world_size, local_world_size)
        If environment variables are not set, returns (-1, -1, 1, 1)
    """
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_dist():
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


def set_default_ddp_config():
    """Set default distributed training configuration for single GPU fallback.
    
    This function automatically sets distributed environment variables
    when they are not present, enabling seamless single GPU training.
    """
    rank, local_rank, _, _ = get_dist_setting()
    if rank == -1 or local_rank == -1:
        # Auto-configure for single GPU training
        os.environ['NPROC_PER_NODE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        
        # Find a free port if MASTER_PORT not set
        if 'MASTER_PORT' not in os.environ:
            free_port = find_free_port()
            os.environ['MASTER_PORT'] = str(free_port)


def initialize_global_process_group(timeout_second=36000):
    """Initialize distributed process group with automatic environment detection.
    
    This function will automatically set default values for single GPU training
    if distributed environment variables are not set.
    """
    from datetime import timedelta

    # Set default configuration if not in distributed environment
    set_default_ddp_config()
    
    # Get final settings after potential auto-configuration
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    
    # Initialize process group only if not already initialized
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            get_nccl_backend(),
            timeout=timedelta(seconds=timeout_second),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    
    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
