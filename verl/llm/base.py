# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from abc import ABC, abstractmethod
from typing import List, Union

import verl
from verl.utils import get_logger, parse_args, seed_everything
from .argument import BaseArguments
from .utils import ProcessorMixin

logger = get_logger()


class VerlPipeline(ABC, ProcessorMixin):
    args_class = BaseArguments

    def __init__(self, args: Union[List[str], args_class, None] = None):
        self.args = self._parse_args(args)
        args = self.args
        if hasattr(args, 'seed'):
            seed = args.seed + max(getattr(args, 'rank', -1), 0)
            seed_everything(seed)
        logger.info(f'args: {args}')
        self._compat_dsw_gradio(args)

    def _parse_args(self, args: Union[List[str], args_class, None] = None) -> args_class:
        if isinstance(args, self.args_class):
            return args
        assert self.args_class is not None
        args, remaining_argv = parse_args(self.args_class, args)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return args

    @staticmethod
    def _compat_dsw_gradio(args) -> None:
        from verl.llm import WebUIArguments, AppArguments
        if (isinstance(args, (WebUIArguments, AppArguments)) and 'JUPYTER_NAME' in os.environ
                and 'dsw-' in os.environ['JUPYTER_NAME'] and 'GRADIO_ROOT_PATH' not in os.environ):
            os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.server_port}"

    def main(self):
        logger.info(f'Start time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        logger.info(f'verl.__version__: {verl.__version__}')
        result = self.run()
        logger.info(f'End time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        return result

    @abstractmethod
    def run(self):
        pass
