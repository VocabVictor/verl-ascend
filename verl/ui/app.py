# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import List, Union

import gradio as gr
from packaging import version
from transformers.utils import strtobool

import verl
from verl.llm import (DeployArguments, EvalArguments, ExportArguments, RLHFArguments, SamplingArguments, VerlPipeline,
                       WebUIArguments)
from verl.ui.llm_eval.llm_eval import LLMEval
from verl.ui.llm_export.llm_export import LLMExport
from verl.ui.llm_grpo.llm_grpo import LLMGRPO
from verl.ui.llm_infer.llm_infer import LLMInfer
from verl.ui.llm_rlhf.llm_rlhf import LLMRLHF
from verl.ui.llm_sample.llm_sample import LLMSample
from verl.ui.llm_train.llm_train import LLMTrain

locale_dict = {
    'title': {
        'zh': '🚀VERL: 轻量级大模型训练推理框架',
        'en': '🚀VERL: Scalable lightWeight Infrastructure for Fine-Tuning and Inference'
    },
    'sub_title': {
        'zh':
        '请查看 <a href=\"https://github.com/modelscope/ms-verl/tree/main/docs/source\" target=\"_blank\">'
        'VERL 文档</a>来查看更多功能，使用VERL_UI_LANG=en环境变量来切换英文界面',
        'en':
        'Please check <a href=\"https://github.com/modelscope/ms-verl/tree/main/docs/source_en\" target=\"_blank\">'
        'VERL Documentation</a> for more usages, Use VERL_UI_LANG=zh variable to switch to Chinese UI',
    },
    'star_beggar': {
        'zh':
        '喜欢<a href=\"https://github.com/modelscope/ms-verl\" target=\"_blank\">VERL</a>就动动手指给我们加个star吧🥺 ',
        'en':
        'If you like <a href=\"https://github.com/modelscope/ms-verl\" target=\"_blank\">VERL</a>, '
        'please take a few seconds to star us🥺 '
    },
}


class VerlWebUI(VerlPipeline):

    args_class = WebUIArguments
    args: args_class

    def run(self):
        lang = os.environ.get('VERL_UI_LANG') or self.args.lang
        share_env = os.environ.get('WEBUI_SHARE')
        share = strtobool(share_env) if share_env else self.args.share
        server = os.environ.get('WEBUI_SERVER') or self.args.server_name
        port_env = os.environ.get('WEBUI_PORT')
        port = int(port_env) if port_env else self.args.server_port
        LLMTrain.set_lang(lang)
        LLMRLHF.set_lang(lang)
        LLMGRPO.set_lang(lang)
        LLMInfer.set_lang(lang)
        LLMExport.set_lang(lang)
        LLMEval.set_lang(lang)
        LLMSample.set_lang(lang)
        with gr.Blocks(title='VERL WebUI', theme=gr.themes.Base()) as app:
            try:
                _version = verl.__version__
            except AttributeError:
                _version = ''
            gr.HTML(f"<h1><center>{locale_dict['title'][lang]}({_version})</center></h1>")
            gr.HTML(f"<h3><center>{locale_dict['sub_title'][lang]}</center></h3>")
            with gr.Tabs():
                LLMTrain.build_ui(LLMTrain)
                LLMRLHF.build_ui(LLMRLHF)
                LLMGRPO.build_ui(LLMGRPO)
                LLMInfer.build_ui(LLMInfer)
                LLMExport.build_ui(LLMExport)
                LLMEval.build_ui(LLMEval)
                LLMSample.build_ui(LLMSample)

            concurrent = {}
            if version.parse(gr.__version__) < version.parse('4.0.0'):
                concurrent = {'concurrency_count': 5}
            app.load(
                partial(LLMTrain.update_input_model, arg_cls=RLHFArguments),
                inputs=[LLMTrain.element('model')],
                outputs=[LLMTrain.element('train_record')] + list(LLMTrain.valid_elements().values()))
            app.load(
                partial(LLMRLHF.update_input_model, arg_cls=RLHFArguments),
                inputs=[LLMRLHF.element('model')],
                outputs=[LLMRLHF.element('train_record')] + list(LLMRLHF.valid_elements().values()))
            app.load(
                partial(LLMGRPO.update_input_model, arg_cls=RLHFArguments),
                inputs=[LLMGRPO.element('model')],
                outputs=[LLMGRPO.element('train_record')] + list(LLMGRPO.valid_elements().values()))
            app.load(
                partial(LLMInfer.update_input_model, arg_cls=DeployArguments, has_record=False),
                inputs=[LLMInfer.element('model')],
                outputs=list(LLMInfer.valid_elements().values()))
            app.load(
                partial(LLMExport.update_input_model, arg_cls=ExportArguments, has_record=False),
                inputs=[LLMExport.element('model')],
                outputs=list(LLMExport.valid_elements().values()))
            app.load(
                partial(LLMEval.update_input_model, arg_cls=EvalArguments, has_record=False),
                inputs=[LLMEval.element('model')],
                outputs=list(LLMEval.valid_elements().values()))
            app.load(
                partial(LLMSample.update_input_model, arg_cls=SamplingArguments, has_record=False),
                inputs=[LLMSample.element('model')],
                outputs=list(LLMSample.valid_elements().values()))
        app.queue(**concurrent).launch(server_name=server, inbrowser=True, server_port=port, height=800, share=share)


def webui_main(args: Union[List[str], WebUIArguments, None] = None):
    return VerlWebUI(args).main()
