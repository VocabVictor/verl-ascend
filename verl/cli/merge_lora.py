# Copyright (c) Alibaba, Inc. and its affiliates.
from verl.llm import ExportArguments, VerlPipeline, merge_lora


class VerlMergeLoRA(VerlPipeline):
    args_class = ExportArguments
    args: args_class

    def run(self):
        merge_lora(self.args)


if __name__ == '__main__':
    VerlMergeLoRA().main()
