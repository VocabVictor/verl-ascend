import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            # ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            target_modules=['all-linear', 'all-embedding'],
            modules_to_save=['all-embedding', 'all-norm'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_unsloth():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            max_steps=5,
            tuner_backend='unsloth',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(TrainArguments(resume_from_checkpoint=last_model_checkpoint, load_data_args=True, max_steps=10))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_mllm_mp():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20'],
            # dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            split_dataset_ratio=0.01,
            train_type='lora',
            target_modules=['all-linear'],
            freeze_aligner=False,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_llm_streaming():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct', dataset=['verl/chinese-c4'], streaming=True, max_steps=16, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_streaming():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation', 'AI-ModelScope/alpaca-gpt4-data-en'],
            streaming=True,
            max_steps=16,
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],  #
            split_dataset_ratio=0.01,
            deepspeed='zero3',
            **kwargs))


def test_qwen_vl():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen-VL-Chat',
            dataset=['AI-ModelScope/LaTeX_OCR#40', 'modelscope/coco_2014_caption:validation#40'],
            split_dataset_ratio=0.01,
            **kwargs))


def test_qwen2_audio():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-Audio-7B-Instruct',
            dataset=['speech_asr/speech_asr_aishell1_trainsets:validation#200'],
            split_dataset_ratio=0.01,
            freeze_parameters_ratio=1,
            trainable_parameters=['audio_tower'],
            train_type='full',
            **kwargs))


def test_llm_gptq():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_llm_awq():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct-AWQ',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_mllm_streaming_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation', 'AI-ModelScope/alpaca-gpt4-data-en'],
            streaming=True,
            max_steps=16,
            deepspeed='zero3',
            **kwargs))


def test_mllm_streaming_mp_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation', 'AI-ModelScope/alpaca-gpt4-data-en'],
            streaming=True,
            max_steps=16,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            **kwargs))


def test_llm_hqq():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            quant_method='hqq',
            quant_bits=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_llm_bnb():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            quant_method='bnb',
            quant_bits=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_moe():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_resume_from_checkpoint():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            max_steps=5,
            streaming=True,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(
        TrainArguments(
            resume_from_checkpoint=last_model_checkpoint,
            streaming=True,
            load_data_args=True,
            max_steps=10,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_resume_only_model():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#10', 'AI-ModelScope/alpaca-gpt4-data-en#10'],
            split_dataset_ratio=0.01,
            max_steps=20,
            save_only_model=True,
            deepspeed='zero3',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(
        TrainArguments(
            resume_from_checkpoint=last_model_checkpoint, load_data_args=True, max_steps=20, resume_only_model=True))


def test_llm_transformers_4_33():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen-7B-Chat',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            **kwargs))


def test_predict_with_generate():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    # 'modelscope/coco_2014_caption:validation#100',
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-en#400'],
            predict_with_generate=True,
            # padding_free=True,
            max_length=512,
            packing=True,
            attn_impl='flash_attn',
            split_dataset_ratio=0.01,
            **kwargs))


def test_predict_with_generate_zero3():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    # 'modelscope/coco_2014_caption:validation#100',
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['AI-ModelScope/LaTeX_OCR#40'],
            split_dataset_ratio=0.01,
            predict_with_generate=True,
            freeze_vit=False,
            split_dataset_ratio=0.5,
            deepspeed='zero3',
            **kwargs))


def test_template():
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    global kwargs
    kwargs = kwargs.copy()
    kwargs['num_train_epochs'] = 3
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['verl/self-cognition#200'],
            split_dataset_ratio=0.01,
            model_name=['小黄'],
            model_author=['verl'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_emu3_gen():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['max_position_embeddings'] = '10240'
    os.environ['image_area'] = '518400'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    kwargs['num_train_epochs'] = 100
    result = sft_main(
        TrainArguments(model='BAAI/Emu3-Gen', dataset=['verl/TextCaps#2'], split_dataset_ratio=0.01, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    args = InferArguments(
        ckpt_dir=last_model_checkpoint,
        infer_backend='pt',
        stream=False,
        use_chat_template=False,
        top_k=2048,
        max_new_tokens=40960)
    infer_main(args)


def test_eval_strategy():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            eval_strategy='no',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_epoch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments

    train_kwargs = kwargs.copy()
    train_kwargs['num_train_epochs'] = 3
    # train_kwargs['save_steps'] = 2  # not use
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#50', 'AI-ModelScope/alpaca-gpt4-data-en#50'],
            split_dataset_ratio=0.01,
            save_strategy='epoch',
            **train_kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_agent():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments

    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['verl/ToolBench#500'],
            split_dataset_ratio=0.01,
            loss_scale='react',
            agent_template='toolbench',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_grounding():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from verl.llm import sft_main, TrainArguments, infer_main, InferArguments

    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['AI-ModelScope/coco#200'],
            split_dataset_ratio=0.01,
            dataset_num_proc=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, stream=True, max_new_tokens=2048))


if __name__ == '__main__':
    # test_llm_ddp()
    # test_mllm_mp()
    # test_llm_streaming()
    # test_mllm_streaming()
    # test_mllm_zero3()
    # test_llm_gptq()
    # test_llm_awq()
    # test_mllm_streaming_zero3()
    # test_mllm_streaming_mp_ddp()
    # test_llm_bnb()
    # test_llm_hqq()
    # test_moe()
    # test_resume_from_checkpoint()
    # test_resume_only_model()
    # test_llm_transformers_4_33()
    # test_predict_with_generate()
    # test_predict_with_generate_zero3()
    # test_template()
    # test_qwen_vl()
    # test_qwen2_audio()
    # test_emu3_gen()
    # test_unsloth()
    # test_eval_strategy()
    # test_epoch()
    # test_agent()
    test_grounding()
