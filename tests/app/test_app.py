def test_llm():
    from verl.llm import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2.5-0.5B-Instruct'))


def test_lora():
    from verl.llm import app_main, AppArguments
    app_main(AppArguments(adapters='verl/test_lora', lang='en', studio_title='小黄'))


def test_mllm():
    from verl.llm import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2-VL-7B-Instruct', stream=True))


def test_audio():
    from verl.llm import AppArguments, app_main, DeployArguments, run_deploy
    deploy_args = DeployArguments(model='Qwen/Qwen2-Audio-7B-Instruct', infer_backend='pt', verbose=False)

    with run_deploy(deploy_args, return_url=True) as url:
        app_main(AppArguments(model='Qwen2-Audio-7B-Instruct', base_url=url, stream=True))


if __name__ == '__main__':
    test_mllm()
