添加使用 Megatron-LM 后端的模型
=========================================

最后更新：2025年04月25日。

模型
-----------

如果使用最新的版本，我们对Megatron后端直接支持``GPTModel``。您可以使用类似的方式来预训练自定义模型。我们在这里列出了步骤：

1. 找到 `model_initializer.py <https://github.com/volcengine/verl/blob/main/verl/models/mcore/model_initializer.py>`_
2. 如果您的模型可以通过``TransformerLayerSpec``进行配置，您可以直接使用``GPTModel``。否则，请在此处实现一个新的``ModelLayerSpec``和``ModelLayer``。
3. 使用正确的``LayerSpec``、``TransformerConfig``和``HuggingfaceConfig``作为参数来初始化GPTModel。
4. 最后返回模型。