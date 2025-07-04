# 文档翻译工具

这个工具可以将 `docs` 目录下的英文文档自动翻译为中文。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_translate.txt
```

### 2. 配置API密钥

复制模板文件并填入你的API信息：

```bash
cp .env.template .env
```

编辑 `.env` 文件，填入你的实际API密钥：

```bash
# 云雾AI API配置
YUNWU_API_KEY=sk-your-actual-api-key-here
YUNWU_BASE_URL=https://yunwu.ai/v1
YUNWU_MODEL=gpt-3.5-turbo

# 翻译配置
TRANSLATION_SOURCE_LANG=en
TRANSLATION_TARGET_LANG=zh-CN
TRANSLATION_BATCH_SIZE=5

# 文档路径配置
DOCS_SOURCE_PATH=docs
DOCS_OUTPUT_PATH=docs_cn
```

### 3. 运行翻译

```bash
python scripts/translate_docs.py
```

## 功能特性

- ✅ **智能翻译**: 使用大模型进行上下文相关的翻译
- ✅ **格式保持**: 保留Markdown和reStructuredText格式
- ✅ **代码保护**: 自动识别并保护代码块不被翻译
- ✅ **批量处理**: 支持整个文档目录的批量翻译
- ✅ **增量翻译**: 跳过已存在的翻译文件
- ✅ **错误处理**: 网络错误时自动重试

## 支持的文件格式

- `.md` - Markdown文件
- `.rst` - reStructuredText文件

## 翻译质量优化

脚本已针对技术文档翻译进行优化：

1. **专业术语处理**: 保持技术术语的准确性
2. **格式保护**: 保留所有标记格式和链接
3. **上下文理解**: 提供文件名作为翻译上下文
4. **分块翻译**: 将长文档分块处理，提高翻译质量

## 输出结构

翻译后的文档将保存在 `docs_cn` 目录中，保持与原文档相同的目录结构：

```
docs_cn/
├── conf.py              # 中文文档配置
├── Makefile            # 中文文档构建脚本
├── index.rst           # 中文首页
├── _static/            # 静态文件
└── ...                 # 其他翻译文档
```

## 构建中文文档

翻译完成后，可以构建中文文档：

```bash
cd docs_cn
pip install -r ../docs/requirements-docs.txt
make html
```

生成的中文文档将在 `docs_cn/_build/html/` 目录中。

## 注意事项

1. **API限制**: 请注意API调用频率限制，脚本已加入延时
2. **文件大小**: 大文件会自动分块翻译
3. **网络稳定**: 确保网络连接稳定，翻译过程可能需要较长时间
4. **成本控制**: 大量文档翻译会消耗API调用次数，请注意成本

## 故障排除

### 常见错误

1. **API密钥错误**
   ```
   请在 .env 文件中配置 YUNWU_API_KEY
   ```
   解决：检查 `.env` 文件中的API密钥是否正确

2. **网络连接失败**
   ```
   API请求失败: 503 - Service Unavailable
   ```
   解决：检查网络连接和API服务状态

3. **权限错误**
   ```
   Permission denied
   ```
   解决：确保对输出目录有写权限

### 调试模式

可以设置环境变量开启详细日志：

```bash
export TRANSLATION_DEBUG=1
python scripts/translate_docs.py
```

## 自定义配置

可以通过修改 `.env` 文件来自定义翻译行为：

- `TRANSLATION_BATCH_SIZE`: 批处理大小，默认5
- `YUNWU_MODEL`: 使用的模型，默认gpt-3.5-turbo
- `DOCS_OUTPUT_PATH`: 输出目录，默认docs_cn