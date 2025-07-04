#!/usr/bin/env python3
"""
异步版文档翻译脚本 - 使用异步机制和httpx客户端
解决SSL问题和翻译质量问题

新特性：
1. 使用OpenAI客户端和httpx禁用SSL验证
2. 异步翻译机制提高效率
3. 更好的错误处理和重试机制
4. 强制重新翻译不完整的文件
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class AsyncDocumentTranslator:
    def __init__(self):
        """初始化翻译器"""
        # 加载环境变量
        load_dotenv()
        
        self.api_key = os.getenv('YUNWU_API_KEY')
        self.base_url = os.getenv('YUNWU_BASE_URL', 'https://yunwu.ai/v1')
        self.model = os.getenv('YUNWU_MODEL', 'gpt-3.5-turbo')
        
        # 验证API配置
        if not self.api_key or self.api_key == 'your-api-key-here':
            raise ValueError("请在 .env 文件中配置 YUNWU_API_KEY")
        
        # 创建禁用SSL验证的httpx客户端
        self.httpx_client = httpx.AsyncClient(
            verify=False,  # 禁用SSL验证
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # 创建OpenAI客户端
        self.openai_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.httpx_client
        )
        
        # 路径配置
        self.docs_path = Path(project_root) / 'docs'
        self.output_path = Path(project_root) / 'docs_cn'
        self.output_path.mkdir(exist_ok=True)
        
        # 支持的文件扩展名
        self.supported_extensions = {'.md', '.rst'}
        
        # 不需要翻译的文件/目录
        self.skip_patterns = {
            '_build', '__pycache__', '.git', 
            'requirements-docs.txt', 'Makefile', 'conf.py'
        }
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(3)  # 最多3个并发请求

    def should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过此文件"""
        # 检查文件扩展名
        if file_path.suffix not in self.supported_extensions:
            return True
            
        # 检查跳过模式
        for pattern in self.skip_patterns:
            if pattern in str(file_path):
                return True
                
        return False

    def is_already_translated(self, file_path: Path) -> bool:
        """检查文件是否已经翻译"""
        output_file = self.output_path / file_path.relative_to(self.docs_path)
        return output_file.exists()

    def is_content_chinese(self, content: str) -> bool:
        """检测内容是否已经是完整的中文"""
        if len(content) < 50:
            return False
        
        # 计算中文字符比例
        chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        english_chars = sum(1 for char in content if char.isalpha() and not '\u4e00' <= char <= '\u9fff')
        
        # 如果中文字符多于英文字符的70%，认为是中文
        if english_chars == 0:
            return chinese_chars > 10  # 至少有一些中文字符
        
        return chinese_chars > english_chars * 0.7

    def is_translation_incomplete(self, file_path: Path) -> bool:
        """检查翻译是否不完整"""
        output_file = self.output_path / file_path.relative_to(self.docs_path)
        if not output_file.exists():
            return True
            
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否还有大量英文内容
            lines = content.split('\n')
            english_lines = 0
            total_text_lines = 0
            
            for line in lines:
                line = line.strip()
                # 跳过代码块、链接等
                if (not line or line.startswith('```') or line.startswith('http') or 
                    line.startswith('![') or line.startswith('.. ') or 
                    line.startswith('#') and len(line.split()) == 1):
                    continue
                    
                total_text_lines += 1
                # 检查是否主要是英文
                english_chars = sum(1 for char in line if char.isalpha() and not '\u4e00' <= char <= '\u9fff')
                chinese_chars = sum(1 for char in line if '\u4e00' <= char <= '\u9fff')
                
                if english_chars > chinese_chars and english_chars > 10:
                    english_lines += 1
            
            # 如果超过30%的文本行仍是英文，认为翻译不完整
            if total_text_lines > 0:
                english_ratio = english_lines / total_text_lines
                return english_ratio > 0.3
                
            return False
            
        except Exception as e:
            print(f"    ⚠️  检查文件时出错: {e}")
            return True

    async def translate_text(self, text: str, context: str = "", max_retries: int = 3) -> str:
        """使用异步API翻译文本"""
        if not text.strip():
            return text
            
        system_prompt = """你是一个专业的技术文档翻译专家。请将以下英文技术文档翻译成中文，要求：

1. 保持技术术语的准确性
2. 保留原有的格式标记（如Markdown、reStructuredText格式）
3. 保持代码块、链接、图片等不变
4. 翻译要自然流畅，符合中文表达习惯
5. 专业术语可以在首次出现时使用"中文(英文)"的形式
6. 保留所有的标题层级和结构
7. 确保完整翻译，不要遗漏任何文本内容

请直接返回翻译结果，不要添加额外说明。"""

        user_prompt = f"""文档上下文: {context}

需要翻译的内容:
{text}"""

        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    print(f"    API调用中... (尝试 {attempt + 1}/{max_retries})")
                    
                    response = await self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=4000
                    )
                    
                    translated = response.choices[0].message.content.strip()
                    print(f"    ✓ 翻译成功")
                    return translated
                    
            except Exception as e:
                print(f"    ✗ 翻译请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                    
        print(f"    ❌ 翻译失败，返回原文")
        return text

    def split_content(self, content: str) -> List[str]:
        """将内容分割成适合翻译的段落"""
        # 按双换行符分割段落
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # 如果是代码块、链接等特殊内容，单独处理
            if (paragraph.startswith('```') or 
                paragraph.startswith('.. code-block::') or
                paragraph.startswith('http') or
                paragraph.startswith('![') or
                paragraph.startswith('.. image::') or
                len(paragraph.split()) < 3):
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(paragraph)
            else:
                if len(current_chunk) + len(paragraph) > 1200:  # 控制每块大小
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [chunk for chunk in chunks if chunk.strip()]

    async def translate_file(self, file_path: Path, force: bool = False) -> bool:
        """异步翻译单个文件"""
        try:
            print(f"正在处理: {file_path}")
            
            # 检查是否已经翻译（除非强制翻译）
            output_file = self.output_path / file_path.relative_to(self.docs_path)
            
            # 检查翻译是否不完整
            is_incomplete = self.is_translation_incomplete(file_path)
            
            if not force and output_file.exists() and not is_incomplete:
                print(f"  ⏭️  文件已完整翻译，跳过: {output_file}")
                return True
            elif is_incomplete:
                print(f"  🔄 检测到翻译不完整，重新翻译")
            
            # 读取原文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"  📄 文件大小: {len(content)} 字符")
            
            # 检查是否已经是完整的中文内容
            if not force and self.is_content_chinese(content):
                print(f"  🇨🇳 内容已经是完整中文，直接复制")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            # 检查是否主要是代码文件
            code_block_count = content.count('```')
            total_lines = content.count('\n') + 1
            if code_block_count > 0 and code_block_count > total_lines / 3:
                print(f"  💻 主要是代码内容，直接复制")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            # 分割内容进行翻译
            chunks = self.split_content(content)
            print(f"  📝 开始翻译，共 {len(chunks)} 个片段")
            
            # 异步翻译所有片段
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"    [{i+1}/{len(chunks)}] 正在翻译片段...")
                
                # 如果是代码块或特殊格式，不翻译
                if (chunk.startswith('```') or 
                    chunk.startswith('.. code-block::') or
                    chunk.startswith('http') or
                    chunk.startswith('![') or
                    chunk.startswith('.. image::') or
                    len(chunk.split()) < 2):  # 很短的内容
                    print(f"    ⏭️  特殊格式，跳过翻译")
                    translated_chunks.append(chunk)
                else:
                    context = f"文件: {file_path.name}"
                    translated = await self.translate_text(chunk, context)
                    translated_chunks.append(translated)
                
                # 短暂延时
                await asyncio.sleep(0.3)
            
            # 合并翻译结果
            translated_content = '\n\n'.join(translated_chunks)
            
            # 保存翻译文件
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
                
            print(f"  ✅ 翻译完成: {output_file}")
            return True
            
        except Exception as e:
            print(f"  ❌ 翻译失败 {file_path}: {e}")
            return False

    def get_all_files(self) -> List[Path]:
        """获取所有需要翻译的文件"""
        all_files = []
        
        for ext in self.supported_extensions:
            pattern = f"**/*{ext}"
            files = list(self.docs_path.glob(pattern))
            all_files.extend(files)
        
        # 过滤掉不需要翻译的文件
        filtered_files = [f for f in all_files if not self.should_skip_file(f)]
        
        return sorted(filtered_files)

    def create_chinese_conf(self):
        """创建中文文档的配置文件"""
        conf_content = '''# -*- coding: utf-8 -*-
"""
中文文档配置文件
基于原文档配置修改
"""

# 导入原配置
import sys
import os
sys.path.insert(0, os.path.abspath('../docs'))
from conf import *

# 修改项目信息为中文
project = "verl - 中文文档"
copyright = "2024 ByteDance Seed Foundation MLSys Team"
author = "Guangming Sheng, Chi Zhang, Yanghua Peng, Haibin Lin"

# 语言设置
language = "zh_CN"

# HTML配置
html_title = "VERL 中文文档"
html_short_title = "VERL 中文文档"

# GitHub Pages 中文文档配置
html_baseurl = "https://vocabvictor.github.io/verl-ascend/zh/"

# 主题选项
html_theme_options.update({
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
})
'''
        
        conf_file = self.output_path / 'conf.py'
        with open(conf_file, 'w', encoding='utf-8') as f:
            f.write(conf_content)
        
        print(f"✅ 创建中文配置文件: {conf_file}")

    def create_makefile(self):
        """创建中文文档的Makefile"""
        makefile_content = '''# 中文文档 Makefile

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = verl-cn
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile

help:
\t@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
\t@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
'''
        
        makefile = self.output_path / 'Makefile'
        with open(makefile, 'w', encoding='utf-8') as f:
            f.write(makefile_content)
        
        print(f"✅ 创建中文Makefile: {makefile}")

    async def run_translation(self, force: bool = False, check_incomplete: bool = True):
        """执行完整的翻译流程"""
        print("🚀 开始异步文档翻译...")
        print(f"源目录: {self.docs_path}")
        print(f"输出目录: {self.output_path}")
        print(f"API地址: {self.base_url}")
        print(f"强制重新翻译: {'是' if force else '否'}")
        print(f"检查不完整翻译: {'是' if check_incomplete else '否'}")
        print("-" * 50)
        
        # 获取所有文件
        files = self.get_all_files()
        print(f"找到 {len(files)} 个文件需要处理")
        
        if not files:
            print("没有找到需要翻译的文件")
            return
        
        # 如果需要检查不完整翻译，先扫描一遍
        if check_incomplete and not force:
            print("\n📋 检查翻译完整性...")
            incomplete_files = []
            for file_path in files:
                if self.is_translation_incomplete(file_path):
                    incomplete_files.append(file_path)
            
            if incomplete_files:
                print(f"发现 {len(incomplete_files)} 个不完整翻译文件，将重新翻译")
                for f in incomplete_files[:5]:  # 显示前5个
                    print(f"  - {f}")
                if len(incomplete_files) > 5:
                    print(f"  ... 还有 {len(incomplete_files) - 5} 个文件")
        
        # 翻译文件
        success_count = 0
        skip_count = 0
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            
            # 检查是否需要跳过
            if not force and not check_incomplete and self.is_already_translated(file_path):
                print(f"⏭️  跳过已翻译: {file_path}")
                skip_count += 1
                continue
                
            if await self.translate_file(file_path, force):
                success_count += 1
        
        # 创建配置文件
        print(f"\n📝 创建中文文档配置...")
        self.create_chinese_conf()
        self.create_makefile()
        
        # 复制静态文件
        import shutil
        static_src = self.docs_path / '_static'
        if static_src.exists():
            static_dst = self.output_path / '_static'
            if static_dst.exists():
                shutil.rmtree(static_dst)
            shutil.copytree(static_src, static_dst)
            print(f"✅ 复制静态文件: {static_dst}")
        
        print(f"\n🎉 异步翻译完成!")
        print(f"成功翻译: {success_count} 个文件")
        print(f"跳过已有: {skip_count} 个文件")
        print(f"总计处理: {success_count + skip_count}/{len(files)} 个文件")
        print(f"输出目录: {self.output_path}")
        print(f"\n构建中文文档:")
        print(f"cd {self.output_path}")
        print(f"make html")

    async def close(self):
        """关闭客户端连接"""
        await self.httpx_client.aclose()
        await self.openai_client.close()


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='异步翻译文档工具')
    parser.add_argument('--force', action='store_true', help='强制重新翻译所有文件')
    parser.add_argument('--no-check', action='store_true', help='不检查不完整翻译')
    args = parser.parse_args()
    
    translator = None
    try:
        translator = AsyncDocumentTranslator()
        await translator.run_translation(
            force=args.force, 
            check_incomplete=not args.no_check
        )
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请检查:")
        print("1. .env 文件是否存在且配置正确")
        print("2. API密钥是否有效")
        print("3. 网络连接是否正常")
        sys.exit(1)
    finally:
        if translator:
            await translator.close()


if __name__ == "__main__":
    asyncio.run(main())