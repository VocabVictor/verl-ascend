# -*- coding: utf-8 -*-
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
