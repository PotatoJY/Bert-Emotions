# -*- coding: utf-8 -*-
# @Time    : 2024-4-28 15:12
# @Author  : jizhengyu
# @File    : create_hug.py
# @Software: PyCharm
from huggingface_hub import create_repo

# 创建仓库
repo_name = "bert-cls"
repo_type = "model"  # 或者其他类型

# 创建仓库（登录后，token应该已经存储好了）
repo_url = create_repo(repo_id=repo_name, repo_type=repo_type)

print(repo_url)
