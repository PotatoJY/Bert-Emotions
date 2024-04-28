# -*- coding: utf-8 -*-
# @Time    : 2024-4-27 09:38
# @Author  : jizhengyu
# @File    : draw_distrust.py
# @Software: PyCharm

import matplotlib.pyplot as plt


'''输出类别分布'''
# df["label_name"].value_counts(ascending=True).plot.barh()
# plt.title("Frequency of Classes")
# plt.show()
'''查看类别分布'''
# df["Words Per Tweet"] = df["text"].str.split().apply(len)  # 按空格切分，获取雷彪长度
# df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False,
#            color="black")
# plt.suptitle("")
# plt.xlabel("")
# plt.show()