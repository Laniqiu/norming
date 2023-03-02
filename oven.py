# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/3/2023 1:48 pm

"""
# from binder.emb2fea.trainer import main
from common.setup import adr

# main(adr.joinpath("binder/new_ratings.xlsx"),  # 输入文件
#      adr.joinpath("lfs"),  # 模型文件夹
#      adr.joinpath("binder/out"),
#      )

# Running on Jing's Desktop
# 已生成或后续需要再处理的target
# skip_folder = ["阶级", "批"] + \
#               ['不断', '中心', '乡', '产业', '介绍', '代表', '企图', '传统', '作', '保障', '党', '内奸', '决定', '决心',
#                '出口', '出现', '分', '创新', '到底', '办', '努力', '协议', '历史', '原则', '反']
# from Jing.report import main
# _dir = adr.joinpath("Jing")
# main(adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"), # 模型路径， str
#     _dir.joinpath("bert-target-year-data"),  # 年份文件数据
#      _dir.joinpath("sens_embs"),  # gt数据
#      _dir.joinpath("scores"),  # 文件生成保存路径
# )


# 已生成或后续需要再处理的target
skip_folder = ["阶级", "批"] + ["建", "节"] + \
              ['不断', '中心', '乡', '产业', '介绍', '代表', '企图', '传统', '作', '保障', '党', '内奸', '决定', '决心',
               '出口', '出现', '分', '创新', '到底', '办', '努力', '协议', '历史', '原则', '反']
from Jing.report import main
_dir = adr.joinpath("Jing")
main(adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"), # 模型路径， str
    _dir.joinpath("bert-target-year-data"),  # 年份文件数据
     _dir.joinpath("sens_embs/batch_2"),  # gt数据
     _dir.joinpath("scores/batch_2"),  # 文件生成保存路径
)
