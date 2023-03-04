# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/3/2023 1:48 pm

"""
# from binder.emb2fea.trainer import baseline
from common.setup import adr
from Jing.extract_sense_emb import collect

# 提取sense emb
_dir = adr.joinpath("Jing")
# collect(_dir.joinpath("samps/batch_3").glob("*.xlsx"),
#         adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"),  # lm模型路径
#         _dir.joinpath("sens_embs/batch_3")  # 单义词sense emb保存目录
#         )

from Jing.report import main

main(adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"),  # 模型路径， str
     _dir.joinpath("bert-target-year-data"),  # 年份文件数据
     _dir.joinpath("sens_embs/batch_3"),  # gt数据
     _dir.joinpath("scores/batch_3"),  # 文件生成保存路径
     )

