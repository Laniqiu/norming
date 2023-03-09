# -*- coding: utf-8 -*-
"""
@time: 7/1/2023 5:03 pm

"""
from pathlib import Path

# 数据路径
try:
    from google.colab import drive
    adr = Path("/content/drive/MyDrive/")
    tmp_dir = Path("/content/tmp")
except:
    cur = Path.cwd()
    if cur.home().match("/Users/laniqiu/"):
        adr = Path("/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/dough")
    else:
        adr = Path("/disk/lani/")

    tmp_dir = adr.joinpath("tmp")


from .log_util import MyLogger
logging = MyLogger().get_logger()
