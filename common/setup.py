# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 5:03 pm

"""

from pathlib import Path

try:
    from google.colab import drive
    adr = Path("/content/drive/MyDrive/")
    tmp_dir = Path("/content/tmp")
except:
    adr = Path("/Users/laniqiu/Drive")
    tmp_dir = Path("/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/tmp")


from .log_util import MyLogger
logging = MyLogger().get_logger()
logging.info("\n\tHome Address: {}\n\tTemp Address: {}".format(adr, tmp_dir))


def move_log(cur_dir):
    from shutil import move
    import time
    """
    @param project: 
    @return: 
    """
    name = time.strftime("%H:%M-%b-%d", time.localtime())
    move("../logs/log.log", "../../{}/{}.log".format(cur_dir, name))


def delete_log():
    from os import remove
    remove("../logs/log.log")
