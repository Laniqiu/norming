# -*- coding: utf-8 -*-
"""
@author: Lani QIU
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
        adr = Path("/disk/lani/dough/")

    tmp_dir = adr.joinpath("tmp")


from .log_util import MyLogger
logging = MyLogger().get_logger()
# logging.info("\n\tHome Address: {}\n\tTemp Address: {}".format(adr, tmp_dir))


def move_log(cur_dir, log=None):
    from shutil import move
    import time
    """
    @param project: 
    @return: 
    """
    name = time.strftime("%H:%M-%b-%d", time.localtime()) if not log else log
    move("../logs/log.log", "../../{}/{}.log".format(cur_dir, name))


def log_env(logger):
    """
    record env into log
    @return:
    """
    pv, tv, cv = "", "", ""
    msg = """
    python: {}\n
    torch: {}\n
    cuda: {}\n
    """ % format(pv, tv, cv)

    logger.info(msg)



def del_log():
    from os import remove
    remove("../logs/log.log")
