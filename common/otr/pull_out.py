# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 5/2/2023 11:34 am
pull project under pies to independent project
"""
import os
from pathlib import Path


from shutil import copytree, ignore_patterns


def main(project, target_folder, ignore=ignore_patterns("*.pyc", "otr*")):
    """
    copy common folder, project folder and create log folder
    @param project:
    @param target_folder:
    @param ignore:
    @return:
    """
    cur = Path.cwd()
    if cur.home().match("/Users/laniqiu/"):
        adr = Path("/Users/laniqiu/Drive/pies/")
    else:
        adr = Path("/home/qiule/drive/pies")

    com_dir = adr.joinpath("common")
    pro_dir = adr.joinpath(project)
    log_dir = Path(target_folder).joinpath("logs")

    copytree(com_dir, target_folder, ignore=ignore)
    copytree(pro_dir, target_folder, ignore=ignore)



    # 新建log文件夾

    # 还要修改common.setup的adr
    # 另外要加一个run.py或sh文件
main("", "")