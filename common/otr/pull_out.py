# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 5/2/2023 11:34 am
pull project under pies to independent project
"""

from pathlib import Path
from shutil import copytree, ignore_patterns, move, rmtree
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-p", "--project",
                  type="string", help="project name")
parser.add_option("-t", "--target",
                  type="string", help="target folder")

opt, args = parser.parse_args()


def main(project, target_folder,
         run_file="run.py",
         to_ignore=ignore_patterns("*.pyc", "otr*", "*.ipynb_checkpoints", "*lani*")):
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

    target_dir = Path(target_folder)
    if target_dir.exists():
        rmtree(target_dir)

    com_dir = adr.joinpath("common")
    com_tgt = target_dir.joinpath("common")
    pro_dir = adr.joinpath(project)
    target_pro = target_dir.joinpath(project)
    log_dir = target_dir.joinpath("logs")
    data_dir = target_dir.joinpath("data")
    out_dir = data_dir.joinpath("out")

    copytree(com_dir, com_tgt, ignore=to_ignore)
    copytree(pro_dir, target_pro, ignore=to_ignore)
    # 新建log文件夾
    log_dir.mkdir(exist_ok=True)
    # 新建data文件夹
    data_dir.mkdir(exist_ok=True)
    # 新建输出文件夹
    out_dir.mkdir(exist_ok=True)

    # 修改log_cofig的LOG_FILE
    log_config = com_tgt.joinpath("log_util/log_config.py")
    new = []

    with open(log_config, "r", encoding="utf-8") as fr:
        for line in fr:
            if line.startswith("LOG_FILE"):
                line = line.replace("../logs/log.log", "logs/log.log")
            new.append(line)
    with open(log_config, "w", encoding="utf-8") as fw:
        fw.writelines(new)

    # 修改common.setup，只保留引入logging的部分
    setup_tgt = com_tgt.joinpath("setup.py")
    setup_tgt.unlink()
    setup = ["# -*- coding: utf-8 -*-\n",
           "from .log_util import MyLogger\n",
           "logging = MyLogger().get_logger()\n",
           "adr = None\n"
             ]
    with open(setup_tgt, "w", encoding="utf-8") as fw:
        fw.writelines(setup)

    # 将run.py移到根路径下
    run_src = target_pro.joinpath(run_file)
    run_tgt = target_dir.joinpath(run_file)

    if run_src.exists():
        move(run_src, run_tgt)

    # 移动requirements.txt
    req_file = target_pro.joinpath("requirements.txt")
    req_tgt = target_dir.joinpath("requirements.txt")
    if req_file.exists():
        move(req_file, req_tgt)


main(opt.project, opt.target)