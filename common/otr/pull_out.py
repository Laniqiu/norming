# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 5/2/2023 11:34 am
pull project under pies to independent project
"""
from shutil import copytree, ignore_patterns


def main(project, target_folder, ignore=ignore_patterns("*.pyc", "otr*")):
    copytree("../../common", target_folder, ignore=ignore)
    copytree("../../{}".format(project), target_folder, ignore=ignore)
    copytree("../../logs".format(project), target_folder, ignore=ignore)
