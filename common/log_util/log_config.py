# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/1/2023 8:02 pm

"""
import os
import inspect


LOGGER_NAME = os.getenv("LOGGER_NAME", "root")
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
REPO = os.getenv("REPO", "/pies/")
FRAME = 3
LOGGER_FORMAT = "%(asctime)s-%(levelname)s-%(message)s"
DATE_FORMAT = "%y-%m-%d %H:%M:%S"
LOG_OUT = os.getenv("LOG_FILE", True)
HANDLER_NUM = 2 if LOG_OUT else 1


def get_frame(which_frame, repo):
    """
    @return: filename and lineno
    """
    callerframerecord = inspect.stack()[which_frame]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    try:
        file, line = info.filename.rsplit(repo)[1], str(info.lineno)
    except:
        file, line = "*", "*"

    return file, line


def get_logfile(fmt="%y-%m-%d-%H:%M"):
    """
    get log file
    @param flag:
    @param fmt:
    @return:
    """
    from datetime import datetime
    dt = datetime.now().strftime(fmt)
    name = "{}.log".format(dt)
    return name








