# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/1/2023 9:51 am
"""
import logging
import inspect
import os

LOGGER_NAME = os.getenv("LOGGER_NAME", "root")
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
REPO = os.getenv("REPO", "pies")
FRAME = 2
LOG_FILE = os.getenv("LOG_FILE", None)

LOGGER_FORMAT = "%(asctime)s-%(levelname)s-%(message)s"
DATE_FORMAT = "%y-%m-%d %H:%M:%S"

class MyLogger(object):
    # 初始化 Logger
    def __init__(self):
        self.logger = self.set_logger(LOGGER_NAME, LOGGER_LEVEL,
                                      LOGGER_FORMAT, DATE_FORMAT, LOG_FILE)
    @staticmethod
    def set_logger(logger_name, logger_level,
                   logger_format, date_format, file):
        """

        @param logger_name:
        @param logger_level:
        @param logger_format:
        @param date_format:
        @param file:
        @return:
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
        fmt = logging.Formatter(logger_format, datefmt=date_format)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logger_level)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setLevel(logger_level)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        return logger

    def format_msg(self, msg):
        """
        msg: original message
        get filename and lineno of the logging func and add to original msg
        @return:
        """
        try:
            callerframerecord = inspect.stack()[FRAME]
            frame = callerframerecord[0]
            info = inspect.getframeinfo(frame)
            file, line = info.filename.rsplit(REPO)[1], str(info.lineno)
        except:
            file, line = "", ""

        return "{}->{}: {}".format(file, line, msg)

    def debug(self, msg):
        return self.logger.debug(self.format_msg(msg))

    def info(self, msg):

        return self.logger.info(self.format_msg(msg))

    def warning(self, msg):
        return self.logger.warning(self.format_msg(msg))

    def error(self, msg):
        return self.logger.error(self.format_msg(msg))

    def critical(self, msg):
        return self.logger.critical(self.format_msg(msg))











