# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/1/2023 9:51 am
"""
import logging

from .log_config import *


class MyLogger(object):
    def __init__(self):
        logger = logging.getLogger(LOGGER_NAME)
        if len(logger.handlers) <= HANDLER_NUM:
            logger.setLevel(LOGGER_LEVEL)
            fmt = logging.Formatter(LOGGER_FORMAT, datefmt=DATE_FORMAT)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(LOGGER_LEVEL)
            stream_handler.setFormatter(fmt)
            logger.addHandler(stream_handler)

            if LOG_FILE:
                file_handler = logging.FileHandler(filename=get_logfile(), mode='a')
                file_handler.setLevel(LOGGER_LEVEL)
                file_handler.setFormatter(fmt)
                logger.addHandler(file_handler)

        self.logger = logger


    def format_msg(self, msg):
        """
        msg: original message
        get filename and lineno of the logging func and add to original msg
        @return:
        """
        file, line = get_frame(FRAME, REPO)
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







