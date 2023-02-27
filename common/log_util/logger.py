# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/1/2023 9:51 am
"""
import logging
from logging.handlers import TimedRotatingFileHandler
import threading
from colorlog import ColoredFormatter

from .log_config import *


class MyLogger(object):

    def __int__(self):
        pass

    def __new__(cls):
        mutex = threading.Lock()
        mutex.acquire()  # 上锁，防止多线程下出问题
        if not hasattr(cls, 'instance'):
            cls.instance = super(MyLogger, cls).__new__(cls)
            cls.instance.log_filename = LOG_FILE  # 日志存放路径

            cls.instance.logger = logging.getLogger(LOGGER_NAME)
            cls.instance.__config_logger()
            cls.instance.logger.setLevel(1)
        mutex.release()
        return cls.instance

    def get_logger(self):
        return self.logger

    def __config_logger(self):
        # 设置日志格式
        plain_formatter = logging.Formatter(PLAIN_FMT, datefmt=DATE_FORMAT)
        corlor_formatter = ColoredFormatter(fmt=COLOR_FMT,
                                            datefmt=DATE_FORMAT,
                                            log_colors=LOG_COLORS,
                                            secondary_log_colors=SECONDARY_COLORS,
                                            style="%")
        """ 控制台日志开关"""
        if LOG_IN_CONSOLE:  # 如果开启控制台日志，非1 关闭
            console = logging.StreamHandler()
            console.setFormatter(corlor_formatter)
            console.setLevel(CONSOLE_LEVEL)
            self.logger.addHandler(console)
            # print(u'当前控制台生效的日志级别为：', self.logger.getEffectiveLevel())

        if LOG_IN_FILE > 0:  # 如果开启文件日志
            rt_file_handler = TimedRotatingFileHandler(self.log_filename, when='D', interval=1,
                                                       backupCount=BACKUP_COUNT)
            rt_file_handler.setFormatter(plain_formatter)
            rt_file_handler.setLevel(FILE_LEVEL)
            self.logger.addHandler(rt_file_handler)






