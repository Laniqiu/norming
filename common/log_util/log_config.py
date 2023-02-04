# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/1/2023 8:02 pm
logging 配置
"""
import os

LOGGER_NAME = "root"
LOG_FILE = "logs/log.log"
# 设置
MAX_BYTES = 512000
BACKUP_COUNT = 10

# 等级
# CRITICAL=50 ERROR=40 WARNING=30 INFO=20 DEBUG=10 NOTSET=0'
CONSOLE_LEVEL = 10
FILE_LEVEL = 20
LOG_IN_CONSOLE = True
LOG_IN_FILE = os.getenv("LOG_IN_FILE", True)

PLAIN_FMT = "%(asctime)s-%(levelname)s-%(filename)s-%(lineno)s > %(message)s"
COLOR_FMT = "%(log_color)s%(asctime)s-%(level_log_color)s%(levelname)s%(reset)s" \
      "%(log_color)s-%(filename)s-%(lineno)s > %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_COLORS = {
    'DEBUG': 'white',
    'INFO': 'white',
    'WARNING': 'white',
    'ERROR': 'white',
    'CRITICAL': 'white',
}
# 颜色配置
SECONDARY_COLORS = {
    'level': {
        "INFO": "blue",
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
        'WARNING': 'yellow',
    }
}





