# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 5:03 pm

"""
def get_root():
    try:
        from google.colab import drive
        return "/content/drive/MyDrive/"
    except:
        return "/Users/laniqiu/My Drive"


# from .log_util import logger as logging

from .log_util import MyLogger
logging = MyLogger(get_root()).get_logger()

