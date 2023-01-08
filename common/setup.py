# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 5:03 pm

"""


try:
    from google.colab import drive
    adr = "/content/drive/MyDrive/"
except:
    adr = "/Users/laniqiu/My Drive"


from .log_util import MyLogger
logging = MyLogger(adr).get_logger()
logging.info("\nThis is Lani's Little Bakery\nAddress: {}".format(adr))

