# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 5:03 pm

"""

from pathlib import Path

try:
    from google.colab import drive
    adr = Path("/content/drive/MyDrive/")
    tmp_dir = Path("/content/tmp")
except:
    adr = Path("/Users/laniqiu/My Drive")
    tmp_dir = Path("/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity")


from .log_util import MyLogger
logging = MyLogger(adr).get_logger()
logging.info("\n\tHome Address: {}\n\tTemp Address: {}".format(adr, tmp_dir))


