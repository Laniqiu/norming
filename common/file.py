# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 12/1/2023 11:16 pm
文件操作
"""
def lazy_download(file):
    """
    download file from google drive
    @param file:
    @return:
    """
    try:
        from google.colab import files
        files.download(file)
    except:
        print("Oops!")

def move_log(cur_dir):
    from shutil import move
    """
    
    @param project: 
    @return: 
    """
    move("../logs/log.log", "../../{}".format(cur_dir))

