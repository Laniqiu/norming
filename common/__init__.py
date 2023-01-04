from .log_util import logger as logging


def get_root():
    try:
        from google.colab import drive
        return "/content/drive/MyDrive/"
    except:
        return "/Users/laniqiu/My Drive"
