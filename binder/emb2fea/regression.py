"""
train regression models from Chinese embeddings to the Binder features for the words of your dataset.
steps:
    extract word embeddings
    train regression models
    evaluation
"""

try:
    from google.colab import drive
    # drive.mount('/content/drive')
    _root = "/content/drive/MyDrive/"
except:
    _root = "/Users/laniqiu/My Drive/"

import logging


from utils import *


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)




if __name__ == '__main__':
    from pathlib import Path
    _path = Path(_root)