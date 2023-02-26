# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 26/2/2023 8:53 pm

"""
import os
from shutil import rmtree
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from glob import glob

fname = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/dough/Jing/效力-蓄洪.pdf"
tmp_dir = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/dough/Jing/tmp"
fout = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/dough/Jing/text.txt"

if os.path.exists(tmp_dir):
    rmtree(tmp_dir)
os.makedirs(tmp_dir)
# pdf to image

tmps = convert_from_path(fname, fmt="png", output_folder=tmp_dir)
imgs = glob(os.path.join(tmp_dir, "*.png"))
with open(fout, "w") as fw:
    for img in imgs:
        # open image
        image = Image.open(img)
        result = pytesseract.image_to_string(image, lang='chi_sim', config="psm 4")
        print("识别结果：", result)
        fw.writelines(result.replace(" ", ""))

rmtree(tmp_dir)
if __name__ == "__main__":
    pass
