"""
transfer scanned pdf to searchable pdf
"""
import shutil

import pytesseract
from pdf2image import convert_from_path
import os
from shutil import rmtree
from PyPDF2 import PdfMerger

os.chdir(os.getcwd())

def tess_ocr(fname, out_name="out.txt", lang="eng"):
	# temp folder to save imgs
	dirname = "temp"
	if os.path.exists(dirname):
		rmtree(dirname)
	os.makedirs(dirname)

	#pdf to img
	images = convert_from_path(fname, fmt="png", output_folder=dirname)

	merger = PdfMerger()
	# ocr from img
	length = len(images)
	for i in range(length):
		img = images[i]
		pdf = pytesseract.image_to_pdf_or_hocr(img, extension="pdf", lang=lang)
		pdf_pth = os.path.join(dirname, str(i) + ".pdf")
		with open(pdf_pth, "w+b") as fw:
			fw.write(pdf)
		merger.append(pdf_pth)

	# Write to an output PDF document
	output = open(out_name, "w+b")
	merger.write(output)

	# Close File Descriptors
	merger.close()
	output.close()
	# shutil.rmtree(dirname)
	# os.remove(fname)

fname = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/dough/Jing/in.pdf"
text = tess_ocr(fname, out_name="test.pdf")
