# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 26/2/2023 8:59 pm

"""
import PyPDF2

# 使用open的‘rb’方法打开pdf文件，使用二进制模式
mypdf = open('test.pdf', mode='rb')

# 调用PdfFileReader函数
pdf_document = PyPDF2.PdfReader(mypdf)

# 使用PdfFileReader对象的变量，获取各个信息，如numPages属性获取PDF文档的页数
# pdf_document.numPages

# 调用PdfFileReader对象的getPage()方法，传入页码，取得Page对象：输出PDF文档的第一页内容
first_page = pdf_document.pages[0]

# 调用Page对象的extractText()方法，返回该页文本的字符串
text = first_page.extract_text()
breakpoint()
if __name__ == "__main__":
    pass
