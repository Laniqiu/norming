# -*- coding:utf-8 -*-

"""
commonly used functions: write, read, etc.
"""
import codecs

def general_reader(fpth, encoding="utf-8", mode="r"):
    """
    open and read file, return a list
    :param fpth:
    :param encoding:
    :param mode:
    :return:
    """

    with codecs.open(fpth, mode=mode) as fr:
        data = fr.readlines()

    return data

def load_json(fpth, mode="r"):
    import json
    with codecs.open(fpth, mode=mode) as fr:
        data = json.load(fr)

    return data

def read_csv_pd(fpth, encoding="utf-8"):
    """
    open csv file with pandas and return a data frame object
    :param fpth:
    :param encoding:
    :return:
    """
    import pandas as pd
    data = pd.read_csv(fpth, encoding=encoding)
    return data

def read_xlsx_pd(fpth, sheet_name=0, cols=None):
    """read xlsx from fpth"""
    import pandas as pd
    data = pd.read_excel(fpth, sheet_name=sheet_name, usecols=cols)
    return data


if __name__ == "__main__":
    pass