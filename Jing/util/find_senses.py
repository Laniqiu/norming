# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 26/2/2023 9:48 pm

"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as etree

def query(content):
    # 请求地址
    url = 'https://baike.baidu.com/item/' + urllib.parse.quote(content)
    # 请求头部
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    # 利用请求地址和请求头部构造请求对象
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    # 发送请求，获得响应
    response = urllib.request.urlopen(req)
    # 读取响应，获得文本
    text = response.read().decode('utf-8')
    # 构造 _Element 对象
    html = etree.ElementTree(text)
    # 使用 xpath 匹配数据，得到匹配字符串列表
    # sen_list = html.xpath('//div[contains(@class,"lemma-summary") or contains(@class,"lemmaWgt-lemmaSummary")]//text()')
    # 过滤数据，去掉空白
    sen_list_after_filter = [item.strip('\n') for item in sen_list]
    # 将字符串列表连成字符串并返回
    return ''.join(sen_list_after_filter)

if __name__ == '__main__':
    while (True):
        # content = input('查询词语：')
        content = "现代汉语"
        result = query(content)
        print("查询结果：%s" % result)