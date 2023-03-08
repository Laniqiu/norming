# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 27/2/2023 4:24 pm

"""
import requests
# temptwit = requests.get('https://api.stocktwits.com/api/2/messages/show/' + str(twit["id"]) + '.json?access_token=' +
#                         conf["stocktwits"]["api"]["app"]["token"]).json()
tid = "5329774"
token = "fd412811b30cf0ee"
temptwit = requests.get('https://api.stocktwits.com/api/2/messages/show/5329774' + str(tid) + '.json?access_token=' +
                        token).json()
print(temptwit)

if __name__ == "__main__":
    pass
