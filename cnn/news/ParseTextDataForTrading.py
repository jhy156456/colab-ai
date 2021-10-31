# import sys
# from pathlib import Path
#
# import pandas as pd
#
# import spacy
# from spacy import displacy
# from textacy.extract import ngrams, entities
# def get_attributes(f):
#     print([a for a in dir(f) if not a.startswith('_')], end=' ')
# nlp = spacy.load('en_core_web_sm')
# sample_text = 'Apple is looking at buying U.K. startup for $1 billion'
# doc = nlp(sample_text)
# get_attributes(doc)


# 네이버 검색 API예제는 블로그를 비롯 전문자료까지 호출방법이 동일하므로 blog검색만 대표로 예제를 올렸습니다.
# 네이버 검색 Open API 예제 - 블로그 검색
import os
import sys
import urllib.request
from bs4 import BeautifulSoup

client_id = "knlQxuT8AkP957KiwCBu"
client_secret = "GtDBKi9ZPQ"
encText = 'sort=sim&start=1&display=2&query='+urllib.parse.quote("삼성전자")
#url = "https://openapi.naver.com/v1/search/blog?query=" + encText # json 결과
url = "https://openapi.naver.com/v1/search/news.xml?" + encText # xml 결과
print(url)

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()

file = open("naver_news.txt", "w", encoding='utf-8')

if(rescode==200):
    response_body = response.read()
    xmlsoup = BeautifulSoup(response_body, 'html.parser')
    items = xmlsoup.find_all('item')
    #print(items)
    for item in items:
        file.write(item.description.get_text(strip=True) + '\n')

    print("Success")
else:
    print("Error Code:" + rescode)
