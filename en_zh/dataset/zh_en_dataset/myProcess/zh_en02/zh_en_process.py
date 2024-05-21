"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/5/6-19:20
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import json

def process():
    with open(r'train.txt.en', 'r', encoding='utf-8') as en:
        lines_en = en.readlines()
    with open(r'train.txt.zh', 'r', encoding='utf-8') as zh:
        lines_zh = zh.readlines()

    list_zh,list_en = [],[]
    for en_text in lines_en:
        list_en.append(en_text.split('\n')[0])
    for zh_text in lines_zh:
        list_zh.append(zh_text.split('\n')[0])
    zh_en_dict = {}
    for i in range(len(list_en)):
        zh_en_dict[list_en[i]] = list_zh[i]

    with open('zh_en.json', 'w', encoding='utf-8') as fn:
        json.dump(zh_en_dict,fn,ensure_ascii=False)


if __name__ == '__main__':
    process()
    pass